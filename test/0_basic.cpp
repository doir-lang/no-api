#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include <numeric>
#include <print>
#include <stdexcept>
#include <string>

#ifdef NOAPI_BACKEND_VULKAN
	#include <volk.h>
	#define GLFW_INCLUDE_VULKAN
#endif

#include <GLFW/glfw3.h>

#ifdef NOAPI_BACKEND_VULKAN
	#include <vulkan/noapi.hpp>
#else
	#include <webgpu/noapi.hpp>
	#include "glfw3webgpu.h"
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif



static const std::string slang_compute = R"slang(
import noapi;
import test_constants; // Registered by the program with gpuAddSlangModuleEXT

// As many elements as the C++ side allocated. The dispatch covers a whole workgroup, so the
// threads past the end have to be sent home rather than left to trample memory that was never
// theirs.
static const uint ELEMENT_COUNT = 5;

// The root data struct the dispatch was handed: {float* upload, float* download}. Conforming it
// to IGpuLoadable is what lets the shader read one in a single `*gpuComputeData<ComputeData>()`
// -- the layout says how wide it is and how to walk it, and everything underneath is the one 32
// bit load the backend provides.
struct ComputeData : IGpuLoadable {
	GpuPtr<float> upload;
	GpuPtr<float> download;

	static const uint gpuStride = 16;

	static ComputeData gpuLoad(GpuAddress at) {
		var from = GpuReader(at); // Walks the fields in order, so no offset is written down
		ComputeData out;
		out.upload = from.next<GpuPtr<float> >();
		out.download = from.next<GpuPtr<float> >();
		return out;
	}
}

[shader("compute")]
[numthreads(16, 1, 1)]
void computeMain(uint3 thread : SV_DispatchThreadID) {
	if(thread.x >= ELEMENT_COUNT) return;

	let data = *gpuComputeData<ComputeData>();
	data.download[thread.x] = data.upload[thread.x] * gpuTestScaleFactor();
}
)slang";

static const std::string slang_triangle = R"slang(
import noapi;

struct Varyings {
	float4 position : SV_Position;
	float3 color : COLOR;
}

// One entry of the vertex array, matching Vertex on the C++ side: five floats, {x, y, r, g, b}.
// Nothing writes a vertex, so the layout is read only -- IGpuStorable and a gpuStore beside it
// would be what a shader that writes one adds, and on a WebGPU graphics stage they could not
// exist anyway, since the monobuffers are bound read only there.
struct Vertex : IGpuLoadable {
	float2 position;
	float3 color;

	static const uint gpuStride = 20;

	static Vertex gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		Vertex out;
		out.position = from.next<float2>();
		out.color = from.next<float3>();
		return out;
	}
}

[shader("vertex")]
Varyings vertexMain(uint index : SV_VertexID) {
	// The root data is one pointer, so the pointer to it is a pointer to a pointer
	let vertices = *gpuVertexData<GpuPtr<Vertex> >();
	let vertex = vertices[index];

	Varyings output;
	output.position = float4(vertex.position, 0.0, 1.0);
	output.color = vertex.color;
	return output;
}

// The fragment stage never touches the root data it was handed; it just interpolates
[shader("fragment")]
float4 fragmentMain(Varyings varyings) : SV_Target {
	return float4(varyings.color, 1.0);
}
)slang";

static const std::string slang_user_module = R"slang(
module test_constants;

public float gpuTestScaleFactor() { return 6.0; }
)slang";

#ifdef NOAPI_BACKEND_VULKAN

	using GpuBackendDefault = GpuVulkanDefault;

	static GpuBackendDefault setup_backend(GLFWwindow* window) {
		auto vk = gpuSetupDefaultVulkanEXT([window](VkInstance instance) -> VkSurfaceKHR {
			VkSurfaceKHR surface;
			if(glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
				throw std::runtime_error("Failed to setup Vulkan surface");
			return surface;
		});
		if(!vk) throw std::runtime_error(vk.error());
		return *vk;
	}

	static void shutdown_backend(GpuBackendDefault& vulkan) {
		vkDestroyDevice(vulkan.device, nullptr);
		vkDestroySurfaceKHR(vulkan.instance, vulkan.surface, nullptr);
		if(vulkan.messenger) vkDestroyDebugUtilsMessengerEXT(vulkan.instance, vulkan.messenger, nullptr);
		vkDestroyInstance(vulkan.instance, nullptr);
		volkFinalize();
	}

#else

	using GpuBackendDefault = GpuWebGPUDefault;

	static GpuBackendDefault setup_backend(GLFWwindow* window) {
		auto wg = gpuSetupDefaultWebGPUEXT([window](WGPUInstance instance) -> WGPUSurface {
			auto surface = glfwCreateWindowWGPUSurface(instance, window);
			if(!surface) throw std::runtime_error("Failed to setup WebGPU surface");
			return surface;
		});
		if(!wg) throw std::runtime_error(wg.error());
		return *wg;
	}

	static void shutdown_backend(GpuBackendDefault& wgpu) {
		wgpuSurfaceRelease(wgpu.surface);
		wgpuDeviceRelease(wgpu.device);
		wgpuAdapterRelease(wgpu.adapter);
		wgpuInstanceRelease(wgpu.instance);
	}

#endif


extern "C" uint16_t noapi_c_compat_packed_default_sampler(void);

typedef struct AppState {
	GLFWwindow *window;
	GpuBackendDefault backend;
	GpuQueue* queue;
	GpuSurface* surface;
	GpuPipeline* pipeline;
	gpu* triangle; // Root data struct: a pointer to the vertex array
	gpu* indices;
	uint32_t width;
	uint32_t height;
} AppState;

struct Vertex {
	float x, y;
	float r, g, b;
};


static void resize_surface(AppState *state) {
	auto desc = gpuSurfaceGetConfigurationEXT(state->surface);
	desc.texture.dimensions = {state->width, state->height, 1};
	gpuSurfaceReconfigureEXT(state->queue, state->surface, desc);
}

static void on_framebuffer_resize(GLFWwindow *window, int width, int height) {
	AppState *state = (AppState *)glfwGetWindowUserPointer(window);
	state->width = width;
	state->height = height;
}


static void render_frame(AppState *state) {
	// A minimized window has nothing to present, and a surface can't be configured for a zero extent
	if (state->width == 0 || state->height == 0) return;

	auto configured = gpuSurfaceGetConfigurationEXT(state->surface).texture.dimensions;
	if (configured.x != state->width || configured.y != state->height)
		resize_surface(state);

	auto target = gpuSurfaceNextTextureEXT(state->queue, state->surface);
	if (!target) { // Out of date, lost or timed out: reconfigure and try again next frame
		resize_surface(state);
		return;
	}

	GpuColorAttachment color {
		.texture = target,
		.loadOp = LOAD_OP_CLEAR,
		.clearValue = {0.05f, 0.05f, 0.08f, 1.0f},
	};
	GpuRenderPassDesc pass {
		.colorAttachments = {&color, 1},
	};

	auto cmd = gpuStartCommandRecording(state->queue);
	gpuBeginRenderPass(cmd, pass);
	gpuSetPipeline(cmd, state->pipeline);
	// The same root data serves both stages; the fragment shader simply never reads it
	gpuDrawIndexedInstanced(cmd, state->triangle, state->triangle, state->indices, 3, 1);
	gpuEndRenderPass(cmd, pass);

	auto submission = gpuSubmit(state->queue, {&cmd, 1});
	gpuSurfacePresentEXT(state->queue, state->surface, submission);
}

#ifdef __EMSCRIPTEN__
static void emscripten_frame(void *arg) {
	AppState *state = (AppState *)arg;
	glfwPollEvents();
	render_frame(state);
}
#endif

int real_main() {
	assert(noapi_c_compat_packed_default_sampler() == GpuSamplerDesc{}.pack() && "C and C++ disagree about GpuSamplerDesc");

	AppState state = {0};
	const int initial_width = 800;
	const int initial_height = 600;

	if (!glfwInit()) {
		std::println(std::cerr, "Glfw initialization failed\n");
		return -1;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#ifdef NOAPI_BACKEND_VULKAN
	state.window = glfwCreateWindow(initial_width, initial_height, "NoAPI Triangle (vulkan)", NULL, NULL);
#elifdef NOAPI_BACKEND_WEBGPU
	state.window = glfwCreateWindow(initial_width, initial_height, "NoAPI Triangle (webgpu)", NULL, NULL);
#else
	state.window = glfwCreateWindow(initial_width, initial_height, "NoAPI Triangle (unknown)", NULL, NULL);
#endif
	assert(state.window && "Failed to create glfw window");

	glfwSetWindowUserPointer(state.window, &state);
	glfwSetFramebufferSizeCallback(state.window, on_framebuffer_resize);

	// Shaders are compiled inside the pipeline creation calls, so anything they import has to be
	// registered first. This also routes the compiler's diagnostics somewhere visible.
	gpuSetShaderDiagnosticCallbackEXT([](GpuStringView message, void*) {
		std::println("[shader] {}", std::string_view(message.ptr, message.count));
	}, nullptr);
	gpuAddShaderModuleEXT("test_constants", GpuStringView{slang_user_module.data(), slang_user_module.size()});

	state.backend = setup_backend(state.window);
	state.queue = gpuCreateQueue(state.backend);

	auto upload = gpuMalloc<float>(state.queue, 5, MEMORY_DEFAULT);
	for(size_t i = 0; i < 5; ++i)
		upload[i] = i;

	auto download = gpuMalloc<float>(state.queue, 5, MEMORY_READBACK);

	struct ShaderData {
		gpu* upload;
		gpu* download;
	};
	auto data = gpuMalloc<ShaderData>(state.queue);
	data->upload = gpuHostToDevicePointer(state.queue, upload);
	data->download = gpuHostToDevicePointer(state.queue, download);
	auto data_gpu = gpuHostToDevicePointer(state.queue, data);
	gpuSyncMemoryEXT(state.queue, data_gpu);

	auto pipe = gpuCreateComputePipeline(state.queue, string_to_bytes(slang_compute));

	{
		auto cmd = gpuStartCommandRecording(state.queue);
		gpuSyncMemoryEXT(cmd, data->upload);
		gpuSetPipeline(cmd, pipe);
		gpuDispatch(cmd, data_gpu, {1, 1, 1});
		gpuSyncMemoryEXT(cmd, data->download);
		auto index = gpuSubmit(state.queue, {&cmd, 1});
		gpuWaitSemaphore(state.queue, gpuGetSubmissionSemaphoreEXT(state.queue), index);
	}

	std::println("upload: {}, download: {}", upload[3], download[3]);

	gpuFreePipeline(state.queue, pipe);

	{
		int fb_width, fb_height;
		glfwGetFramebufferSize(state.window, &fb_width, &fb_height);
		state.width = (uint32_t)fb_width;
		state.height = (uint32_t)fb_height;
	}

	state.surface = gpuCreateSurfaceEXT(state.queue, state.backend, GpuSurfaceDescriptor{
		.texture = {
			.dimensions = {state.width, state.height, 1},
			.format = FORMAT_NONE, // Whatever the surface prefers, until the capabilities are known
			.usage = USAGE_COLOR_ATTACHMENT,
		},
		.presentMode = PRESENT_MODE_FIFO,
	});

	auto caps = gpuGetSurfaceCapabilitiesEXT(state.queue, state.surface);
	std::print("surface formats (best first):");
	for(auto format: caps.formatList())
		std::print(" {}{}", (int)format, gpuFormatIsSrgbEXT(format) ? " (srgb)" : "");
	std::print("\npresent modes (best first):");
	for(auto mode: caps.presentModeList())
		std::print(" {}", (int)mode);
	std::println("\ntransparency: {}", caps.supportsTransparency);

	GpuSurfaceDescriptor config;
	for(auto format: caps.formatList())
		if(!gpuFormatIsSrgbEXT(format)) {
			config = gpuSurfaceGetConfigurationEXT(state.surface);
			config.texture.format = format;
			gpuSurfaceReconfigureEXT(state.queue, state.surface, config);
			break;
		}

	std::println("surface: {}x{}, format {}, present mode {}", config.texture.dimensions.x, config.texture.dimensions.y, (int)config.texture.format, (int)config.presentMode);

	auto vertices = gpuMalloc<Vertex>(state.queue, 3);
	vertices[0] = {-0.5f, -0.5f, 0.0f, 0.0f, 1.0f};
	vertices[1] = { 0.5f, -0.5f, 0.0f, 1.0f, 0.0f};
	vertices[2] = { 0.0f,  0.5f, 1.0f, 0.0f, 0.0f};

	auto indices = gpuMalloc<uint32_t>(state.queue, 3);
	std::iota(indices, indices + 3, 0);

	struct TriangleData {
		gpu* vertices;
	};
	auto triangle = gpuMalloc<TriangleData>(state.queue);
	triangle->vertices = gpuHostToDevicePointer(state.queue, vertices);
	state.triangle = gpuHostToDevicePointer(state.queue, triangle);
	state.indices = gpuHostToDevicePointer(state.queue, indices);

	gpuSyncMemoryEXT(state.queue, triangle->vertices);
	gpuSyncMemoryEXT(state.queue, state.indices);
	gpuSyncMemoryEXT(state.queue, state.triangle);

	GpuColorTarget target { .format = config.texture.format };
	state.pipeline = gpuCreateGraphicsPipeline(state.queue, string_to_bytes(slang_triangle), string_to_bytes(slang_triangle), GpuRasterDesc{
		.colorTargets = {&target, 1},
		.blendstate = GpuBlendDesc{
			.srcColorFactor = FACTOR_SRC_ALPHA,
			.dstColorFactor = FACTOR_ONE_MINUS_SRC_ALPHA,
		},
	});
	assert(state.pipeline && "Pipeline creation failed");

#ifdef __EMSCRIPTEN__
	emscripten_set_main_loop_arg(emscripten_frame, &state, 0, true);
#else
	while (!glfwWindowShouldClose(state.window)) {
		glfwPollEvents();
		render_frame(&state);
	}

	gpuWaitIdleEXT(state.queue);

	gpuFreePipeline(state.queue, state.pipeline);
	gpuFreeSurfaceEXT(state.queue, state.surface);
	gpuFreeQueue(state.queue);
	shutdown_backend(state.backend);

	glfwDestroyWindow(state.window);
	glfwTerminate();
#endif

	return 0;
}




#ifdef _WIN32
	#include <windows.h>
#endif

#ifdef _WIN32
	int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
		AllocConsole();
		freopen("CONOUT$", "w", stdout);
		freopen("CONOUT$", "w", stderr);
		freopen("CONIN$", "r", stdin);
		return real_main();
	}
#else
	int main() {
		return real_main();
	}
#endif
