#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <numeric>
#include <print>
#include <stdexcept>
#include <string>

// Which backend noapi::noapi resolved to is published by the CMake target, so an application only has
// to pick a shading language and a way of handing the backend its window.
#ifdef NOAPI_BACKEND_VULKAN
	#include <volk.h>
	#define GLFW_INCLUDE_VULKAN
#endif

#include <GLFW/glfw3.h>

#ifdef NOAPI_BACKEND_VULKAN
	#include <vulkan/noapi.hpp>

	// The Vulkan backend takes SPIR-V, so the test brings its own GLSL compiler
	#include <glslang/Public/ShaderLang.h>
	#include <glslang/Public/ResourceLimits.h>
	#include <glslang/SPIRV/GlslangToSpv.h>
	#include <vector>
#else
	#include <webgpu/noapi.hpp>
	#include "glfw3webgpu.h"
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif


#ifdef NOAPI_BACKEND_VULKAN

	// The prologues the backend exports declare the push constant block holding whatever root pointers
	// the dispatch or draw was given — pc.compute_data for a dispatch, pc.vertex_data/fragment_data/
	// index_data for a draw — along with the sampler lookup helpers. A pointer is a plain device address
	// here, so walking one is an ordinary buffer reference cast rather than a generated helper.

	// Multiplies an array of floats by six, through the two pointers its root data struct holds
	static const std::string glsl_compute = R"glsl(
	layout(buffer_reference, std430, buffer_reference_align = 4) buffer Floats {
		float data[];
	};

	layout(buffer_reference, std430, buffer_reference_align = 8) buffer ComputeData {
		Floats upload;
		Floats download;
	};

	// As many elements as the C++ side allocated. The dispatch covers a whole workgroup, so the threads
	// past the end have to be sent home rather than left to trample memory that was never theirs.
	const uint ELEMENT_COUNT = 5;

	layout(local_size_x = 16) in;

	void main() {
		if(gl_GlobalInvocationID.x >= ELEMENT_COUNT) return;

		ComputeData data = ComputeData(pc.compute_data);
		Floats u = data.upload;
		Floats d = data.download;

		d.data[gl_GlobalInvocationID.x] = u.data[gl_GlobalInvocationID.x] * 6;
	}
	)glsl";

	// There are no vertex buffers — the vertex shader loads each vertex out of the array its root data
	// points at.
	static const std::string glsl_vertex = R"glsl(
	layout(buffer_reference, std430, buffer_reference_align = 4) buffer Vertices {
		float data[];
	};

	layout(buffer_reference, std430, buffer_reference_align = 8) buffer TriangleData {
		Vertices vertices;
	};

	// One entry of that array is {x, y, r, g, b}: five floats, matching Vertex on the C++ side
	const uint VERTEX_STRIDE = 5;

	layout(location = 0) out vec3 frag_color;

	void main() {
		Vertices verts = TriangleData(pc.vertex_data).vertices;
		uint at = gl_VertexIndex * VERTEX_STRIDE;

		// Vulkan's clip space points the opposite way down the Y axis to WebGPU's, so the same vertices
		// have to be flipped to land the triangle the same way up
		gl_Position = vec4(verts.data[at], -verts.data[at + 1], 0.0, 1.0);
		frag_color = vec3(verts.data[at + 2], verts.data[at + 3], verts.data[at + 4]);
	}
	)glsl";

	// The fragment stage never touches the root data it was handed; it just interpolates
	static const std::string glsl_fragment = R"glsl(
	layout(location = 0) in vec3 frag_color;
	layout(location = 0) out vec4 out_color;

	void main() {
		out_color = vec4(frag_color, 1.0);
	}
	)glsl";

	using GpuBackendDefault = GpuVulkanDefault;

	// Prepends the prologue the backend expects and hands the result to glslang. The pipeline is built
	// from the words rather than the source, so each blob outlives the call it was passed to.
	static std::vector<uint32_t> compile_glsl(EShLanguage stage, const std::string& body) {
		static const bool initialized = glslang::InitializeProcess();
		assert(initialized && "glslang initialization failed");

		auto source = std::string(stage == EShLangCompute ? COMPUTE_SHADER_PROLOGUE : GRAPHICS_SHADER_PROLOGUE) + body;
		const char* source_array = source.c_str();

		glslang::TShader shader(stage);
		shader.setStrings(&source_array, 1);
		shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 460);
		shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_4);
		shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);

		auto messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules | EShMsgDefault);
		if(!shader.parse(GetDefaultResources(), 460, false, messages))
			throw std::runtime_error(std::string("glslang parse error:\n") + shader.getInfoLog() + shader.getInfoDebugLog());

		glslang::TProgram program;
		program.addShader(&shader);
		if(!program.link(messages))
			throw std::runtime_error(std::string("glslang link error:\n") + program.getInfoLog() + program.getInfoDebugLog());

		std::vector<uint32_t> spirv;
		glslang::SpvOptions options;
		options.validate = true;
		glslang::GlslangToSpv(*program.getIntermediate(stage), spirv, &options);
		return spirv;
	}

	static std::span<const std::byte> compute_shader() {
		static const auto spirv = compile_glsl(EShLangCompute, glsl_compute);
		return byte_span<uint32_t>(spirv);
	}

	static std::span<const std::byte> vertex_shader() {
		static const auto spirv = compile_glsl(EShLangVertex, glsl_vertex);
		return byte_span<uint32_t>(spirv);
	}

	static std::span<const std::byte> fragment_shader() {
		static const auto spirv = compile_glsl(EShLangFragment, glsl_fragment);
		return byte_span<uint32_t>(spirv);
	}

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

	// Everything gpuSetupDefaultVulkanEXT created, destroyed once the queue built on top of it is gone
	static void shutdown_backend(GpuBackendDefault& vulkan) {
		vkDestroyDevice(vulkan.device, nullptr);
		vkDestroySurfaceKHR(vulkan.instance, vulkan.surface, nullptr);
		if(vulkan.messenger) vkDestroyDebugUtilsMessengerEXT(vulkan.instance, vulkan.messenger, nullptr);
		vkDestroyInstance(vulkan.instance, nullptr);
		volkFinalize();
	}

#else

	// @generated_noapi_bindings expands into the monobuffer, texture and sampler declarations, the
	// shader_data uniform holding whatever root pointers the dispatch or draw was given, and the helpers
	// for walking one of those pointers: gpuPtrOffset, gpuLoadU32/F32/Ptr, and (compute only, since the
	// rasterizer never gets the monobuffers writable) gpuStoreU32/F32.

	// Multiplies an array of floats by six, through the two pointers its root data struct holds
	static const std::string wgsl_compute = R"wgsl(
	@generated_noapi_bindings

	// As many elements as the C++ side allocated. The dispatch covers a whole workgroup, so the threads
	// past the end have to be sent home rather than left to trample the rest of the monobuffer.
	const ELEMENT_COUNT : u32 = 5u;

	@compute @workgroup_size(16)
	fn main(@builtin(global_invocation_id) global_id : vec3u) {
		if (global_id.x >= ELEMENT_COUNT) { return; }

		let sizeof_f32 : u32 = 4;
		let data = shader_data.compute;
		let u = gpuPtrOffset(gpuLoadPtr(data), global_id.x * sizeof_f32);
		let d = gpuPtrOffset(gpuLoadPtr(gpuPtrOffset(data, 8u)), global_id.x * sizeof_f32);

		gpuStoreF32(d, gpuLoadF32(u) * 6);
	}
	)wgsl";

	// Both stages live in one blob: the pipeline leaves the entry point undefined, so each stage picks the
	// only one it finds. There are no vertex buffers — the vertex shader loads each vertex out of the array
	// its root data points at.
	static const std::string wgsl_triangle = R"wgsl(
	@generated_noapi_bindings

	struct Varyings {
		@builtin(position) position : vec4<f32>,
		@location(0) color : vec3<f32>,
	}

	// One entry of that array is {x, y, r, g, b}: five floats, matching Vertex on the C++ side
	const VERTEX_STRIDE : u32 = 20u;

	@vertex
	fn vertex(@builtin(vertex_index) index : u32) -> Varyings {
		let sizeof_f32 : u32 = 4;
		let vertices = gpuLoadPtr(shader_data.vertex);
		let at = gpuPtrOffset(vertices, index * VERTEX_STRIDE);

		return Varyings(
			vec4<f32>(gpuLoadF32(at), gpuLoadF32(gpuPtrOffset(at, 1 * sizeof_f32)), 0.0, 1.0),
			vec3<f32>(
				gpuLoadF32(gpuPtrOffset(at, 2 * sizeof_f32)),
				gpuLoadF32(gpuPtrOffset(at, 3 * sizeof_f32)),
				gpuLoadF32(gpuPtrOffset(at, 4 * sizeof_f32))
			)
		);
	}

	@fragment
	fn fragment(varyings : Varyings) -> @location(0) vec4<f32> {
		return vec4<f32>(varyings.color, 1.0);
	}
	)wgsl";

	using GpuBackendDefault = GpuWebGPUDefault;

	static std::span<const std::byte> compute_shader() { return string_to_bytes(wgsl_compute); }
	static std::span<const std::byte> vertex_shader() { return string_to_bytes(wgsl_triangle); }
	static std::span<const std::byte> fragment_shader() { return string_to_bytes(wgsl_triangle); }

	static GpuBackendDefault setup_backend(GLFWwindow* window) {
		auto wg = gpuSetupDefaultWebGPUEXT([window](WGPUInstance instance) -> WGPUSurface {
			auto surface = glfwCreateWindowWGPUSurface(instance, window);
			if(!surface) throw std::runtime_error("Failed to setup WebGPU surface");
			return surface;
		});
		if(!wg) throw std::runtime_error(wg.error());
		return *wg;
	}

	// Everything gpuSetupDefaultWebGPUEXT created, released once the queue built on top of it is gone
	static void shutdown_backend(GpuBackendDefault& wgpu) {
		wgpuSurfaceRelease(wgpu.surface);
		wgpuDeviceRelease(wgpu.device);
		wgpuAdapterRelease(wgpu.adapter);
		wgpuInstanceRelease(wgpu.instance);
	}

#endif


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

// What the vertex shader walks. None of this layout is declared to the API; the shader loads the
// fields it wants out of the pointer it was handed.
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

	// An acquired texture keeps the size it was configured with, so a resize has to be answered before
	// the frame rather than after it
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
	AppState state = {0};
	const int initial_width = 800;
	const int initial_height = 600;

	if (!glfwInit()) {
		fprintf(stderr, "Glfw initialization failed\n");
		return -1;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	state.window = glfwCreateWindow(initial_width, initial_height, "NoAPI Triangle", NULL, NULL);
	assert(state.window && "Failed to create glfw window");

	glfwSetWindowUserPointer(state.window, &state);
	glfwSetFramebufferSizeCallback(state.window, on_framebuffer_resize);

	state.backend = setup_backend(state.window);
	state.queue = gpuCreateQueue(state.backend);

	//
	// Compute: multiply five floats by six on the GPU and read them back
	//
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

	auto pipe = gpuCreateComputePipeline(state.queue, compute_shader());

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

	//
	// The presentation surface
	//
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

	auto caps = gpuGetSurfaceCapabilities(state.queue, state.surface);
	std::print("surface formats (best first):");
	for(auto format: caps.formats)
		std::print(" {}{}", (int)format, gpuFormatIsSrgb(format) ? " (srgb)" : "");
	std::print("\npresent modes (best first):");
	for(auto mode: caps.presentModes)
		std::print(" {}", (int)mode);
	std::println("\ntransparency: {}", caps.supportsTransparency);

	GpuSurfaceDescriptor config;
	for(auto format: caps.formats)
		if(!gpuFormatIsSrgb(format)) {
			config = gpuSurfaceGetConfigurationEXT(state.surface);
			config.texture.format = format;
			gpuSurfaceReconfigureEXT(state.queue, state.surface, config);
			break;
		}

	// What the surface settled on, which is not necessarily what was asked for
	std::println("surface: {}x{}, format {}, present mode {}", config.texture.dimensions.x, config.texture.dimensions.y, (int)config.texture.format, (int)config.presentMode);

	//
	// Graphics: the triangle's vertices and indices, and a pipeline targeting the surface's format
	//
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

	// None of it ever changes, so one push into the monobuffers is enough
	gpuSyncMemoryEXT(state.queue, triangle->vertices);
	gpuSyncMemoryEXT(state.queue, state.indices);
	gpuSyncMemoryEXT(state.queue, state.triangle);

	GpuColorTarget target { .format = config.texture.format };
	state.pipeline = gpuCreateGraphicsPipeline(state.queue, vertex_shader(), fragment_shader(), GpuRasterDesc{
		// .topology = TOPOLOGY_TRIANGLE_LIST,
		// .cull = CULL_NONE,
		.colorTargets = {&target, 1},
		// Baked in rather than applied dynamically, so the pipeline never gets rebuilt for it
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
