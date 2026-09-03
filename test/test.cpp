#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <stdexcept>

#include <GLFW/glfw3.h>
#include "compute.hpp"
#include "glfw3webgpu.h"
#include "sync.hpp"
#include "webgpu/webgpu.h"

#include <webgpu/noapi.hpp>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

static const char *wgsl_shaders = R"wgsl(
struct Varyings {
	@builtin(position) pos : vec4<f32>,
	@location(0) color : vec3<f32>,
}

const positions = array<vec2<f32>, 3>(
	vec2<f32>(-0.5, -0.5),
	vec2<f32>( 0.5, -0.5),
	vec2<f32>( 0.0,  0.5)
);

const colors = array<vec3<f32>, 3>(
	vec3<f32>(0.0, 0.0, 1.0),
	vec3<f32>(0.0, 1.0, 0.0),
	vec3<f32>(1.0, 0.0, 0.0)
);

@vertex
fn vertex(@builtin(vertex_index) vertex_index : u32) -> Varyings {
	return Varyings(
		vec4<f32>(positions[vertex_index], 0.0, 1.0),
		colors[vertex_index]
	);
}

@fragment
fn fragment(varyings : Varyings) -> @location(0) vec4<f32> {
	return vec4<f32>(varyings.color, 1.0);
})wgsl";


typedef struct AppState {
	GLFWwindow *window;
	GpuWebGPUDefault wgpu;
	GpuQueue* queue;
	WGPUTextureFormat format;
	WGPURenderPipeline pipeline;
	uint32_t width;
	uint32_t height;
} AppState;


typedef struct RequestAdapterResult {
	WGPUAdapter adapter;
	bool done;
} RequestAdapterResult;

static void on_adapter_request(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void *userdata1, void *userdata2) {
	(void)userdata2;
	RequestAdapterResult *result = (RequestAdapterResult *)userdata1;
	result->done = true;
	if (status != WGPURequestAdapterStatus_Success) {
		fprintf(stderr, "Request adapter failed: %.*s\n", (int)message.length, message.data);
		exit(1);
	}
	result->adapter = adapter;
}

typedef struct RequestDeviceResult {
	WGPUDevice device;
	bool done;
} RequestDeviceResult;

static void on_device_request(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void *userdata1, void *userdata2) {
	(void)userdata2;
	RequestDeviceResult *result = (RequestDeviceResult *)userdata1;
	result->done = true;
	if (status != WGPURequestDeviceStatus_Success) {
		fprintf(stderr, "Request device failed: %.*s\n", (int)message.length, message.data);
		exit(2);
	}
	result->device = device;
}

static void on_uncaptured_error(WGPUDevice const *device, WGPUErrorType type, WGPUStringView message, void *userdata1, void *userdata2) {
	(void)device; (void)userdata1; (void)userdata2;
	fprintf(stderr, "WGPU Device Error %d: %.*s\n", (int)type, (int)message.length, message.data);
}

static void configure_surface(AppState *state) {
	WGPUSurfaceConfiguration config = {0};
	config.device = state->wgpu.device;
	config.format = state->format;
	config.usage = WGPUTextureUsage_RenderAttachment;
	config.width = state->width;
	config.height = state->height;
	config.presentMode = WGPUPresentMode_Fifo;
	config.alphaMode = WGPUCompositeAlphaMode_Auto;

	wgpuSurfaceConfigure(state->wgpu.surface, &config);
}

static void on_framebuffer_resize(GLFWwindow *window, int width, int height) {
	AppState *state = (AppState *)glfwGetWindowUserPointer(window);
	state->width = (uint32_t)width;
	state->height = (uint32_t)height;
}


static void render_frame(AppState *state) {
	WGPUSurfaceTexture surface_texture;
	wgpuSurfaceGetCurrentTexture(state->wgpu.surface, &surface_texture);

	if (surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal && surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
		configure_surface(state);
		return;
	}

	WGPUTextureView view = wgpuTextureCreateView(surface_texture.texture, NULL);

	WGPURenderPassColorAttachment color_attachment = {0};
	color_attachment.view = view;
	color_attachment.loadOp = WGPULoadOp_Clear;
	color_attachment.storeOp = WGPUStoreOp_Store;
	color_attachment.clearValue = WGPUColor{0.05, 0.05, 0.08, 1.0};
	color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

	WGPURenderPassDescriptor rp_descriptor = {0};
	rp_descriptor.colorAttachmentCount = 1;
	rp_descriptor.colorAttachments = &color_attachment;

	WGPUCommandEncoderDescriptor encoder_descriptor = {0};
	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(state->wgpu.device, &encoder_descriptor);

	WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(encoder, &rp_descriptor);
	wgpuRenderPassEncoderSetPipeline(render_pass, state->pipeline);
	wgpuRenderPassEncoderDraw(render_pass, 3, 1, 0, 0);
	wgpuRenderPassEncoderEnd(render_pass);
	wgpuRenderPassEncoderRelease(render_pass);

	WGPUCommandBufferDescriptor cb_descriptor = {0};
	WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cb_descriptor);
	wgpuCommandEncoderRelease(encoder);

	wgpuQueueSubmit(state->queue->queue, 1, &commands);
	wgpuCommandBufferRelease(commands);

	wgpuTextureViewRelease(view);
	wgpuTextureRelease(surface_texture.texture);

#ifndef __EMSCRIPTEN__
	wgpuSurfacePresent(state->wgpu.surface);
#endif
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
	state.window = glfwCreateWindow(initial_width, initial_height, "WebGPU Triangle", NULL, NULL);
	assert(state.window && "Failed to create glfw window");

	glfwSetWindowUserPointer(state.window, &state);
	glfwSetFramebufferSizeCallback(state.window, on_framebuffer_resize);

	auto wg = gpuSetupDefaultWebGPUEXT([&state](WGPUInstance instance) -> WGPUSurface {
		auto surface =  glfwCreateWindowWGPUSurface(instance, state.window);
		if(!surface) throw std::runtime_error("Failed to setup WebGPU surface");
		return surface;
	});
	if(!wg) throw std::runtime_error(wg.error());
	state.wgpu = *wg;

	state.queue = gpuCreateQueue(state.wgpu);

	auto upload = gpuMalloc<float>(state.queue, 5, MEMORY_DEFAULT);
	for(size_t i = 0; i < 5; ++i)
		upload[i] = i;
	auto upload_gpu = gpuHostToDevicePointer(state.queue, upload);
	auto upload_range = std::get<GpuQueue::MonobufferRange>(state.queue->allocations[upload_gpu]);
	auto download = gpuMalloc<float>(state.queue, 5, MEMORY_READBACK);
	auto download_gpu = gpuHostToDevicePointer(state.queue, download);
	auto download_range = std::get<GpuQueue::MonobufferRange>(state.queue->allocations[download_gpu]);

	auto pipe = gpuCreateComputePipeline(state.queue, string_to_bytes(R"wgsl(
		@generated_noapi_bindings

		const GPU_ADDRESS_MAX_HI : u32 = 0x1FFFFFFFu; // lower 29 bits

		struct DecodedAddress {
			monobuffer: u32,
			address: vec2<u32>,
		};

		fn gpuEncodeAddress(monobuffer: u32, address: vec2<u32>) -> vec2<u32> {
			let tag = monobuffer + 1u;

			return vec2<u32>(
				address.x,
				(address.y & GPU_ADDRESS_MAX_HI) | (tag << 29u)
			);
		}

		// Decodes the monobuffer and recovers the original 61-bit address.
		fn gpuDecodeAddress(encoded: vec2<u32>) -> DecodedAddress {
			let tag = encoded.y >> 29u;

			return DecodedAddress(
				tag - 1u,
				vec2<u32>(
					encoded.x,
					encoded.y & GPU_ADDRESS_MAX_HI
				)
			);
		}

		fn loadMonobuffer(address: vec2<u32>) -> u32 {
			let buf = gpuDecodeAddress(address);
			switch buf.monobuffer {
				case 1: {
					return mono1[buf.address.x / 4];
				}
				case 2: {
					return mono2[buf.address.x / 4];
				}
				case 3: {
					return mono3[buf.address.x / 4];
				}
				case 4: {
					return mono4[buf.address.x / 4];
				}
				case 5: {
					return mono5[buf.address.x / 4];
				}
				default: {
					return mono0[buf.address.x / 4];
				}
			}
		}

		fn storeMonobuffer(address: vec2<u32>, value: u32) {
			let buf = gpuDecodeAddress(address);
			switch buf.monobuffer {
				case 1: {
					mono1[buf.address.x / 4] = value;
				}
				case 2: {
					mono2[buf.address.x / 4] = value;
				}
				case 3: {
					mono3[buf.address.x / 4] = value;
				}
				case 4: {
					mono4[buf.address.x / 4] = value;
				}
				case 5: {
					mono5[buf.address.x / 4] = value;
				}
				default: {
					mono0[buf.address.x / 4] = value;
				}
			}
		}

		// End Prologue

		@compute @workgroup_size(16)
		fn main(@builtin(global_invocation_id) global_id : vec3u) {
			var u = shader_data.upload_buffer;
			u.x += global_id.x * 4;
			var d = shader_data.download_buffer;
			d.x += global_id.x * 4;
			let tmp = loadMonobuffer(u);
			storeMonobuffer(d, tmp * 5);
		}
	)wgsl"));

	auto cmd = gpuStartCommandRecording(state.queue);
	gpuSyncMemoryEXT(cmd, upload_gpu);
	WGPUBufferDescriptor d {
		.label = {"Temporary Storage", WGPU_STRLEN},
		.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
		.size = 5 * sizeof(float),
	};
	auto tmp = wgpuDeviceCreateBuffer(state.queue->device, &d);
	wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, state.queue->monobuffers[upload_range.buffer], upload_range.start, tmp, 0, upload_range.size());
	wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, tmp, 0, state.queue->monobuffers[download_range.buffer], download_range.start, upload_range.size());
	gpuSyncMemoryEXT(cmd, download_gpu);
	auto index = gpuSubmit(state.queue, {&cmd, 1});
	gpuWaitSemaphore(state.queue, gpuGetSubmissionSemaphoreEXT(state.queue), index);

	auto dbg = download[3];
	wgpuBufferRelease(tmp);

	gpuFreePipeline(state.queue, pipe);

	WGPUSurfaceCapabilities caps = {0};
	wgpuSurfaceGetCapabilities(state.wgpu.surface, state.wgpu.adapter, &caps);
	state.format = caps.formats[0]; // preferred format first in the list
	wgpuSurfaceCapabilitiesFreeMembers(caps);

	{
		int fb_width, fb_height;
		glfwGetFramebufferSize(state.window, &fb_width, &fb_height);
		state.width = (uint32_t)fb_width;
		state.height = (uint32_t)fb_height;
	}
	configure_surface(&state);

	WGPUShaderModule shader;
	{
		WGPUShaderSourceWGSL wgsl_source = {0};
		wgsl_source.chain.sType = WGPUSType_ShaderSourceWGSL;
		wgsl_source.code = {wgsl_shaders, WGPU_STRLEN};

		WGPUShaderModuleDescriptor shader_desc = {0};
		shader_desc.nextInChain = &wgsl_source.chain;
		shader = wgpuDeviceCreateShaderModule(state.wgpu.device, &shader_desc);
	}
	assert(shader && "Shader compilation failed");

	{
		WGPUBlendState blend = {};
		blend.color.operation = WGPUBlendOperation_Add;
		blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
		blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
		blend.alpha.operation = WGPUBlendOperation_Add;
		blend.alpha.srcFactor = WGPUBlendFactor_One;
		blend.alpha.dstFactor = WGPUBlendFactor_Zero;

		WGPUColorTargetState color_target = {0};
		color_target.format = state.format;
		color_target.blend = &blend;
		color_target.writeMask = WGPUColorWriteMask_All;

		WGPUFragmentState fragment = {0};
		fragment.module = shader;
		fragment.entryPoint = {"fragment", WGPU_STRLEN};
		fragment.constantCount = 0;
		fragment.targetCount = 1;
		fragment.targets = &color_target;

		WGPURenderPipelineDescriptor descriptor = {0};
		descriptor.vertex.module = shader;
		descriptor.vertex.entryPoint = {"vertex", WGPU_STRLEN};
		descriptor.vertex.bufferCount = 0;
		descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
		descriptor.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
		descriptor.primitive.frontFace = WGPUFrontFace_CCW;
		descriptor.primitive.cullMode = WGPUCullMode_None;
		descriptor.multisample.count = 1;
		descriptor.multisample.mask = 0xFFFFFFFF;
		descriptor.multisample.alphaToCoverageEnabled = false;
		descriptor.fragment = &fragment;

		state.pipeline = wgpuDeviceCreateRenderPipeline(state.wgpu.device, &descriptor);
	}
	assert(state.pipeline && "Pipeline creation failed");

	wgpuShaderModuleRelease(shader);

#ifdef __EMSCRIPTEN__
	emscripten_set_main_loop_arg(emscripten_frame, &state, 0, true);
#else
	while (!glfwWindowShouldClose(state.window)) {
		glfwPollEvents();
		render_frame(&state);
	}

	gpuWaitIdleEXT(state.queue);

	wgpuRenderPipelineRelease(state.pipeline);
	gpuFreeQueue(state.queue);
	wgpuSurfaceRelease(state.wgpu.surface);
	wgpuDeviceRelease(state.wgpu.device);
	wgpuAdapterRelease(state.wgpu.adapter);
	wgpuInstanceRelease(state.wgpu.instance);

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