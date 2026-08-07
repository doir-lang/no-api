#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <GLFW/glfw3.h>
#include "glfw3webgpu.h"
#include <webgpu/webgpu.h>

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
	WGPUInstance instance;
	WGPUDevice device;
	WGPUQueue queue;
	WGPUSurface surface;
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
	config.device = state->device;
	config.format = state->format;
	config.usage = WGPUTextureUsage_RenderAttachment;
	config.width = state->width;
	config.height = state->height;
	config.presentMode = WGPUPresentMode_Fifo;
	config.alphaMode = WGPUCompositeAlphaMode_Auto;

	wgpuSurfaceConfigure(state->surface, &config);
}

static void on_framebuffer_resize(GLFWwindow *window, int width, int height) {
	AppState *state = (AppState *)glfwGetWindowUserPointer(window);
	state->width = (uint32_t)width;
	state->height = (uint32_t)height;
}


static void render_frame(AppState *state) {
	WGPUSurfaceTexture surface_texture;
	wgpuSurfaceGetCurrentTexture(state->surface, &surface_texture);

	if (surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal && surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
		fprintf(stderr, "Surface texture unavailable status=%d\n", (int)surface_texture.status);
		configure_surface(state);
		return;
	}

	WGPUTextureView view = wgpuTextureCreateView(surface_texture.texture, NULL);

	WGPURenderPassColorAttachment color_attachment = {0};
	color_attachment.view = view;
	color_attachment.loadOp = WGPULoadOp_Clear;
	color_attachment.storeOp = WGPUStoreOp_Store;
	color_attachment.clearValue = (WGPUColor){0.05, 0.05, 0.08, 1.0};
	color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

	WGPURenderPassDescriptor rp_descriptor = {0};
	rp_descriptor.colorAttachmentCount = 1;
	rp_descriptor.colorAttachments = &color_attachment;

	WGPUCommandEncoderDescriptor encoder_descriptor = {0};
	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(state->device, &encoder_descriptor);

	WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(encoder, &rp_descriptor);
	wgpuRenderPassEncoderSetPipeline(render_pass, state->pipeline);
	wgpuRenderPassEncoderDraw(render_pass, 3, 1, 0, 0);
	wgpuRenderPassEncoderEnd(render_pass);
	wgpuRenderPassEncoderRelease(render_pass);

	WGPUCommandBufferDescriptor cb_descriptor = {0};
	WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cb_descriptor);
	wgpuCommandEncoderRelease(encoder);

	wgpuQueueSubmit(state->queue, 1, &commands);
	wgpuCommandBufferRelease(commands);

	wgpuTextureViewRelease(view);
	wgpuTextureRelease(surface_texture.texture);

#ifndef __EMSCRIPTEN__
	wgpuSurfacePresent(state->surface);
#endif
}

#ifdef __EMSCRIPTEN__
static void emscripten_frame(void *arg) {
	AppState *state = (AppState *)arg;
	glfwPollEvents();
	render_frame(state);
}
#endif

/* -------------------------------------------------------------------------
* Entry point
* ---------------------------------------------------------------------- */
#ifdef _WIN32
#include <windows.h>
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpCmdLine, int nCmdShow)
#else
int main(void)
#endif
{
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

	state.instance = wgpuCreateInstance(NULL);
	assert(state.instance && "Failed to create webgpu instance");

	WGPUSurface surface = glfwCreateWindowWGPUSurface(state.instance, state.window);
	assert(surface && "Failed to create webgpu surface");
	state.surface = surface;

	WGPUAdapter adapter = NULL;
	{
		WGPURequestAdapterOptions adapter_opts = {0};
		adapter_opts.compatibleSurface = state.surface;
		adapter_opts.powerPreference = WGPUPowerPreference_HighPerformance;

		RequestAdapterResult result = {0};
		WGPURequestAdapterCallbackInfo callback_info = {0};
		callback_info.mode = WGPUCallbackMode_AllowSpontaneous;
		callback_info.callback = on_adapter_request;
		callback_info.userdata1 = &result;

		wgpuInstanceRequestAdapter(state.instance, &adapter_opts, callback_info);
		while (!result.done) {
			wgpuInstanceProcessEvents(state.instance);
		#ifdef __EMSCRIPTEN__
			emscripten_sleep(1); // yields back to the browser event loop
		#endif
		}
		adapter = result.adapter;
	}
	assert(adapter && "No adapter obtained");

	{
		WGPUDeviceDescriptor device_desc = {0};
		device_desc.uncapturedErrorCallbackInfo.callback = on_uncaptured_error;

		RequestDeviceResult result = {0};
		WGPURequestDeviceCallbackInfo callback_info = {0};
		callback_info.mode = WGPUCallbackMode_AllowSpontaneous;
		callback_info.callback = on_device_request;
		callback_info.userdata1 = &result;

		wgpuAdapterRequestDevice(adapter, &device_desc, callback_info);
		while (!result.done) {
			wgpuInstanceProcessEvents(state.instance);
		#ifdef __EMSCRIPTEN__
			emscripten_sleep(1); // yields back to the browser event loop
		#endif
		}
		state.device = result.device;
	}
	assert(state.device && "No device obtained");

	state.queue = wgpuDeviceGetQueue(state.device);
	WGPUSurfaceCapabilities caps = {0};
	wgpuSurfaceGetCapabilities(state.surface, adapter, &caps);
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
		shader = wgpuDeviceCreateShaderModule(state.device, &shader_desc);
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

		state.pipeline = wgpuDeviceCreateRenderPipeline(state.device, &descriptor);
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

	wgpuRenderPipelineRelease(state.pipeline);
	wgpuQueueRelease(state.queue);
	wgpuSurfaceRelease(state.surface);
	wgpuDeviceRelease(state.device);
	wgpuAdapterRelease(adapter);
	wgpuInstanceRelease(state.instance);

	glfwDestroyWindow(state.window);
	glfwTerminate();
#endif

	return 0;
}