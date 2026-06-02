#include <print>
// #include <iostream>
#include <cassert>
#include "glfw3webgpu.h"
#include <webgpu/webgpu_cpp.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/html5.h>
#endif

#include "callback_adapter.hpp"

constexpr std::string_view wgsl_shaders = R"~wgsl~(
@vertex
fn vertex(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
	let frame_uv = frag_pos.xy / vec2<f32>(800, 600);
	return vec4<f32>(frame_uv.x, frame_uv.y, 1 - frame_uv.x * frame_uv.y, 1);
}
)~wgsl~";

constexpr std::array<float, 6> vertices {
	0, .6,
	-.6, -.6,
	.6, -.6
};

#ifdef _WIN32
#include <Windows.h>
INT WINAPI WinMain(HINSTANCE hInts, HINSTANCE previous, LPSTR, INT)
#else
int main()
#endif
{
	size_t constexpr width = 800;
	size_t constexpr height = 600;

	if(!glfwInit()) {
		// std::cerr << "Glfw initialization failed" << std::endl;
		std::println(stderr, "Glfw initialization failed");
		return -1;
	}
	struct DeferGLFWUninit {
		~DeferGLFWUninit() {
			glfwTerminate();
		}
	} defer_glfw_uninit;

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	GLFWwindow* window = glfwCreateWindow(width, height, "WebGPU Triangle", nullptr, nullptr);
	assert(window && "Failed to create glfw window");
	struct DeferWindowDeletion {
		GLFWwindow* window;
		~DeferWindowDeletion() {
			glfwDestroyWindow(window);
		}
	} defer_window_deletion = {window};

	wgpu::Instance instance = wgpu::CreateInstance();
	assert(instance && "Failed to create webgpu instance");

	wgpu::Surface temporary = wgpu::Surface::Acquire(glfwCreateWindowWGPUSurface(instance.Get(), window));
	assert(temporary && "Failed to create webgpu surface");

	wgpu::Adapter adapter; // Vulkan calls an adapter a PhysicalDevice
	{
		wgpu::RequestAdapterOptions opts;
		opts.compatibleSurface = temporary;
		opts.powerPreference = wgpu::PowerPreference::HighPerformance;

		struct RequestHelper {
			wgpu::Adapter& adapter;
			bool found;
		} helper = {adapter, false};

		instance.RequestAdapter(&opts, wgpu::CallbackMode::AllowSpontaneous, [](wgpu::RequestAdapterStatus status, wgpu::Adapter result, wgpu::StringView msg, RequestHelper* helper) {
			helper->found = true;
			if(status != wgpu::RequestAdapterStatus::Success) {
				exit(1);
			}
			helper->adapter = std::move(result);
		}, &helper);
#ifdef __EMSCRIPTEN__
		while(helper.found == false) emscripten_sleep(1);
#endif
	}
	assert(adapter && "No adapter obtained");

	wgpu::Device device;
	{
		wgpu::DeviceDescriptor descriptor;
		descriptor.label = "triangle device";
		descriptor.defaultQueue.label = "triangle queue";

		descriptor.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView msg) {
			std::println(stderr, "WGPU Device Error {}: {}", (int)type, msg.data);
		});

		struct RequestHelper {
			wgpu::Device& device;
			bool found;
		} helper = {device, false};
		adapter.RequestDevice(&descriptor, wgpu::CallbackMode::AllowSpontaneous, [](wgpu::RequestDeviceStatus status, wgpu::Device result, wgpu::StringView msg, RequestHelper* helper){
			helper->found = true;
			if(status != wgpu::RequestDeviceStatus::Success) {
				std::println(stderr, "Request device failed: {}", msg.data);
				exit(2);
			}
			helper->device = std::move(result);
		}, &helper);
#ifdef __EMSCRIPTEN__
		while(helper.found == false) emscripten_sleep(1);
#endif
	}
	assert(device && "No device obtained");

	wgpu::Queue queue = device.GetQueue();
	wgpu::Surface surface = std::move(temporary);

	wgpu::SurfaceConfiguration surface_config;
	surface_config.device = device;
	surface_config.format = wgpu::TextureFormat::BGRA8Unorm;
	surface_config.width = width;
	surface_config.height = height;
	surface_config.presentMode = wgpu::PresentMode::Fifo;
	surface.Configure(&surface_config);

	wgpu::ShaderModule shader;
	{
		wgpu::ShaderSourceWGSL code;
		code.code = {wgsl_shaders.data(), wgsl_shaders.size()};

		wgpu::ShaderModuleDescriptor descriptor;
		descriptor.nextInChain = &code;
		descriptor.label = "triangle shaders";
		shader = device.CreateShaderModule(&descriptor);
	}
	assert(shader && "Shader compilation failed");

	wgpu::Buffer vertex_buffer;
	{
		wgpu::BufferDescriptor descriptor;
		descriptor.label = "triangle buffer";
		descriptor.size = vertices.size() * sizeof(vertices[0]);
		descriptor.usage = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst;
		descriptor.mappedAtCreation = false;
		vertex_buffer = device.CreateBuffer(&descriptor);
	}
	assert(vertex_buffer && "Vertex buffer allocation failed");
	queue.WriteBuffer(vertex_buffer, 0, vertices.data(), vertices.size() * sizeof(vertices[0]));

	wgpu::RenderPipeline pipeline;
	{
		wgpu::VertexAttribute attribute;
		attribute.shaderLocation = 0;
		attribute.format = wgpu::VertexFormat::Float32x2;
		attribute.offset = 0;

		wgpu::VertexBufferLayout layout;
		layout.arrayStride = 2 * sizeof(float);
		layout.attributeCount = 1;
		layout.attributes = &attribute;

		wgpu::BlendState blend;
		blend.color.operation = wgpu::BlendOperation::Add;
		blend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
		blend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
		// pixel = pixel * (1 - src.alpha) + (src.alpha) * src.color
		blend.alpha.operation = wgpu::BlendOperation::Add;
		blend.alpha.srcFactor = wgpu::BlendFactor::One;
		blend.alpha.dstFactor = wgpu::BlendFactor::Zero;

		wgpu::ColorTargetState color_target;
		color_target.format = surface_config.format;
		color_target.blend = &blend;

		wgpu::FragmentState fragment;
		fragment.module = shader;
		fragment.entryPoint = "fragment";
		fragment.constantCount = 0;
		fragment.targetCount = 1;
		fragment.targets = &color_target;

		wgpu::RenderPipelineDescriptor descriptor;
		descriptor.label = "triangle pipeline";
		descriptor.vertex.module = shader;
		descriptor.vertex.entryPoint = "vertex";
		descriptor.vertex.bufferCount = 1;
		descriptor.vertex.buffers = &layout;
		descriptor.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
		descriptor.primitive.stripIndexFormat = wgpu::IndexFormat::Undefined;
		descriptor.primitive.frontFace = wgpu::FrontFace::CCW;
		descriptor.primitive.cullMode = wgpu::CullMode::None;
		// descriptor.depthStencil = nullptr;
		descriptor.fragment = &fragment;

		pipeline = device.CreateRenderPipeline(&descriptor);
	}
	assert(pipeline && "Pipeline creation failed");

#ifdef __EMSCRIPTEN__
	callback_adapter tick { [&] {
#else
	while(!glfwWindowShouldClose(window)) {
#endif
		glfwPollEvents();

		wgpu::SurfaceTexture surface_texture;
		surface.GetCurrentTexture(&surface_texture);
		if(surface_texture.status > wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
			std::println(stderr, "Surface texture unavailable status={}", (int)surface_texture.status);
#ifdef __EMSCRIPTEN__
			return;
#else
			break;
#endif
		}

		wgpu::TextureView view = surface_texture.texture.CreateView();
		wgpu::RenderPassColorAttachment color_attachment;
		color_attachment.view = view;
		color_attachment.loadOp = wgpu::LoadOp::Clear;
		color_attachment.storeOp = wgpu::StoreOp::Store;
		color_attachment.clearValue = {.05, .05, .08, 1};

		wgpu::RenderPassDescriptor rp_descriptor;
		rp_descriptor.label = "triangle render pass";
		rp_descriptor.colorAttachmentCount = 1;
		rp_descriptor.colorAttachments = &color_attachment;

		wgpu::CommandEncoderDescriptor encoder_descriptor;
		encoder_descriptor.label = "frame encoder";
		wgpu::CommandEncoder encoder = device.CreateCommandEncoder(&encoder_descriptor);
		wgpu::RenderPassEncoder render_pass = encoder.BeginRenderPass(&rp_descriptor);
		{
			render_pass.SetPipeline(pipeline);
			render_pass.SetVertexBuffer(0, vertex_buffer);
			render_pass.Draw(3);
		}
		render_pass.End();
		wgpu::CommandBufferDescriptor cb_descriptor;
		cb_descriptor.label = "frame commands";
		wgpu::CommandBuffer commands = encoder.Finish(&cb_descriptor);
		queue.Submit(1, &commands);

#ifdef __EMSCRIPTEN__
	} };
	emscripten_set_main_loop_arg(decltype(tick)::trailing_user_data, &tick, 0, true);
#else
		surface.Present();
	}
#endif

	return 0;
}
