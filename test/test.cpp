#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <volk.h>
#include <VkBootstrap.h>

#include <vulkan/noapi.hpp>
#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <print>
#include <iostream>

static constexpr uint32_t WIDTH = 800;
static constexpr uint32_t HEIGHT = 600;
static constexpr int FRAMES_IN_FLIGHT = 2;

static const char* vertex_glsl = R"glsl(
	#version 450

	layout(location = 0) out vec3 frag_color;

	vec2 positions[3] = vec2[](
		vec2(-0.5, 0.5),
		vec2( 0.5, 0.5),
		vec2( 0.0, -0.5)
	);

	vec3 colors[3] = vec3[](
		vec3(0.0, 0.0, 1.0),
		vec3(0.0, 1.0, 0.0),
		vec3(1.0, 0.0, 0.0)
	);

	void main() {
		gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
		frag_color = colors[gl_VertexIndex];
	}
)glsl";

static const char* fragment_glsl = R"glsl(
	#version 450

	layout(location = 0) in vec3 frag_color;
	layout(location = 0) out vec4 out_color;

	void main() {
		out_color = vec4(frag_color, 1.0);
	}
)glsl";

struct render_state {
	GLFWwindow* window = {};

	VkInstance instance = {};
	VkDebugUtilsMessengerEXT debug_messenger = {};
	VkSurfaceKHR raw_surface = {};
	GpuQueue* graphics_queue = {};

	GpuSurface* surface = nullptr;
	std::vector<VkSemaphore> swapchain_render_finished_semaphore = {};

	VkRenderPass render_pass = {};
	GpuPipeline* pipeline = nullptr;
	std::vector<VkFramebuffer> framebuffers;

	VkCommandPool command_pool = {};
	std::array<VkCommandBuffer, FRAMES_IN_FLIGHT> command_buffers = {};
	std::array<VkSemaphore, FRAMES_IN_FLIGHT> image_available_semaphore = {};
	std::array<VkFence, FRAMES_IN_FLIGHT> render_fence = {};
	size_t current_frame = 0;
};

void ensure_glslang_initialized() {
	static bool glslang_initialized = false;

	if (!glslang_initialized) {
		if (!glslang::InitializeProcess())
			throw std::runtime_error("glslang::InitializeProcess() failed");
		glslang_initialized = true;
	}
}

static std::vector<uint32_t> compile_glsl(EShLanguage stage, const char* source) {
	ensure_glslang_initialized();

	glslang::TShader shader(stage);
	shader.setStrings(&source, 1);
	shader.setEnvInput (glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
	shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
	shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);

	const TBuiltInResource* resources = GetDefaultResources();
	EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules | EShMsgDefault);

	if (!shader.parse(resources, 460, false, messages)) {
		std::string err = "glslang parse error:\n";
		err += shader.getInfoLog();
		err += shader.getInfoDebugLog();
		throw std::runtime_error(err);
	}

	glslang::TProgram program;
	program.addShader(&shader);
	if (!program.link(messages)) {
		std::string err = "glslang link error:\n";
		err += program.getInfoLog();
		err += program.getInfoDebugLog();
		throw std::runtime_error(err);
	}

	std::vector<uint32_t> spv;
	glslang::SpvOptions spvOpts;
	spvOpts.validate = true;
	glslang::GlslangToSpv(*program.getIntermediate(stage), spv, &spvOpts);
	return spv;
}

static VkShaderModule create_shader_module(render_state& state, const std::vector<uint32_t>& spv) {
	VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = spv.size() * sizeof(uint32_t),
		.pCode = spv.data(),
	};
	VkShaderModule out;
	VK_CHECK(vkCreateShaderModule(state.graphics_queue->device, &info, nullptr, &out), nullptr);
	return out;
}

void create_framebuffers(render_state& state) {
	state.framebuffers.resize(state.surface->swapchain->image_count);
	for (size_t i = 0; i < state.surface->swapchain->image_count; ++i) {
		VkFramebufferCreateInfo fbInfo = {
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = state.render_pass,
			.attachmentCount = 1,
			.pAttachments = &state.surface->image_views[i],
			.width = state.surface->swapchain->extent.width,
			.height = state.surface->swapchain->extent.height,
			.layers = 1,
		};
		VK_CHECK(vkCreateFramebuffer(state.graphics_queue->device, &fbInfo, nullptr, &state.framebuffers[i]), /*nothing*/);
	}
}

void init_vulkan(render_state& state) {
	auto vk = gpuSetupDefaultVulkan([&state](VkInstance instance) -> VkSurfaceKHR{
		VkSurfaceKHR out;
		VK_CHECK(glfwCreateWindowSurface(instance, state.window, nullptr, &out), nullptr);
		return out;
	});
	if(!vk) throw std::runtime_error(vk.error());
	state.instance = vk->instance;
	state.debug_messenger = vk->messenger;
	state.raw_surface = vk->surface;

	auto queue = gpuCreateQueue(*vk);
	if(!queue) throw std::runtime_error("Failed to create noapi queue");
	state.graphics_queue = queue;

	state.surface = gpuCreateSurfaceEXT(queue, state.raw_surface, GpuSurfaceDescriptor{
		.texture = {
			.dimensions = {WIDTH, HEIGHT, 1},
			.format = FORMAT_RGBA8_UNORM
		}
	});

	auto upload = gpuMalloc<float>(state.graphics_queue, 16);
	gpu* upload_gpu = gpuHostToDevicePointer(state.graphics_queue, upload);
	for(size_t i = 0; i < 16; ++i)
		upload[i] = i;

	auto download = gpuMalloc<float>(state.graphics_queue, 16, MEMORY_READBACK);
	gpu* download_gpu = gpuHostToDevicePointer(state.graphics_queue, download);

	auto texture_heap = gpuMalloc<GpuTextureDescriptor>(state.graphics_queue);
	auto texture_heap_gpu = gpuHostToDevicePointer(state.graphics_queue, texture_heap);

	GpuTextureDesc descriptor {
		.dimensions = {WIDTH, HEIGHT, 1},
		.format = FORMAT_RGBA8_UNORM,
		.usage = USAGE_STORAGE,
	};
	auto texture_gpu = (gpu*)gpuMalloc(state.graphics_queue, gpuTextureSizeAlign(state.graphics_queue, descriptor));
	auto texture = gpuCreateTexture(state.graphics_queue, descriptor, texture_gpu);
	*texture_heap = gpuRWTextureViewDescriptor(state.graphics_queue, texture, {});

	auto glsl = compile_glsl(EShLangCompute, R"glsl(
		#version 460
		#extension GL_EXT_shader_explicit_arithmetic_types : require
		#extension GL_EXT_buffer_reference : require

		const uint ADDRESS_MODE_CLAMP = 0;
		const uint ADDRESS_MODE_MIRROR_REPEAT = 1;
		const uint ADDRESS_MODE_REPEAT = 2;

		const uint FILTER_NEAREST = 0;
		const uint FILTER_LINEAR = 1;

		struct GpuSamplerDesc {
			uint address_mode_u; // CLAMP, REPEAT, MIRROR_REPEAT
			uint address_mode_v; // CLAMP, REPEAT, MIRROR_REPEAT
			uint address_mode_w; // CLAMP, REPEAT, MIRROR_REPEAT
			uint mag_filter; // NEAREST, LINEAR
			uint min_filter; // NEAREST, LINEAR
			uint mip_filter; // NEAREST, LINEAR
		};

		uint gpuPackSamplerDesc(const GpuSamplerDesc d) {
			return (d.address_mode_u) 
			| (d.address_mode_v << 2) 
			| (d.address_mode_w << 4) 
			| (d.mag_filter << 6) 
			| (d.min_filter << 7) 
			| (d.mip_filter << 8);
		}

		GpuSamplerDesc gpuDefaultSampler() {
			return GpuSamplerDesc(ADDRESS_MODE_REPEAT, ADDRESS_MODE_REPEAT, ADDRESS_MODE_REPEAT, FILTER_LINEAR, FILTER_LINEAR, FILTER_LINEAR);
		}

		layout(buffer_reference, std430) buffer GpuSamplerMap {
			uint data[];
		};

		layout(push_constant) uniform PushConstants {
			uint64_t compute_data;
			GpuSamplerMap sampler_map;
		} pc;

		uint gpuGetSamplerIndex(const GpuSamplerDesc desc) {
			return pc.sampler_map.data[gpuPackSamplerDesc(desc)];
		}

		// End prologue

		#extension GL_EXT_nonuniform_qualifier : require
		#extension GL_EXT_descriptor_heap : require

		layout(local_size_x = 16) in;

		layout(descriptor_heap, rgba8) uniform image2D images[];
		layout(descriptor_heap) uniform sampler2D samplers[];

		layout(buffer_reference, std430) buffer Floats {
			float data[];
		};

		void main() {
			Floats floats = Floats(pc.compute_data);

			uint i = gl_GlobalInvocationID.x;
			floats.data[i] *= 5;
			imageStore(images[0], ivec2(0), vec4(0));
			samplers[gpuGetSamplerIndex(gpuDefaultSampler())];
		}
	)glsl");

	GpuPipeline* pipeline = gpuCreateComputePipeline(state.graphics_queue, byte_span<uint32_t>(glsl));
	GpuCommandBuffer* cmd = gpuStartCommandRecording(state.graphics_queue);

	gpuSetActiveTextureHeapPtr(cmd, texture_heap_gpu, true);
	gpuSetEnabledSamplersEXT(cmd, {});
	gpuSetPipeline(cmd, pipeline);
	gpuDispatch(cmd, upload_gpu, {1, 1, 1});
	gpuMemCpy(cmd, download_gpu, upload_gpu, 16 * sizeof(float));

	auto submission_index = gpuSubmit(state.graphics_queue, {&cmd, 1});
	gpuWaitSemaphore(state.graphics_queue, gpuGetSubmissionSemaphoreEXT(state.graphics_queue), submission_index);

	auto dbg_up = upload[5];
	auto dbg_down = download[5];
	gpuFreePipeline(state.graphics_queue, pipeline);
	gpuFree(state.graphics_queue, upload);
	gpuFree(state.graphics_queue, download);

	gpuFree(state.graphics_queue, texture_gpu);
	gpuFree(state.graphics_queue, texture_heap);

	{ // create_swapchain()
		VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
		state.swapchain_render_finished_semaphore.resize(state.surface->swapchain->image_count);
		for (auto& sema : state.swapchain_render_finished_semaphore)
			VK_CHECK(vkCreateSemaphore(state.graphics_queue->device, &semaphore_info, nullptr, &sema), /*nothing*/);
	}
	{ // create_render_pass();
		VkAttachmentDescription color_attachment = {
			.format = state.surface->swapchain->image_format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		};

		VkAttachmentReference color_reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

		VkSubpassDescription subpass = {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &color_reference,
		};

		VkSubpassDependency dep = {
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		};

		VkRenderPassCreateInfo rpInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &color_attachment,
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = 1,
			.pDependencies = &dep,
		};
		VK_CHECK(vkCreateRenderPass(state.graphics_queue->device, &rpInfo, nullptr, &state.render_pass), /*nothing*/);
	}
	{ // create_pipeline(state);
		auto vertex_spirv = compile_glsl(EShLangVertex, vertex_glsl);
		auto fragment_spirv = compile_glsl(EShLangFragment, fragment_glsl);

		ColorTarget target { .format = FORMAT_RGBA8_UNORM };
		state.pipeline = gpuCreateGraphicsPipeline(state.graphics_queue, byte_span<uint32_t>(vertex_spirv), byte_span<uint32_t>(fragment_spirv), {
			.colorTargets = {&target, 1}
		});
	}
	create_framebuffers(state);
	{ // create_command_pool();
		VkCommandPoolCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = state.graphics_queue->queue_family,
		};
		VK_CHECK(vkCreateCommandPool(state.graphics_queue->device, &info, nullptr, &state.command_pool), /*nothing*/);
	}{ // allocate_command_buffers();
		VkCommandBufferAllocateInfo info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = state.command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = FRAMES_IN_FLIGHT,
		};
		VK_CHECK(vkAllocateCommandBuffers(state.graphics_queue->device, &info, state.command_buffers.data()), /*nothing*/);
	}{ // create_sync_objects();
		VkSemaphoreCreateInfo sema{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
		VkFenceCreateInfo fence{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT
		};

		for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
			VK_CHECK(vkCreateSemaphore(state.graphics_queue->device, &sema, nullptr, &state.image_available_semaphore[i]), /*nothing*/);
			VK_CHECK(vkCreateFence (state.graphics_queue->device, &fence, nullptr, &state.render_fence[i]), /*nothing*/);
		}
	}
}

void recreate_swapchain(render_state& state, uvec2 extent) {
	gpuWaitIdleEXT(state.graphics_queue);

	auto config = gpuSurfaceGetConfigurationEXT(state.surface);
	config.texture.dimensions = {extent.x, extent.y, 1};
	gpuSurfaceReconfigureEXT(state.graphics_queue, state.surface, config);

	for(auto framebuffer: state.framebuffers)
		vkDestroyFramebuffer(state.graphics_queue->device, framebuffer, nullptr);
	create_framebuffers(state);
}

void cleanup(render_state& state) {
	for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
		vkDestroySemaphore(state.graphics_queue->device, state.image_available_semaphore[i], nullptr);
		vkDestroyFence (state.graphics_queue->device, state.render_fence[i], nullptr);
	}
	vkDestroyCommandPool (state.graphics_queue->device, state.command_pool, nullptr);
	for (auto sema : state.swapchain_render_finished_semaphore)
		vkDestroySemaphore(state.graphics_queue->device, sema, nullptr);
	for (auto framebuffer: state.framebuffers)
		vkDestroyFramebuffer(state.graphics_queue->device, framebuffer, nullptr);
	gpuFreeSurfaceEXT(state.graphics_queue, state.surface);
	// vkDestroyPipeline (state.graphics_queue->device, state.pipeline, nullptr);
	// vkDestroyPipelineLayout(state.graphics_queue->device, state.pipeline_layout, nullptr);
	gpuFreePipeline(state.graphics_queue, state.pipeline);
	vkDestroyRenderPass (state.graphics_queue->device, state.render_pass, nullptr);
	gpuFreeQueue(state.graphics_queue);
	vkDestroyDevice(state.graphics_queue->device, nullptr);
	vkDestroySurfaceKHR (state.instance, state.raw_surface, nullptr);
	vkDestroyDebugUtilsMessengerEXT(state.instance, state.debug_messenger, nullptr);
	vkDestroyInstance(state.instance, nullptr);
	volkFinalize();
	glfwDestroyWindow(state.window);
	glfwTerminate();
	glslang::FinalizeProcess();
}

void draw_frame(render_state& state) {
	size_t frame_index = state.current_frame % FRAMES_IN_FLIGHT;
	VkFence fence = state.render_fence[frame_index];
	vkWaitForFences(state.graphics_queue->device, 1, &fence, VK_TRUE, UINT64_MAX);

	int w = 0, h = 0;
	glfwGetFramebufferSize(state.window, &w, &h);
	while (w == 0 || h == 0) {
		glfwGetFramebufferSize(state.window, &w, &h);
		glfwWaitEvents();
	}
	uvec2 needed_size = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
	auto surface_size = gpuSurfaceGetConfigurationEXT(state.surface).texture.dimensions;
	if(surface_size.x != w && surface_size.y != h)
		recreate_swapchain(state, needed_size);

	auto texture = gpuSurfaceNextTextureEXT(state.graphics_queue, state.surface);
	if(errno == SURFACE_SUBOPTIMAL || errno == SURFACE_OUT_OF_DATE) {
		recreate_swapchain(state, needed_size);
		return;
	}

	vkResetFences(state.graphics_queue->device, 1, &fence);

	VkCommandBuffer cmd = state.command_buffers[frame_index];
	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo cmd_begin = {};
	cmd_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin), /*nothing*/);

	VkClearValue clear_color{{{0.0f, 0.0f, 0.0f, 1.0f}}};
	VkRenderPassBeginInfo rp_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = state.render_pass,
		.framebuffer = state.framebuffers[state.surface->current_image],
		.renderArea { .extent = state.surface->swapchain->extent },
		.clearValueCount = 1,
		.pClearValues = &clear_color,
	};

	vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
	VkViewport viewport{0, 0, (float)state.surface->swapchain->extent.width, (float)state.surface->swapchain->extent.height, 0, 1};
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	VkRect2D scissor{{0, 0}, state.surface->swapchain->extent};
	vkCmdSetScissor(cmd, 0, 1, &scissor);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state.pipeline->pipeline);
	vkCmdDraw(cmd, 3, 1, 0, 0);
	vkCmdEndRenderPass(cmd);
	VK_CHECK(vkEndCommandBuffer(cmd), /*nothing*/);

	auto submit_index = state.graphics_queue->command_submission_timeline_semaphore_next_value;
	{ // Submit queue
		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSemaphoreSubmitInfo wait {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = texture->available_semaphore,
			.stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
		};
		VkCommandBufferSubmitInfo cmds {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
			.commandBuffer = cmd
		};
		VkSemaphoreSubmitInfo signal {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = state.graphics_queue->command_submission_timeline_semaphore,
			.value = state.graphics_queue->command_submission_timeline_semaphore_next_value++,
			.stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
		};
		VkSubmitInfo2 info = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.waitSemaphoreInfoCount = 1,
			.pWaitSemaphoreInfos = &wait,
			.commandBufferInfoCount = 1,
			.pCommandBufferInfos = &cmds,
			.signalSemaphoreInfoCount = 1,
			.pSignalSemaphoreInfos = &signal
		};
		VK_CHECK(vkQueueSubmit2(state.graphics_queue->queue, 1, &info, fence), /*nothing*/);
	}

	gpuSurfacePresentEXT(state.graphics_queue, state.surface, submit_index);
	if(errno == SURFACE_SUBOPTIMAL || errno == SURFACE_OUT_OF_DATE)
		recreate_swapchain(state, needed_size);

	state.current_frame++;
}

int main() {
	try {
		render_state state;
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		state.window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Triangle", nullptr, nullptr);

		init_vulkan(state);
		while (!glfwWindowShouldClose(state.window)) {
			glfwPollEvents();
			draw_frame(state);
		}
		vkDeviceWaitIdle(state.graphics_queue->device);
		cleanup(state);
	} catch (const std::exception& e) {
		std::println(std::cerr, "Fatal: {}\n", e.what());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}