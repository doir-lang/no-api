#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#define VOLK_IMPLEMENTATION
#include <volk.h>
#include <VkBootstrap.h>

#include <vulkan/noapi.hpp>

#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

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
		vec2( 0.0, -0.5),
		vec2( 0.5, 0.5),
		vec2(-0.5, 0.5)
	);

	vec3 colors[3] = vec3[](
		vec3(1.0, 0.0, 0.0),
		vec3(0.0, 1.0, 0.0),
		vec3(0.0, 0.0, 1.0)
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

	vkb::Swapchain vkb_swapchain = {};

	VkInstance instance = {};
	VkDebugUtilsMessengerEXT debug_messenger = {};
	VkSurfaceKHR surface = {};
	VkPhysicalDevice gpu = {};
	VkDevice device = {};
	VkQueue graphics_queue = {};
	uint32_t graphics_family = {};

	VmaAllocator allocator = {};

	VkSwapchainKHR swapchain = {};
	VkFormat swapchain_format = {};
	VkExtent2D swapchain_extent = {};
	std::vector<VkImage> swapchain_images;
	std::vector<VkImageView> swapchain_views;
	std::vector<VkSemaphore> swapchain_render_finished_semaphore = {};

	VkRenderPass render_pass = {};
	VkPipelineLayout pipeline_layout = {};
	VkPipeline pipeline = {};
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

// ─── Vulkan helpers ───────────────────────────────────────────────────────────
#define VK_CHECK(expr) \
	do { \
		VkResult _r = (expr); \
		if (_r != VK_SUCCESS) { \
			std::println(std::cerr, "Vulkan error %d at %s:%d", (int)_r, __FILE__, __LINE__); \
			std::abort(); \
		} \
	} while(0)

static VkShaderModule create_shader_module(render_state& state, const std::vector<uint32_t>& spv) {
	VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = spv.size() * sizeof(uint32_t),
		.pCode = spv.data(),
	};
	VkShaderModule out;
	VK_CHECK(vkCreateShaderModule(state.device, &info, nullptr, &out));
	return out;
}

void create_swapchain(render_state& state, int width, int height) {
	vkb::SwapchainBuilder builder{state.gpu, state.device, state.surface, state.graphics_family};
	auto swap = builder
		.set_old_swapchain(state.swapchain)
		.use_default_present_mode_selection()
		.set_desired_extent(width, height)
		.build();
	if (!swap) throw std::runtime_error(swap.error().message());

	vkb::destroy_swapchain(state.vkb_swapchain);
	state.vkb_swapchain = swap.value();
	state.swapchain = state.vkb_swapchain.swapchain;
	state.swapchain_format = state.vkb_swapchain.image_format;
	state.swapchain_extent = state.vkb_swapchain.extent;
	state.swapchain_images = state.vkb_swapchain.get_images().value();
	state.swapchain_views = state.vkb_swapchain.get_image_views().value();

	VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
	state.swapchain_render_finished_semaphore.resize(state.swapchain_images.size());
	for (auto& sema : state.swapchain_render_finished_semaphore)
		VK_CHECK(vkCreateSemaphore(state.device, &semaphore_info, nullptr, &sema));
}

void create_pipeline(render_state& state) {
	auto vertex_spirv = compile_glsl(EShLangVertex, vertex_glsl);
	auto fragment_spirv = compile_glsl(EShLangFragment, fragment_glsl);

	VkShaderModule vertex_module = create_shader_module(state, vertex_spirv);
	VkShaderModule fragment_module = create_shader_module(state, fragment_spirv);

	VkPipelineShaderStageCreateInfo stages[2] = {
		VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_module,
			.pName = "main",
		},
		VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_module,
			.pName = "main",
		}
	};

	VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	};
	VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	VkViewport viewport{0, 0, (float)state.swapchain_extent.width, (float)state.swapchain_extent.height, 0, 1};
	VkRect2D scissor{{0, 0}, state.swapchain_extent};

	VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = &viewport,
		.scissorCount = 1,
		.pScissors = &scissor,
	};
	VkPipelineRasterizationStateCreateInfo raster = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f,
	};
	VkPipelineMultisampleStateCreateInfo msaa = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	VkPipelineColorBlendAttachmentState blend_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
	};
	VkPipelineColorBlendStateCreateInfo blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	VkPipelineLayoutCreateInfo layoutInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
	};
	VK_CHECK(vkCreatePipelineLayout(state.device, &layoutInfo, nullptr, &state.pipeline_layout));

	VkGraphicsPipelineCreateInfo pipeInfo = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &raster,
		.pMultisampleState = &msaa,
		.pColorBlendState = &blend,
		.layout = state.pipeline_layout,
		.renderPass = state.render_pass,
		.subpass = 0,
	};
	VK_CHECK(vkCreateGraphicsPipelines(state.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &state.pipeline));

	// Shader modules are only needed during pipeline creation
	vkDestroyShaderModule(state.device, vertex_module, nullptr);
	vkDestroyShaderModule(state.device, fragment_module, nullptr);
}

void create_framebuffers(render_state& state) {
	state.framebuffers.resize(state.swapchain_views.size());
	for (size_t i = 0; i < state.swapchain_views.size(); ++i) {
		VkFramebufferCreateInfo fbInfo = {
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = state.render_pass,
			.attachmentCount = 1,
			.pAttachments = &state.swapchain_views[i],
			.width = state.swapchain_extent.width,
			.height = state.swapchain_extent.height,
			.layers = 1,
		};
		VK_CHECK(vkCreateFramebuffer(state.device, &fbInfo, nullptr, &state.framebuffers[i]));
	}
}

void init_vulkan(render_state& state) {
	// Instance
	VK_CHECK(volkInitialize());

	vkb::InstanceBuilder instance_builder;
	auto inst = instance_builder
		.set_app_name("VulkanTriangle")
		.request_validation_layers(true)
		.use_default_debug_messenger()
		.require_api_version(1, 3, 0)
		.build();
	if (!inst) throw std::runtime_error(inst.error().message());
	auto instance = inst.value();
	state.instance = instance.instance;
	state.debug_messenger = instance.debug_messenger;

	volkLoadInstance(state.instance);

	// Surface
	VK_CHECK(glfwCreateWindowSurface(state.instance, state.window, nullptr, &state.surface));

	// Physical Device
	vkb::PhysicalDeviceSelector gpu_selector{instance};
	auto phys = gpu_selector
		.set_surface(state.surface)
		.set_minimum_version(1, 3)
		.select();
	if (!phys) throw std::runtime_error(phys.error().message());
	auto gpu = phys.value();
	state.gpu = gpu.physical_device;

	// Logical Device
	vkb::DeviceBuilder device_builder{gpu};
	auto dev = device_builder.build();
	if (!dev) throw std::runtime_error(dev.error().message());
	auto device = dev.value();
	state.device = device.device;

	state.graphics_queue = device.get_queue(vkb::QueueType::graphics).value();
	state.graphics_family = device.get_queue_index(vkb::QueueType::graphics).value();
	// state.present_queue = state.vkb_device.get_queue(vkb::QueueType::present).value();
	// state.present_family = state.vkb_device.get_queue_index(vkb::QueueType::present).value();

	volkLoadDevice(state.device);

	// VMA
	VmaVulkanFunctions functions = {
		.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
		.vkGetDeviceProcAddr = vkGetDeviceProcAddr,
	};

	VmaAllocatorCreateInfo vma_info = {
		.physicalDevice = state.gpu,
		.device = state.device,
		.pVulkanFunctions = &functions,
		.instance = state.instance,
		.vulkanApiVersion = VK_API_VERSION_1_3,
	};
	VK_CHECK(vmaCreateAllocator(&vma_info, &state.allocator));

	create_swapchain(state, WIDTH, HEIGHT);
	{ // create_render_pass();
		VkAttachmentDescription color_attachment = {
			.format = state.swapchain_format,
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
		VK_CHECK(vkCreateRenderPass(state.device, &rpInfo, nullptr, &state.render_pass));
	}
	create_pipeline(state);
	create_framebuffers(state);
	{ // create_command_pool();
		VkCommandPoolCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = state.graphics_family,
		};
		VK_CHECK(vkCreateCommandPool(state.device, &info, nullptr, &state.command_pool));
	}{ // allocate_command_buffers();
		VkCommandBufferAllocateInfo info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = state.command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = FRAMES_IN_FLIGHT,
		};
		VK_CHECK(vkAllocateCommandBuffers(state.device, &info, state.command_buffers.data()));
	}{ // create_sync_objects();
		VkSemaphoreCreateInfo sema{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
		VkFenceCreateInfo fence{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT
		};

		for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
			VK_CHECK(vkCreateSemaphore(state.device, &sema, nullptr, &state.image_available_semaphore[i]));
			VK_CHECK(vkCreateFence (state.device, &fence, nullptr, &state.render_fence[i]));
		}
	}
}

void cleanup_swapchain(render_state& state) {
	for (auto sema : state.swapchain_render_finished_semaphore)
		vkDestroySemaphore(state.device, sema, nullptr);
	state.swapchain_render_finished_semaphore.clear();
	for (auto fb : state.framebuffers) vkDestroyFramebuffer(state.device, fb, nullptr);
	state.framebuffers.clear();
	for (auto iv : state.swapchain_views) vkDestroyImageView(state.device, iv, nullptr);
	state.swapchain_views.clear();
}

void recreate_swapchain(render_state& state) {
	int w = 0, h = 0;
	glfwGetFramebufferSize(state.window, &w, &h);
	while (w == 0 || h == 0) {
		glfwGetFramebufferSize(state.window, &w, &h);
		glfwWaitEvents();
	}
	vkDeviceWaitIdle(state.device);
	cleanup_swapchain(state);
	create_swapchain(state, w, h);
	create_framebuffers(state);
}

void cleanup(render_state& state) {
	for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
		vkDestroySemaphore(state.device, state.image_available_semaphore[i], nullptr);
		vkDestroyFence (state.device, state.render_fence[i], nullptr);
	}
	vkDestroyCommandPool (state.device, state.command_pool, nullptr);
	cleanup_swapchain(state);
	vkDestroyPipeline (state.device, state.pipeline, nullptr);
	vkDestroyPipelineLayout(state.device, state.pipeline_layout, nullptr);
	vkDestroyRenderPass (state.device, state.render_pass, nullptr);
	vmaDestroyAllocator(state.allocator);
	vkb::destroy_swapchain(state.vkb_swapchain);
	vkDestroyDevice(state.device, nullptr);
	vkDestroySurfaceKHR (state.instance, state.surface, nullptr);
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
	vkWaitForFences(state.device, 1, &fence, VK_TRUE, UINT64_MAX);

	uint32_t image_index = {};
	VkResult result = vkAcquireNextImageKHR(state.device, state.swapchain, UINT64_MAX, state.image_available_semaphore[frame_index], VK_NULL_HANDLE, &image_index);
	if (result == VK_ERROR_OUT_OF_DATE_KHR) { recreate_swapchain(state); return; }
	if ( !(result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) )
		throw std::runtime_error("vkAcquireNextImageKHR failed");

	vkResetFences(state.device, 1, &fence);

	VkCommandBuffer cmd = state.command_buffers[frame_index];
	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo cmd_begin = {};
	cmd_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin));

	VkClearValue clear_color{{{0.0f, 0.0f, 0.0f, 1.0f}}};
	VkRenderPassBeginInfo rp_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = state.render_pass,
		.framebuffer = state.framebuffers[image_index],
		.renderArea { .extent = state.swapchain_extent },
		.clearValueCount = 1,
		.pClearValues = &clear_color,
	};

	vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state.pipeline);
	vkCmdDraw(cmd, 3, 1, 0, 0);
	vkCmdEndRenderPass(cmd);
	VK_CHECK(vkEndCommandBuffer(cmd));

	{ // Submit queue
		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo info = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &state.image_available_semaphore[frame_index],
			.pWaitDstStageMask = &wait_stage,
			.commandBufferCount = 1,
			.pCommandBuffers = &cmd,
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &state.swapchain_render_finished_semaphore[image_index],
		};
		VK_CHECK(vkQueueSubmit(state.graphics_queue, 1, &info, fence));
	}{ // Present
		VkPresentInfoKHR presentInfo = {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &state.swapchain_render_finished_semaphore[image_index],
			.swapchainCount = 1,
			.pSwapchains = &state.swapchain,
			.pImageIndices = &image_index,
		};

		result = vkQueuePresentKHR(state.graphics_queue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
			recreate_swapchain(state);
		else if (result != VK_SUCCESS)
			throw std::runtime_error("vkQueuePresentKHR failed");
	}

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
		vkDeviceWaitIdle(state.device);
		cleanup(state);
	} catch (const std::exception& e) {
		std::println(std::cerr, "Fatal: %s\n", e.what());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}