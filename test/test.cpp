#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <volk.h>
#include <vulkan/noapi.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <print>
#include <iostream>
#include <cassert>

static constexpr uint32_t WIDTH = 800;
static constexpr uint32_t HEIGHT = 600;

static const char* vertex_glsl = R"glsl(
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
	layout(location = 0) in vec3 frag_color;
	layout(location = 0) out vec4 out_color;

	void main() {
		out_color = vec4(frag_color, 1.0);
	}
)glsl";

struct render_state {
	GLFWwindow* window = nullptr;

	VkInstance instance = VK_NULL_HANDLE;
	VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
	VkSurfaceKHR vulkan_surface = VK_NULL_HANDLE;
	GpuQueue* graphics_queue = nullptr;

	GpuSurface* surface = nullptr;
	GpuPipeline* pipeline = nullptr;
	gpu* index_buffer = nullptr;

	uint64_t last_submission_index = -1;
};

void ensure_glslang_initialized() {
	static bool glslang_initialized = false;

	if (!glslang_initialized) {
		assert(glslang::InitializeProcess());
		glslang_initialized = true;
	}
}

static std::vector<uint32_t> compile_glsl(EShLanguage stage, std::string source) {
	ensure_glslang_initialized();
	const char* source_array = source.c_str();

	glslang::TShader shader(stage);
	shader.setStrings(&source_array, 1);
	shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 460);
	shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_4);
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

void init_vulkan(render_state& state) {
	auto vk = gpuSetupDefaultVulkanEXT([&state](VkInstance instance) -> VkSurfaceKHR{
		VkSurfaceKHR out;
		VK_CHECK(glfwCreateWindowSurface(instance, state.window, nullptr, &out), nullptr);
		return out;
	});
	if(!vk) throw std::runtime_error(vk.error());
	state.instance = vk->instance;
	state.debug_messenger = vk->messenger;
	state.vulkan_surface = vk->surface;

	auto queue = gpuCreateQueue(*vk);
	if(!queue) throw std::runtime_error("Failed to create NoAPI queue");
	state.graphics_queue = queue;

	state.surface = gpuCreateSurfaceEXT(queue, state.vulkan_surface, GpuSurfaceDescriptor{
		.texture = {
			.dimensions = {WIDTH, HEIGHT, 1},
			.format = FORMAT_RGBA8_UNORM
		}
	});

	auto indices = gpuMalloc<uint8_t>(state.graphics_queue, 3);
	state.index_buffer = gpuHostToDevicePointer(state.graphics_queue, indices);
	for(size_t i = 0; i < 3; ++i)
		indices[i] = i;

	auto vertex_spirv = compile_glsl(EShLangVertex, vertex_glsl);
	auto fragment_spirv = compile_glsl(EShLangFragment, fragment_glsl);

	GpuColorTarget target { .format = FORMAT_RGBA8_UNORM };
	state.pipeline = gpuCreateGraphicsPipeline(state.graphics_queue, byte_span<uint32_t>(vertex_spirv), byte_span<uint32_t>(fragment_spirv), {
		.colorTargets = {&target, 1}
	});
}

void recreate_swapchain(render_state& state, uvec2 extent) {
	gpuWaitIdleEXT(state.graphics_queue);

	auto config = gpuSurfaceGetConfigurationEXT(state.surface);
	config.texture.dimensions = {extent.x, extent.y, 1};
	gpuSurfaceReconfigureEXT(state.graphics_queue, state.surface, config);
}

void cleanup(render_state& state) {
	gpuFree(state.graphics_queue, state.index_buffer);
	gpuFreeSurfaceEXT(state.graphics_queue, state.surface);
	gpuFreePipeline(state.graphics_queue, state.pipeline);
	auto device = state.graphics_queue->device;
	gpuFreeQueue(state.graphics_queue);
	vkDestroyDevice(state.graphics_queue->device, nullptr);
	vkDestroySurfaceKHR(state.instance, state.vulkan_surface, nullptr);
	vkDestroyDebugUtilsMessengerEXT(state.instance, state.debug_messenger, nullptr);
	vkDestroyInstance(state.instance, nullptr);
	volkFinalize();
	glfwDestroyWindow(state.window);
	glfwTerminate();
	glslang::FinalizeProcess();
}

void draw_frame(render_state& state) {
	int w = 0, h = 0;
	glfwGetFramebufferSize(state.window, &w, &h);
	while (w == 0 || h == 0) {
		glfwGetFramebufferSize(state.window, &w, &h);
		glfwWaitEvents();
	}
	uvec2 needed_size = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
	auto surface_config = gpuSurfaceGetConfigurationEXT(state.surface);
	auto surface_size = surface_config.texture.dimensions;
	if(surface_size.x != w && surface_size.y != h)
		recreate_swapchain(state, needed_size);

	auto texture = gpuSurfaceNextTextureEXT(state.graphics_queue, state.surface);
	if(errno == SURFACE_SUBOPTIMAL || errno == SURFACE_OUT_OF_DATE) {
		recreate_swapchain(state, needed_size);
		return;
	}

	if(state.last_submission_index != -1)
		gpuWaitSemaphore(state.graphics_queue, gpuGetSubmissionSemaphoreEXT(state.graphics_queue), state.last_submission_index);

	auto cmd = gpuStartCommandRecording(state.graphics_queue);
	GpuColorAttachment target { .texture = texture };
	GpuRenderPassDesc rp = { .colorAttachments = std::span<GpuColorAttachment>(&target, 1) };
	gpuBeginRenderPass(cmd, rp);
	{
		gpuSetPipeline(cmd, state.pipeline);
		gpuDrawIndexedInstanced(cmd, nullptr, nullptr, state.index_buffer, 3, 1, INDEX_TYPE_UINT8);
	}
	gpuEndRenderPass(cmd, rp);

	state.last_submission_index = gpuSubmit(state.graphics_queue, {&cmd, 1});
	gpuSurfacePresentEXT(state.graphics_queue, state.surface, state.last_submission_index);
	if(errno == SURFACE_SUBOPTIMAL || errno == SURFACE_OUT_OF_DATE)
		recreate_swapchain(state, needed_size);
}

int real_main() try {
	render_state state;
	assert(glfwInit() == GLFW_TRUE);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	state.window = glfwCreateWindow(WIDTH, HEIGHT, "NoAPI Triangle", nullptr, nullptr);

	init_vulkan(state);
	while (!glfwWindowShouldClose(state.window)) {
		glfwPollEvents();
		draw_frame(state);
	}

	gpuWaitIdleEXT(state.graphics_queue);
	cleanup(state);
	return EXIT_SUCCESS;
} catch (const std::exception& e) {
	std::println(std::cerr, "Fatal: {}\n", e.what());
	return EXIT_FAILURE;
}





#ifdef _WIN32
	#include <windows.h>
#endif

#ifdef _WIN32
	int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
		return real_main();
	}
#else
	int main() {
		return real_main();
	}
#endif