#pragma once

// The backend's C interface: the setup helper, the feature/extension requirements, the
// queue and surface constructors and the constants. This file adds the C++ conveniences on
// top of it, along with the definitions of the objects behind the API's opaque handles.
#include "noapi.h"

#include <vk_mem_alloc.h>

#include <vector>
#include <unordered_map>
#include <functional>
#include <expected>
#include <memory>
#include <string>
#include <array>

namespace GPU {
#ifdef __cpp_lib_function_ref
	template<typename T>
	using function_t = std::function_ref<T>;
#else
	template<typename T>
	using function_t = std::function<T>;
#endif

	namespace default_ {
		// The C hook, under the name the rest of the C++ code knows it by
		inline constexpr GpuErrorCallbackEXT error_callback = gpuDefaultErrorCallbackEXT;
	}
}

#define VK_CHECK(expr, RETURN) do {\
	auto res = expr;\
	if(res != VK_SUCCESS) {\
		errno = res;\
		return RETURN;\
	}\
} while(false)

/**
 * gpuSetupDefaultVulkanEXT – C++ flavour of the setup helper: takes any callable (a
 * capturing lambda, say) as the surface loader and reports failure as the string a
 * std::expected carries, rather than through out parameters.
 *
 * @param surface_loader Creates the presentation surface from the new instance.
 * @param error_callback Where validation messages are reported.
 * @param severity_filter Messages below this severity are dropped.
 * @param instance_extensions Extra instance extensions to enable.
 * @param extra_layers Extra instance layers to enable.
 * @param device_extensions Extra device extensions to enable.
 * @param debug Enable the validation layers and the debug messenger.
 */
inline std::expected<GpuVulkanDefault, std::string> gpuSetupDefaultVulkanEXT(
	GPU::function_t<VkSurfaceKHR(VkInstance)> surface_loader,
	GpuErrorCallbackEXT error_callback = gpuDefaultErrorCallbackEXT,
	VkDebugUtilsMessageSeverityFlagBitsEXT severity_filter = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
	std::span<const char*> instance_extensions = {}, std::span<const char*> extra_layers = {},
	std::span<const char*> device_extensions = {}, bool debug = true
) {
	// The loader is handed across the C boundary as a plain function and the callable it
	// stands for rides along as the userdata
	auto load_surface = +[](VkInstance instance, void* userdata) -> VkSurfaceKHR {
		return (*(GPU::function_t<VkSurfaceKHR(VkInstance)>*)userdata)(instance);
	};

	GpuVulkanDefault out;
	char error[512] = {};
	if(!gpuSetupDefaultVulkanEXT(load_surface, &surface_loader, error_callback, severity_filter,
		instance_extensions, extra_layers, device_extensions, debug, &out, error, sizeof(error)))
		return std::unexpected(std::string(error));
	return out;
}

struct GpuQueue {
	CpuAllocatorFunc cpu_allocator;
	VkPhysicalDevice gpu;
	VkDevice device;
	VkQueue queue;
	uint32_t queue_family;
	bool is_graphics_queue;

	VkAllocationCallbacks* callbacks = nullptr;

	// Memory Allocator
	VmaAllocator gpu_allocator = VK_NULL_HANDLE;
	// Mapping from gpu* (Device Addresses) to buffer allocations
	std::unordered_map<VkDeviceAddress, std::tuple<VkBuffer, VmaAllocation, VkDeviceSize>> allocations;
	// Mapping from gpu* (Device Addresses) to a mapped descriptor heap
	std::unordered_map<VkDeviceAddress, std::tuple<VkBuffer, VmaAllocation, VkDeviceSize, VkDeviceAddress>> descriptor_heaps;
	VkDeviceSize minimum_descriptor_heap_size = 0, sampler_size = 0;
	// Mappings between cpu and gpu pointers
	std::unordered_map<void*, VkDeviceAddress> host2gpu;
	std::unordered_map<VkDeviceAddress, void*> gpu2host;
	// Mapping from gpu* (Device Addresses) to an associated image
	std::unordered_map<VkDeviceAddress, VkImage> gpu2image;
	// Mapping from gpu* (Device Addresses) to an associated index buffer
	std::unordered_map<VkDeviceAddress, std::tuple<VkBuffer, VmaAllocation, VkDeviceSize>> gpu2index;

	std::unordered_map<std::vector<GpuSamplerDesc>, VkDeviceAddress, GpuSamplerDescListHash> sampler_cache;

	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkSemaphore command_submission_timeline_semaphore = VK_NULL_HANDLE;
	uint64_t command_submission_timeline_semaphore_next_value = 1;
	std::vector<std::pair<VkCommandBuffer, uint64_t>> command_buffers_pending_free;

	// gpuBlitTextureEXT's internal pipeline. It samples one texture into another and binds nothing
	// else, so rather than going through the descriptor heap every other pipeline here uses it keeps
	// a classic combined image sampler layout of its own, built lazily on the first blit.
	std::array<VkShaderModule, 2> blit_shader_modules = {}; // vertex, fragment
	VkDescriptorSetLayout blit_descriptor_set_layout = VK_NULL_HANDLE;
	VkPipelineLayout blit_pipeline_layout = VK_NULL_HANDLE;
	std::array<VkSampler, 2> blit_samplers = {}; // indexed by "linear"
	std::unordered_map<uint64_t, VkPipeline> blit_pipelines; // keyed by destination VkFormat
	std::vector<VkDescriptorPool> blit_descriptor_pools;
	// Sets are recycled once the submission that referenced them has finished
	std::vector<VkDescriptorSet> blit_descriptor_sets_free;
	std::vector<std::pair<VkDescriptorSet, uint64_t>> blit_descriptor_sets_in_flight;

	// Image views handed to a render pass or to a blit, destroyed once the submission that
	// referenced them has finished. Shared rather than blit specific because gpuBeginRenderPass
	// builds one per attachment, which is what lets an attachment's mip level and slice be honored.
	std::vector<std::pair<VkImageView, uint64_t>> views_in_flight;
};
inline GpuQueue* gpuCreateQueue(const GpuVulkanDefault& vulkan, CpuAllocatorFunc allocator = default_::cpu_allocator, VkAllocationCallbacks* callbacks = nullptr) {
	return gpuCreateQueue(vulkan.instance, vulkan.gpu, vulkan.device, vulkan.graphics_queue, vulkan.graphics_queue_family, true, allocator, callbacks);
}

struct GpuPipeline {
	VkPipeline pipeline;
	std::optional<size_t> color_target_count = {}; // When null indicates a compute pipeline
};

struct GpuTexture {
	VkImage image;
	VkImageView full_view; // Only a surface's own images keep one; render passes build their own
	GpuTextureDesc descriptor;
	VkSemaphore available_semaphore = VK_NULL_HANDLE;
	// Where this backend last left the image. Tracked rather than assumed so that an attachment
	// with LOAD_OP_LOAD can be transitioned without discarding what is already in it.
	VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct GpuCommandBuffer {
	GpuQueue* queue;
	VkCommandBuffer command_buffer;
	enum State {
		Recording,
		RecordingRenderPass,
		Ended
	} state;
	const GpuPipeline* bound_pipeline = nullptr;
	VkDeviceAddress sampler_map = {};
	std::vector<VkSemaphore> wait_semaphores;
};

struct GpuSemaphore {
	VkSemaphore semaphore;
};

struct GpuDepthStencilState {
	GpuDepthStencilDesc descriptor;
};

struct GpuBlendState {
	GpuBlendDesc descriptor;
};

namespace vkb {
	struct Swapchain;
}

struct GpuSurface {
	VkSurfaceKHR surface;
	std::shared_ptr<vkb::Swapchain> swapchain = nullptr;
	GpuSurfaceDescriptor descriptor;
	std::vector<GpuTexture> images;
	std::vector<VkImageView> image_views;
	std::vector<VkSemaphore> image_available_semaphores;
	std::vector<VkSemaphore> render_finished_semaphores;

	uint32_t current_image = uint32_t(-1), semaphore_counter = 0;
};
inline GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, const GpuVulkanDefault& vulkan, const GpuSurfaceDescriptor& desc) {
	return gpuCreateSurfaceEXT(queue, vulkan.surface, desc);
}
