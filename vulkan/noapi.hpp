#pragma once

#include "../surface.hpp"
#include "../sync.hpp"
#include "../allocator.hpp"

#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>

#include <unordered_map>
#include <functional>
#include <expected>
#include <memory>


#define VK_CHECK(expr, RETURN) do {\
	auto res = expr;\
	if(res != VK_SUCCESS) {\
		errno = res;\
		return RETURN;\
	}\
} while(false)


struct GpuVulkanDefault {
	VkInstance instance = VK_NULL_HANDLE;
	VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	VkPhysicalDevice gpu = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue graphics_queue = VK_NULL_HANDLE;
	uint32_t graphics_queue_family;
};
std::expected<GpuVulkanDefault, std::string> gpuSetupDefaultVulkan(
	std::function_ref<VkSurfaceKHR(VkInstance)> surface_loader, std::span<const char*> instance_extensions = {}, std::span<const char*> extra_layers = {}, std::span<const char*> device_extensions = {}, bool debug = true
);

inline VkPhysicalDeviceFeatures gpuEnableRequiredVulkanFeatures(VkPhysicalDeviceFeatures features) {
	features.shaderInt64 = true;
	return features;
}

inline VkPhysicalDeviceVulkan12Features gpuEnableRequiredVulkan12Features(VkPhysicalDeviceVulkan12Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	features.bufferDeviceAddress = true;
	features.timelineSemaphore = true;
	return features;
}

inline VkPhysicalDeviceVulkan13Features gpuEnableRequiredVulkan13Features(VkPhysicalDeviceVulkan13Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	features.synchronization2 = true;
	return features;
}

inline VkPhysicalDeviceVulkan14Features gpuEnableRequiredVulkan14Features(VkPhysicalDeviceVulkan14Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
	features.pushDescriptor = true;
	return features;
}

inline std::vector<const char*> gpuRequiredVulkanDeviceExtensions() {
	return {"VK_EXT_descriptor_heap", "VK_KHR_shader_untyped_pointers"};
}

inline void* gpuRequiredVulkanDeviceCreateInfoPnext() {
	static VkPhysicalDeviceDescriptorHeapFeaturesEXT descriptor_heap_info {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
		.descriptorHeap = true
	};
	static VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untyped_pointers_info {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
		.pNext = &descriptor_heap_info,
		.shaderUntypedPointers = true
	};
	return &untyped_pointers_info;
}

struct GpuQueue {
	GpuAllocatorFunc cpu_allocator;
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
	VkDeviceSize minimum_descriptor_heap_size = 0;
	// Mappings between cpu and gpu pointers
	std::unordered_map<void*, VkDeviceAddress> host2gpu;
	std::unordered_map<VkDeviceAddress, void*> gpu2host;
	std::unordered_map<VkDeviceAddress, VkImage> gpu2image;

	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkSemaphore command_submission_timeline_semaphore = VK_NULL_HANDLE;
	uint64_t command_submission_timeline_semaphore_next_value = 1;
	std::vector<std::pair<VkCommandBuffer, uint64_t>> command_buffers_pending_free;
};
GpuQueue* gpuCreateQueue(VkInstance instance, VkPhysicalDevice gpu, VkDevice device, VkQueue queue = VK_NULL_HANDLE, uint32_t queue_family = -1, GpuAllocatorFunc allocator = default_::gpu_allocator, VkAllocationCallbacks* callbacks = nullptr, bool is_graphics_queue = true);
inline GpuQueue* gpuCreateQueue(const GpuVulkanDefault& vulkan, GpuAllocatorFunc allocator = default_::gpu_allocator, VkAllocationCallbacks* callbacks = nullptr) {
	return gpuCreateQueue(vulkan.instance, vulkan.gpu, vulkan.device, vulkan.graphics_queue, vulkan.graphics_queue_family, allocator, callbacks, true);
}

struct GpuPipeline {
	// VkShaderModule compute_module; // TODO: Do we have a need to carry this around?
	VkPipeline pipeline;
	std::optional<size_t> color_target_count = {}; // When null indicates a compute pipeline
};

struct GpuTexture {
	VkImage image;
	GpuTextureDesc descriptor;
	VkSemaphore available_semaphore = VK_NULL_HANDLE;
};

struct GpuCommandBuffer {
	GpuQueue* queue;
	VkCommandBuffer command_buffer;
	bool ended = false;
	const GpuPipeline* bound_pipeline = nullptr;
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

constexpr static int SURFACE_SUBOPTIMAL = VK_SUBOPTIMAL_KHR;
constexpr static int SURFACE_OUT_OF_DATE = VK_ERROR_OUT_OF_DATE_KHR;

struct GpuSurface {
	VkSurfaceKHR surface;
	std::shared_ptr<vkb::Swapchain> swapchain = nullptr;
	GpuSurfaceDescriptor descriptor;
	std::vector<GpuTexture> images;
	std::vector<VkImageView> image_views;
	std::vector<VkSemaphore> image_available_semaphores;
	std::vector<VkSemaphore> render_finished_semaphores;

	uint32_t current_image = -1, semaphore_counter = 0;
};
GpuSurface* gpuCreateSurface(GpuQueue* queue, VkSurfaceKHR surface, const GpuSurfaceDescriptor& desc);