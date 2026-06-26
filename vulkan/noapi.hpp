#pragma once

#include "../surface.hpp"

#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>

#include <unordered_map>
#include <functional>
#include <expected>


#ifndef NOAPI_NO_EXCEPTIONS
	struct VkResultExecption : std::exception {
		VkResult result;

		VkResultExecption(VkResult result) : result(result) {}
	};

	#define VK_CHECK(expr) do {\
		auto res = expr;\
		if(res != VK_SUCCESS)\
			throw VkResultExecption(res);\
	} while(false)
#else
	#define VK_CHECK(expr) do {\
		auto res = expr;\
		if(res != VK_SUCCESS)\
			std::exit((int)res);\
	} while(false)
#endif


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
	VkPhysicalDevice gpu;
	VkDevice device;
	VkQueue queue;
	uint32_t queue_family;
	bool is_graphics_queue;

	VkAllocationCallbacks* callbacks = nullptr;

	// Memory Allocator
	VmaAllocator allocator = VK_NULL_HANDLE;
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
std::optional<GpuQueue> gpuCreateQueue(VkInstance instance, VkPhysicalDevice gpu, VkDevice device, VkQueue queue = VK_NULL_HANDLE, uint32_t queue_family = -1, VkAllocationCallbacks* callbacks = nullptr, bool is_graphics_queue = true);
inline std::optional<GpuQueue> gpuCreateQueue(const GpuVulkanDefault& vulkan, VkAllocationCallbacks* callbacks = nullptr) {
	return gpuCreateQueue(vulkan.instance, vulkan.gpu, vulkan.device, vulkan.graphics_queue, vulkan.graphics_queue_family, callbacks, true);
}

struct GpuPipeline {
	// VkShaderModule compute_module; // TODO: Do we have a need to carry this around?
	VkPipeline pipeline;
};

struct GpuTexture {
	VkImage image;
	GpuTextureDesc descriptor;
};

struct GpuCommandBuffer {
	GpuQueue* queue;
	VkCommandBuffer command_buffer;
	bool ended = false;
};

struct GpuSemaphore {
	VkSemaphore semaphore;
};
