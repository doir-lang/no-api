#pragma once

#include "../surface.hpp"

#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>

#include <unordered_map>
#include <functional>
#include <expected>

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
	std::function_ref<VkSurfaceKHR(VkInstance)> surface_loader, std::span<const char*> extra_extensions = {}, std::span<const char*> extra_layers = {}, bool debug = true
);

inline VkPhysicalDeviceVulkan12Features gpuEnableRequiredVulkan12Features(VkPhysicalDeviceVulkan12Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	features.bufferDeviceAddress = true;
	return features;
}

struct GpuQueue {
	VkPhysicalDevice gpu;
	VkDevice device;
	VkQueue queue;
	uint32_t queue_family;
	bool is_graphics_queue;

	// Memory Allocator
	VmaAllocator allocator;
	// Mapping from gpu* (Device Addresses) to buffer allocations
	std::unordered_map<VkDeviceAddress, std::pair<VkBuffer, VmaAllocation>> allocations;
	// Mappings between cpu and gpu pointers
	std::unordered_map<void*, VkDeviceAddress> host2gpu;
	std::unordered_map<VkDeviceAddress, void*> gpu2host;
};
std::optional<GpuQueue> gpuCreateQueue(VkInstance instance, VkPhysicalDevice gpu, VkDevice device, VkQueue queue = VK_NULL_HANDLE, uint32_t queue_family = -1, bool is_graphics_queue = true);
inline std::optional<GpuQueue> gpuCreateQueue(const GpuVulkanDefault& vulkan) {
	return gpuCreateQueue(vulkan.instance, vulkan.gpu, vulkan.device, vulkan.graphics_queue, vulkan.graphics_queue_family, true);
}