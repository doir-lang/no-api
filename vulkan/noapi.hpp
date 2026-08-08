#pragma once

#include "../surface.hpp"
#include "../sync.hpp"
#include "../samplers.hpp"
#include "../allocator.hpp"

#include <vulkan/vulkan_core.h>
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
		void error_callback(void* queue, int type, std::string_view message);
	}
}

template<>
struct std::hash<std::vector<GpuSamplerDesc>> {
	size_t operator()(const std::vector<GpuSamplerDesc>& descs) const noexcept {
		size_t out = 0;
		for(auto& desc: descs)
			out ^= desc.pack();
		return out;
	}
};

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
std::expected<GpuVulkanDefault, std::string> gpuSetupDefaultVulkanEXT(
	GPU::function_t<VkSurfaceKHR(VkInstance)> surface_loader, void(*error_callback)(void* queue, int type, std::string_view message) = GPU::default_::error_callback, VkDebugUtilsMessageSeverityFlagBitsEXT severity_filter = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
	std::span<const char*> instance_extensions = {}, std::span<const char*> extra_layers = {}, std::span<const char*> device_extensions = {}, bool debug = true
);

inline VkPhysicalDeviceFeatures gpuEnableRequiredVulkanFeaturesEXT(VkPhysicalDeviceFeatures features) {
	features.shaderInt64 = true;
	return features;
}

inline VkPhysicalDeviceVulkan12Features gpuEnableRequiredVulkan12FeaturesEXT(VkPhysicalDeviceVulkan12Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	features.bufferDeviceAddress = true;
	features.timelineSemaphore = true;
	return features;
}

inline VkPhysicalDeviceVulkan13Features gpuEnableRequiredVulkan13FeaturesEXT(VkPhysicalDeviceVulkan13Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	features.dynamicRendering = true;
	features.synchronization2 = true;
	return features;
}

inline VkPhysicalDeviceVulkan14Features gpuEnableRequiredVulkan14FeaturesEXT(VkPhysicalDeviceVulkan14Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
	// features.pushDescriptor = true;
	features.indexTypeUint8 = true;
	return features;
}

inline std::vector<const char*> gpuRequiredVulkanDeviceExtensionsEXT() {
	return {"VK_EXT_descriptor_heap", "VK_KHR_shader_untyped_pointers"};
}

inline void* gpuRequiredVulkanDeviceCreateInfoPnextEXT() {
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

	std::unordered_map<std::vector<GpuSamplerDesc>, VkDeviceAddress> sampler_cache;

	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkSemaphore command_submission_timeline_semaphore = VK_NULL_HANDLE;
	uint64_t command_submission_timeline_semaphore_next_value = 1;
	std::vector<std::pair<VkCommandBuffer, uint64_t>> command_buffers_pending_free;
};
GpuQueue* gpuCreateQueue(VkInstance instance, VkPhysicalDevice gpu, VkDevice device, VkQueue queue = VK_NULL_HANDLE, uint32_t queue_family = -1, bool is_graphics_queue = true, CpuAllocatorFunc allocator = default_::cpu_allocator, VkAllocationCallbacks* callbacks = nullptr);
inline GpuQueue* gpuCreateQueue(const GpuVulkanDefault& vulkan, CpuAllocatorFunc allocator = default_::cpu_allocator, VkAllocationCallbacks* callbacks = nullptr) {
	return gpuCreateQueue(vulkan.instance, vulkan.gpu, vulkan.device, vulkan.graphics_queue, vulkan.graphics_queue_family, true, allocator, callbacks);
}

struct GpuPipeline {
	VkPipeline pipeline;
	std::optional<size_t> color_target_count = {}; // When null indicates a compute pipeline
};

struct GpuTexture {
	VkImage image;
	VkImageView full_view;
	GpuTextureDesc descriptor;
	VkSemaphore available_semaphore = VK_NULL_HANDLE;
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

	uint32_t current_image = uint32_t(-1), semaphore_counter = 0;
};
GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, VkSurfaceKHR surface, const GpuSurfaceDescriptor& desc);



constexpr static std::string_view COMPUTE_SHADER_PROLOGUE = R"(
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
)";

constexpr static std::string_view GRAPHICS_SHADER_PROLOGUE = R"(
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
	uint64_t vertex_data;
	uint64_t fragment_data;
	uint64_t index_data;
	GpuSamplerMap sampler_map;
} pc;

uint gpuGetSamplerIndex(const GpuSamplerDesc desc) {
	return pc.sampler_map.data[gpuPackSamplerDesc(desc)];
}

// End prologue
)";