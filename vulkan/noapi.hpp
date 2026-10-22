#pragma once

// The C interface plus the C++ conveniences, and the definitions behind the API's opaque handles.
#include "noapi.h"

#include <vk_mem_alloc.h>

#include <vector>
#include <map>
#include <set>
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
		inline constexpr GpuErrorCallbackEXT error_callback = gpuDefaultErrorCallbackEXT;
	}
}

/**
 * gpuSetupDefaultVulkanEXT – C++ flavour of the setup helper: takes any callable (a capturing lambda, say) as the surface loader and reports failure as the string a std::expected carries, rather than through out parameters.
 *
 * @code{.cpp}
 * auto vulkan = gpuSetupDefaultVulkanEXT([](VkInstance) -> VkSurfaceKHR { return VK_NULL_HANDLE; });
 * assert(vulkan);
 * vkDestroyDevice(vulkan->device, nullptr);
 * vkDestroyDebugUtilsMessengerEXT(vulkan->instance, vulkan->messenger, nullptr);
 * vkDestroyInstance(vulkan->instance, nullptr);
 * @endcode
 *
 * @param surface_loader Creates the presentation surface from the new instance. Returning no surface sets up a headless device that can still render into textures.
 * @param error_callback Where validation messages are reported.
 * @param severity_filter Messages below this severity are dropped.
 * @param instance_extensions Extra instance extensions to enable.
 * @param extra_layers Extra instance layers to enable.
 * @param device_extensions Extra device extensions to enable.
 * @param debug Enable the validation layers and the debug messenger.
 * @return The created objects, or why they could not be created (in which case nothing was).
 */
inline std::expected<GpuVulkanDefault, std::string> gpuSetupDefaultVulkanEXT(
	GPU::function_t<VkSurfaceKHR(VkInstance)> surface_loader,
	GpuErrorCallbackEXT error_callback = gpuDefaultErrorCallbackEXT,
	VkDebugUtilsMessageSeverityFlagBitsEXT severity_filter = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
	std::span<const char*> instance_extensions = {}, std::span<const char*> extra_layers = {},
	std::span<const char*> device_extensions = {}, bool debug = true
) {
	// The callable crosses the C boundary as the userdata of a plain function
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

	VmaAllocator gpu_allocator = VK_NULL_HANDLE;

	// `descriptor_heap` records whether the buffer can be bound as a heap in place, or has to be snapshotted.
	struct Allocation {
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = nullptr;
		VkDeviceSize size = 0;
		bool descriptor_heap = false;
	};
	// Ordered, so "which allocation holds this address" is one upper_bound (see GPU::detail::closest_buffer).
	std::map<VkDeviceAddress, Allocation> allocations;
	// Snapshot heaps made for texture heaps that could not be bound in place, by the address they copy.
	std::unordered_map<VkDeviceAddress, std::tuple<VkBuffer, VmaAllocation, VkDeviceSize, VkDeviceAddress>> descriptor_heaps;
	VkDeviceSize minimum_descriptor_heap_size = 0, sampler_size = 0, image_size = 0;
	VkDeviceSize descriptor_heap_alignment = 0; // resourceHeapAlignment

	// Native descriptor slots per GpuTextureDescriptor, which shaders multiply heap indices by; more than one when the driver's descriptors are narrower.
	uint32_t heap_stride_ratio = 1;
	std::unordered_map<void*, VkDeviceAddress> host2gpu;
	std::unordered_map<VkDeviceAddress, void*> gpu2host;
	std::unordered_map<VkDeviceAddress, GpuTexture*> gpu2texture; // The texture created on each allocation, which goes when the allocation does
	// Still VK_IMAGE_LAYOUT_UNDEFINED; the next command buffer started moves them to GENERAL (see gpuStartCommandRecording).
	std::vector<GpuTexture*> textures_pending_initial_layout;

	std::unordered_map<std::vector<GpuSamplerDesc>, VkDeviceAddress, GpuSamplerDescListHash> sampler_cache;

	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkSemaphore command_submission_timeline_semaphore = VK_NULL_HANDLE;
	uint64_t command_submission_timeline_semaphore_next_value = 1;
	std::vector<std::pair<VkCommandBuffer, uint64_t>> command_buffers_pending_free;

	// gpuBlitTextureEXT samples through a combined image sampler of its own rather than the descriptor heap; built on the first blit.
	std::array<VkShaderModule, 2> blit_shader_modules = {}; // vertex, fragment
	VkDescriptorSetLayout blit_descriptor_set_layout = VK_NULL_HANDLE;
	VkPipelineLayout blit_pipeline_layout = VK_NULL_HANDLE;
	std::array<VkSampler, 2> blit_samplers = {}; // indexed by "linear"
	std::unordered_map<uint64_t, VkPipeline> blit_pipelines; // keyed by destination VkFormat
	std::vector<VkDescriptorPool> blit_descriptor_pools;
	std::vector<VkDescriptorSet> blit_descriptor_sets_free;
	std::vector<std::pair<VkDescriptorSet, uint64_t>> blit_descriptor_sets_in_flight;

	// Views made for render pass attachments and blits, destroyed once the submission using them finishes.
	std::vector<std::pair<VkImageView, uint64_t>> views_in_flight;

	GpuCapabilities capabilities = {};

	// VK_KHR_swapchain_maintenance1: a present can carry a fence, so retired swapchains are freed exactly (see GpuSurface::RetiredSwapchain).
	bool present_fences_supported = false;

	// Messages are deduplicated because they fire per draw, and one report per draw would bury the first.
	GpuDiagnosticCallbackEXT diagnostic_callback = nullptr;
	void* diagnostic_userdata = nullptr;
	std::set<std::string> reported_diagnostics;

	// SIGNAL_ATOMIC_MAX and OR need a real atomic, so they are a one thread dispatch; the operation is a push constant.
	VkShaderModule signal_shader_module = VK_NULL_HANDLE;
	VkPipelineLayout signal_pipeline_layout = VK_NULL_HANDLE;
	VkPipeline signal_pipeline = VK_NULL_HANDLE;
	// Split barrier events, recycled once their submission finishes, since only the host can safely unsignal one.
	std::vector<VkEvent> events_free;
	std::vector<std::pair<VkEvent, uint64_t>> events_in_flight;
};

/**
 * gpuCreateQueue – Wrap the objects gpuSetupDefaultVulkanEXT created as a GpuQueue.
 *
 * @code{.cpp}
 * GpuQueue* second = gpuCreateQueue(backend);
 * assert(second);
 * gpuFreeQueue(second);
 * @endcode
 *
 * @param vulkan The objects to wrap.
 * @param allocator Host allocation hook.
 * @param callbacks Vulkan allocation callbacks, or nullptr.
 * @return The queue, or nullptr with errno set; see the C flavour in noapi.h.
 */
inline GpuQueue* gpuCreateQueue(const GpuVulkanDefault& vulkan, CpuAllocatorFunc allocator = default_::cpu_allocator, VkAllocationCallbacks* callbacks = nullptr) {
	return gpuCreateQueue(vulkan.instance, vulkan.gpu, vulkan.device, vulkan.graphics_queue, vulkan.graphics_queue_family, true, allocator, callbacks);
}

struct GpuPipeline {
	VkPipeline pipeline;
	std::optional<size_t> color_target_count = {}; // Empty for a compute pipeline

	// Blending is dynamic state, so binding a pipeline applies its own description until gpuSetBlendState overrides it.
	std::vector<VkBool32> blend_enables;
	std::vector<VkColorBlendEquationEXT> blend_equations;
	std::vector<uint8_t> color_write_masks; // In the API's own bits, so a blend state can mask them
};

struct GpuTexture {
	VkImage image;
	VkImageView full_view; // Only a surface's own images keep one; render passes build their own
	GpuTextureDesc descriptor;
	VkSemaphore available_semaphore = VK_NULL_HANDLE;
	// Tracked rather than assumed, so a LOAD_OP_LOAD attachment can be transitioned without discarding its contents.
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
	// Reapplied after gpuSetPipeline, which would otherwise replace it with the pipeline's own blending.
	const GpuBlendState* bound_blend_state = nullptr;
	VkDeviceAddress sampler_map = {};
	std::vector<VkSemaphore> wait_semaphores;

	// A signal's event waiting for its gpuWaitBefore. The barrier is kept because vkCmdWaitEvents2 must repeat vkCmdSetEvent2's dependency.
	struct PendingSignal {
		VkDeviceAddress ptr;
		uint64_t value;
		VkEvent event;
		VkMemoryBarrier2 barrier;
	};
	std::vector<PendingSignal> pending_signals;
	std::vector<VkEvent> events_used;
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

	// The submission index after which each semaphore is free again; zero means never used.
	std::vector<uint64_t> image_available_free_after;
	std::vector<uint64_t> render_finished_free_after;

	uint32_t current_image = uint32_t(-1), semaphore_counter = 0;
	uint32_t current_semaphore = uint32_t(-1); // The acquire semaphore the held image was acquired with
	// Acquiring again before presenting would hold more images than allowed (VUID-vkAcquireNextImageKHR-surface-07783).
	bool image_acquired = false;

	// A replaced swapchain the presentation engine may still be using; destroying it is deferred so a resize never stalls.
	struct RetiredSwapchain {
		std::shared_ptr<vkb::Swapchain> swapchain;
		std::vector<VkImageView> views;
		std::vector<VkFence> present_fences;
		// Its semaphores retire with it: a semaphore a present consumed only frees once its image is re-acquired, which never happens.
		std::vector<VkSemaphore> image_available;
		std::vector<VkSemaphore> render_finished;
		uint64_t free_after_submission = 0;
		uint64_t retired_at_present = 0; // Without present fences, it is held until enough later presents have gone out
		uint32_t image_count = 0;
	};
	std::vector<RetiredSwapchain> retired;

	std::vector<VkFence> present_fences;
	uint64_t presents_issued = 0;
	std::vector<VkFence> free_present_fences;
	// From failed presents, so possibly never signalled; only destroyed once the device is known idle.
	std::vector<VkFence> quarantined_present_fences;
};

/**
 * gpuCreateSurfaceEXT – Build a swapchain for the surface gpuSetupDefaultVulkanEXT created and wrap it as a GpuSurface.
 *
 * @code{.cpp}
 * auto surface = gpuCreateSurfaceEXT(queue, backend, {.texture = {.dimensions = {640, 480, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
 * assert((surface != nullptr) == (backend.surface != VK_NULL_HANDLE));
 * if(surface) gpuFreeSurfaceEXT(queue, surface);
 * @endcode
 *
 * @param queue Queue the surface presents from.
 * @param vulkan The objects holding the surface.
 * @param desc Requested configuration; what was actually granted is reported by gpuSurfaceGetConfigurationEXT.
 * @return The surface, or nullptr with errno set if the setup was headless.
 */
inline GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, const GpuVulkanDefault& vulkan, const GpuSurfaceDescriptor& desc) {
	return gpuCreateSurfaceEXT(queue, vulkan.surface, desc);
}
