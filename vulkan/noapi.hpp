#pragma once

// The backend's C interface: the setup helper, the feature/extension requirements, the
// queue and surface constructors and the constants. This file adds the C++ conveniences on
// top of it, along with the definitions of the objects behind the API's opaque handles.
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

	// One gpuMalloc's worth of memory: the buffer it lives in, the allocation backing it, and how
	// wide it is. `descriptor_heap` records whether the buffer carries
	// VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT, which decides whether gpuSetActiveTextureHeapPtr can
	// bind it where it lies or has to snapshot it into one that can be bound.
	struct Allocation {
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = nullptr;
		VkDeviceSize size = 0;
		bool descriptor_heap = false;
	};
	// Mapping from gpu* (Device Addresses) to buffer allocations.
	//
	// Ordered rather than hashed because the lookup that matters is not "is this an allocation
	// base" but "which allocation holds this address", which an ordered map answers with one
	// upper_bound instead of a scan over every live allocation (see GPU::detail::closest_buffer).
	std::map<VkDeviceAddress, Allocation> allocations;
	// Mapping from gpu* (Device Addresses) to a mapped descriptor heap
	std::unordered_map<VkDeviceAddress, std::tuple<VkBuffer, VmaAllocation, VkDeviceSize, VkDeviceAddress>> descriptor_heaps;
	VkDeviceSize minimum_descriptor_heap_size = 0, sampler_size = 0, image_size = 0;
	// What address a resource descriptor heap has to start at (resourceHeapAlignment)
	VkDeviceSize descriptor_heap_alignment = 0;

	/**
	 * How many native descriptor slots one GpuTextureDescriptor spans.
	 *
	 * The heap a program builds is an array of GpuTextureDescriptor, whose width is fixed by the
	 * API, while ResourceDescriptorHeap[] in a shader indexes in units of the driver's own image
	 * descriptor. Those are only the same number on a driver that lays an image out in the full
	 * 64 bytes; everywhere else the shader has to multiply, and the rest of each slot goes unused.
	 */
	uint32_t heap_stride_ratio = 1;
	// Mappings between cpu and gpu pointers
	std::unordered_map<void*, VkDeviceAddress> host2gpu;
	std::unordered_map<VkDeviceAddress, void*> gpu2host;
	// Mapping from gpu* (Device Addresses) to an associated image
	std::unordered_map<VkDeviceAddress, VkImage> gpu2image;

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

	// What this device can do, for the parts of the API that can't be provided everywhere.
	// Filled in by gpuCreateQueue and handed out by gpuGetCapabilitiesEXT.
	GpuCapabilities capabilities = {};

	// Whether VK_KHR_swapchain_maintenance1 is available, which lets a present carry a fence.
	// Without it nothing tells the application when the presentation engine is finished with an
	// image, and a retired swapchain has to be held on a heuristic instead (see
	// GpuSurface::RetiredSwapchain). Not part of GpuCapabilities: it changes how presentation is
	// implemented, not what the API will do for you.
	bool present_fences_supported = false;

	// Where unsupported and emulated calls are reported, and the messages already reported.
	// Deduplicated because these fire from per draw calls, and one report per draw would bury the
	// first one; a set of the messages themselves rather than of call sites, since the message is
	// what distinguishes one cause from another.
	GpuDiagnosticCallbackEXT diagnostic_callback = nullptr;
	void* diagnostic_userdata = nullptr;
	std::set<std::string> reported_diagnostics;

	// gpuSignalAfter's atomics. SIGNAL_ATOMIC_SET is a plain 8 byte write, which
	// vkCmdUpdateBuffer does; MAX and OR need an actual atomic, so they go through a one thread
	// compute dispatch per signal. The pipelines are built on first use and keyed by SIGNAL.
	VkShaderModule signal_shader_module = VK_NULL_HANDLE;
	VkPipelineLayout signal_pipeline_layout = VK_NULL_HANDLE;
	VkPipeline signal_pipeline = VK_NULL_HANDLE; // Which operation runs is a push constant, so one covers both
	// Events backing the split barrier pairs, recycled once the submission that used one has
	// finished (an event has to be unsignalled again before it can be set, and only the host can
	// do that safely, so reuse waits for completion rather than for the next command buffer).
	std::vector<VkEvent> events_free;
	std::vector<std::pair<VkEvent, uint64_t>> events_in_flight;
};
inline GpuQueue* gpuCreateQueue(const GpuVulkanDefault& vulkan, CpuAllocatorFunc allocator = default_::cpu_allocator, VkAllocationCallbacks* callbacks = nullptr) {
	return gpuCreateQueue(vulkan.instance, vulkan.gpu, vulkan.device, vulkan.graphics_queue, vulkan.graphics_queue_family, true, allocator, callbacks);
}

struct GpuPipeline {
	VkPipeline pipeline;
	std::optional<size_t> color_target_count = {}; // When null indicates a compute pipeline

	// The blend configuration the pipeline was created with.
	//
	// Blending is dynamic state here, which means the values in the create info are ignored and
	// something has to set them per command buffer. A pipeline still describes its own blending,
	// so binding one applies these -- and gpuSetBlendState then overrides them for as long as it
	// is in effect. Without this a command buffer that never set a blend state would be drawing
	// with whatever the previous one left behind, or with nothing set at all.
	std::vector<VkBool32> blend_enables;
	std::vector<VkColorBlendEquationEXT> blend_equations;
	std::vector<uint8_t> color_write_masks; // In the API's own bits, so a blend state can mask them
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
	// The blend state explicitly set on this recording, if any. Binding a pipeline applies that
	// pipeline's own blend description, so without remembering this an explicit state would be
	// silently dropped by the next gpuSetPipeline -- making the two calls order dependent.
	const GpuBlendState* bound_blend_state = nullptr;
	VkDeviceAddress sampler_map = {};
	std::vector<VkSemaphore> wait_semaphores;

	// A gpuSignalAfter that recorded an event, waiting to be claimed by the gpuWaitBefore that
	// pairs with it. Matched on the counter and the value, which is what makes a pair a pair.
	//
	// The barrier is kept verbatim because Vulkan requires the dependency a vkCmdWaitEvents2 names
	// to be the one its vkCmdSetEvent2 named. The consumer's stage is therefore not expressible
	// here -- it isn't known yet when the event is set -- which is why the stored dependency ends
	// at every stage and a wait that wants narrower or wants cache invalidation records a second,
	// ordinary barrier of its own after the wait.
	struct PendingSignal {
		VkDeviceAddress ptr;
		uint64_t value;
		VkEvent event;
		VkMemoryBarrier2 barrier;
	};
	std::vector<PendingSignal> pending_signals;
	// Events this recording set, to be recycled once its submission finishes
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

	// When each of those semaphores comes free again, as a value of the queue's submission
	// timeline: the submission that waits on one is the frame that used it, so once the timeline
	// reaches that value the semaphore is idle and can be signalled again. Zero means never used.
	//
	// Without this the semaphores were picked round robin and reused after as many frames as there
	// are swapchain images, which is not the same thing as waiting for those frames to finish -- a
	// CPU running ahead of the GPU would come back to one while a submission was still waiting on
	// it, which is what VUID-vkAcquireNextImageKHR-semaphore-01779 and
	// VUID-vkQueueSubmit2-semaphore-03868 were reporting.
	std::vector<uint64_t> image_available_free_after;
	std::vector<uint64_t> render_finished_free_after;

	uint32_t current_image = uint32_t(-1), semaphore_counter = 0;
	// Which acquire semaphore the image currently held was acquired with, so presenting can record
	// when it comes free (and drain it if nothing ever waited on it)
	uint32_t current_semaphore = uint32_t(-1);
	// Whether an image is acquired and not yet presented. Acquiring again without presenting would
	// hold more images than the surface allows (VUID-vkAcquireNextImageKHR-surface-07783).
	bool image_acquired = false;

	// A swapchain that has been replaced but may not be destroyed yet: the presentation engine can
	// still be displaying images acquired from it, and a submission can still reference its views.
	// Destroying one is therefore deferred rather than waited for, which is what keeps a resize --
	// a run of reconfigurations, one per mouse movement -- from stalling the GPU each time.
	//
	// When the device has VK_KHR_swapchain_maintenance1 the wait is exact: every present carries a
	// fence, and the swapchain is destroyed once all of its presents have signalled. Without it,
	// `retired_at_present` records how many presents this surface had made when the swapchain was
	// retired, and it is held until enough later presents have gone out that the presentation
	// engine cannot still be holding one of its images.
	struct RetiredSwapchain {
		std::shared_ptr<vkb::Swapchain> swapchain;
		std::vector<VkImageView> views;
		std::vector<VkFence> present_fences;
		// The semaphores go with the swapchain they were used against rather than being kept and
		// reused. A binary semaphore a present consumed is not free again until the image comes
		// back round or a present fence says so, and across a reconfiguration neither ever happens:
		// the image is never re-acquired, because the swapchain it belonged to is gone. Keeping them
		// was what left VUID-vkQueueSubmit2-semaphore-03868 to fire once per resize.
		std::vector<VkSemaphore> image_available;
		std::vector<VkSemaphore> render_finished;
		uint64_t free_after_submission = 0;
		uint64_t retired_at_present = 0;
		uint32_t image_count = 0;
	};
	std::vector<RetiredSwapchain> retired;

	// Presents issued on the current swapchain, and the fences that go with them when the device
	// can provide them. Both move into the retirement entry when the swapchain is replaced.
	std::vector<VkFence> present_fences;
	uint64_t presents_issued = 0;
	// Fences that have been signalled and reset, ready to be attached to another present
	std::vector<VkFence> free_present_fences;
	// Fences belonging to a present that failed, which may never be signalled and so can be
	// neither recycled nor destroyed until the device is known idle. Emptied by the forced reclaim.
	std::vector<VkFence> quarantined_present_fences;
};
inline GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, const GpuVulkanDefault& vulkan, const GpuSurfaceDescriptor& desc) {
	return gpuCreateSurfaceEXT(queue, vulkan.surface, desc);
}
