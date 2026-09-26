#pragma once

// ---------------------------------------------------------------------------
// Vulkan Backend – C interface
// ---------------------------------------------------------------------------

/**
 * The backend specific half of the API: everything a program has to name to get a
 * GpuQueue (and a GpuSurface) out of a Vulkan instance, device and surface, plus the
 * features and extensions this backend requires of that device.
 *
 * This header, like the API headers it includes, parses as both C and C++, so a C
 * program can drive the whole backend. noapi.hpp includes it and adds the C++ only
 * conveniences (the std::expected returning setup helper, the overloads that take a
 * GpuVulkanDefault directly) plus the definitions of the backend's internal objects.
 */

#include "../allocator.h"
#include "../samplers.h"
#include "../surface.h"
#include "../sync.h"

#include <vulkan/vulkan_core.h>

NOAPI_EXTERN_C_BEGIN

/**
 * GpuErrorCallbackEXT – Hook the backend reports device and validation errors through.
 *
 * @param queue The queue the error was raised on, or NULL when it happened before (or
 * outside of) a queue.
 * @param type Backend specific error category (VkDebugUtilsMessageTypeFlagsEXT here).
 * @param message Human readable description of the error, not null terminated.
 */
typedef void (*GpuErrorCallbackEXT)(void* queue, int type, GpuStringView message);

/**
 * gpuDefaultErrorCallbackEXT – The error callback used when none is provided: prints the
 * message along with its message type.
 */
void gpuDefaultErrorCallbackEXT(void* queue, int type, GpuStringView message);

/**
 * COMPUTE_SHADER_PROLOGUE / GRAPHICS_SHADER_PROLOGUE – GLSL this backend expects to be
 * prepended to a shader: the version and extension declarations, the push constant block
 * holding whatever root pointers the dispatch or draw was given, and the sampler lookup
 * helpers.
 */
extern const char* const COMPUTE_SHADER_PROLOGUE;
extern const char* const GRAPHICS_SHADER_PROLOGUE;

/**
 * GpuVulkanDefault – The Vulkan objects gpuSetupDefaultVulkanEXT creates. The caller owns
 * all of them and destroys them (in reverse order) once the queue built on top of them is
 * gone.
 */
typedef struct GpuVulkanDefault {
	VkInstance instance NOAPI_DEFAULT(VK_NULL_HANDLE);
	VkDebugUtilsMessengerEXT messenger NOAPI_DEFAULT(VK_NULL_HANDLE);
	VkSurfaceKHR surface NOAPI_DEFAULT(VK_NULL_HANDLE);
	VkPhysicalDevice gpu NOAPI_DEFAULT(VK_NULL_HANDLE);
	VkDevice device NOAPI_DEFAULT(VK_NULL_HANDLE);
	VkQueue graphics_queue NOAPI_DEFAULT(VK_NULL_HANDLE);
	uint32_t graphics_queue_family;
} GpuVulkanDefault;

/**
 * GpuVulkanSurfaceLoaderEXT – Called by gpuSetupDefaultVulkanEXT once the instance exists,
 * to bridge whatever windowing library is in use to a VkSurfaceKHR.
 *
 * @param instance The instance the surface has to be created from.
 * @param userdata The pointer handed to gpuSetupDefaultVulkanEXT alongside the loader.
 */
typedef VkSurfaceKHR (*GpuVulkanSurfaceLoaderEXT)(VkInstance instance, void* userdata);

/**
 * gpuSetupDefaultVulkanEXT – Create an instance, surface, physical device, device and
 * queue with the features and extensions this backend needs, so that a program only has to
 * supply the window.
 *
 * @note C++ callers can instead use the overload in noapi.hpp, which takes any callable as
 * the surface loader and reports failure as a std::expected.
 *
 * @param surface_loader Creates the presentation surface from the new instance.
 * @param surface_loader_userdata Passed back to \p surface_loader untouched.
 * @param error_callback Where validation messages are reported
 * (gpuDefaultErrorCallbackEXT is the usual choice).
 * @param severity_filter Messages below this severity are dropped.
 * @param instance_extensions Extra instance extensions to enable (those the windowing
 * library asks for).
 * @param extra_layers Extra instance layers to enable.
 * @param device_extensions Extra device extensions to enable, on top of
 * gpuRequiredVulkanDeviceExtensionsEXT.
 * @param debug Enable the validation layers and the debug messenger.
 * @param out_default Filled in with the created objects on success.
 * @param out_error Buffer the failure reason is written to as a null terminated string, or
 * NULL to discard it.
 * @param out_error_capacity Size of \p out_error in bytes.
 * @return True on success; on failure nothing is created and \p out_error explains why.
 */
bool gpuSetupDefaultVulkanEXT(GpuVulkanSurfaceLoaderEXT surface_loader, void* surface_loader_userdata,
	GpuErrorCallbackEXT error_callback, VkDebugUtilsMessageSeverityFlagBitsEXT severity_filter,
	GpuCStringSpan instance_extensions, GpuCStringSpan extra_layers, GpuCStringSpan device_extensions, bool debug,
	GpuVulkanDefault* out_default, char* out_error, size_t out_error_capacity);

/**
 * gpuEnableRequiredVulkanFeaturesEXT – Turns on the core features this backend needs in a
 * feature struct of your own, so that a hand rolled device creation enables them too.
 */
NOAPI_INLINE VkPhysicalDeviceFeatures gpuEnableRequiredVulkanFeaturesEXT(VkPhysicalDeviceFeatures features) {
	features.shaderInt64 = true;
	return features;
}

/**
 * gpuEnableRequiredVulkan12FeaturesEXT – The Vulkan 1.2 features this backend needs.
 */
NOAPI_INLINE VkPhysicalDeviceVulkan12Features gpuEnableRequiredVulkan12FeaturesEXT(VkPhysicalDeviceVulkan12Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	features.bufferDeviceAddress = true;
	features.timelineSemaphore = true;
	return features;
}

/**
 * gpuEnableRequiredVulkan13FeaturesEXT – The Vulkan 1.3 features this backend needs.
 */
NOAPI_INLINE VkPhysicalDeviceVulkan13Features gpuEnableRequiredVulkan13FeaturesEXT(VkPhysicalDeviceVulkan13Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	features.dynamicRendering = true;
	features.synchronization2 = true;
	return features;
}

/**
 * gpuEnableRequiredVulkan14FeaturesEXT – The Vulkan 1.4 features this backend needs.
 */
NOAPI_INLINE VkPhysicalDeviceVulkan14Features gpuEnableRequiredVulkan14FeaturesEXT(VkPhysicalDeviceVulkan14Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
	// features.pushDescriptor = true;
	features.indexTypeUint8 = true;
	return features;
}

/**
 * gpuRequiredVulkanDeviceExtensionsEXT – The device extensions this backend can't do
 * without, as a span of statically allocated strings.
 */
NOAPI_INLINE GpuCStringSpan gpuRequiredVulkanDeviceExtensionsEXT(void) {
	static const char* const extensions[] = {"VK_EXT_descriptor_heap", "VK_KHR_shader_untyped_pointers"};

	GpuCStringSpan out;
	out.ptr = extensions;
	out.count = sizeof(extensions) / sizeof(extensions[0]);
	return out;
}

/**
 * gpuRequiredVulkanDeviceCreateInfoPnextEXT – A pNext chain turning on the feature bits
 * those extensions come with, to be linked into your own VkDeviceCreateInfo.
 */
NOAPI_INLINE void* gpuRequiredVulkanDeviceCreateInfoPnextEXT(void) {
	static VkPhysicalDeviceDescriptorHeapFeaturesEXT descriptor_heap_info = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
		.descriptorHeap = true
	};
	static VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untyped_pointers_info = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
		.pNext = &descriptor_heap_info,
		.shaderUntypedPointers = true
	};
	return &untyped_pointers_info;
}

/**
 * gpuCreateQueue – Wrap an existing Vulkan device and queue as a GpuQueue.
 *
 * @note CommandBuffers created from a queue may internally store pointers to it, so it is
 * unsafe to move a queue after sub objects have been created.
 *
 * @param instance The instance the device came from.
 * @param gpu The physical device the device was created on.
 * @param device Device all GPU objects are created on.
 * @param queue The queue work is submitted to, or VK_NULL_HANDLE to look one up.
 * @param queue_family The family that queue belongs to, or -1 to look one up.
 * @param is_graphics_queue False for a compute only queue.
 * @param allocator Host allocation hook (gpuDefaultCpuAllocator by default).
 * @param callbacks Vulkan allocation callbacks, or NULL.
 */
GpuQueue* gpuCreateQueue(VkInstance instance, VkPhysicalDevice gpu, VkDevice device,
	VkQueue queue NOAPI_DEFAULT(VK_NULL_HANDLE), uint32_t queue_family NOAPI_DEFAULT(-1),
	bool is_graphics_queue NOAPI_DEFAULT(true), CpuAllocatorFunc allocator NOAPI_DEFAULT(gpuDefaultCpuAllocator),
	VkAllocationCallbacks* callbacks NOAPI_DEFAULT(NULL));

/**
 * gpuCreateSurfaceEXT – Build a swapchain for an existing VkSurfaceKHR and wrap it as a
 * GpuSurface.
 *
 * @param queue Queue the surface presents from.
 * @param surface The platform surface to build on.
 * @param desc Requested configuration; what was actually granted is reported by
 * gpuSurfaceGetConfigurationEXT.
 */
GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, VkSurfaceKHR surface, NOAPI_CONST_REF(GpuSurfaceDescriptor) desc);

/**
 * SURFACE_SUBOPTIMAL / SURFACE_OUT_OF_DATE – Statuses gpuSurfaceNextTextureEXT leaves in
 * errno. A suboptimal texture is still perfectly usable (it just no longer matches the
 * window), an out of date one means the configuration has to be redone.
 */
#ifdef __cplusplus
	constexpr static int SURFACE_SUBOPTIMAL = VK_SUBOPTIMAL_KHR;
	constexpr static int SURFACE_OUT_OF_DATE = VK_ERROR_OUT_OF_DATE_KHR;
#else
	#define SURFACE_SUBOPTIMAL VK_SUBOPTIMAL_KHR
	#define SURFACE_OUT_OF_DATE VK_ERROR_OUT_OF_DATE_KHR
#endif

NOAPI_EXTERN_C_END
