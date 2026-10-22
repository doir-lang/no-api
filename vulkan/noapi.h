#pragma once

// ---------------------------------------------------------------------------
// Vulkan Backend – C interface
// ---------------------------------------------------------------------------

/**
 * The backend specific half of the API: everything a program has to name to get a GpuQueue (and a GpuSurface) out of a Vulkan instance, device and surface, plus the features and extensions this backend requires of that device.
 *
 * This header, like the API headers it includes, parses as both C and C++, so a C program can drive the whole backend. noapi.hpp includes it and adds the C++ only conveniences (the std::expected returning setup helper, the overloads that take a GpuVulkanDefault directly) plus the definitions of the backend's internal objects.
 */

#include "../allocator.h"
#include "../samplers.h"
#include "../shaders.h"
#include "../surface.h"
#include "../sync.h"

#include <vulkan/vulkan_core.h>

NOAPI_EXTERN_C_BEGIN

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/**
 * GpuErrorCallbackEXT – Hook the backend reports device and validation errors through.
 *
 * @param queue The queue the error was raised on, or NULL when it happened before (or outside of) a queue.
 * @param type Backend specific error category (VkDebugUtilsMessageTypeFlagsEXT here).
 * @param message Human readable description of the error, not null terminated.
 */
typedef void (*GpuErrorCallbackEXT)(void* queue, int type, GpuStringView message);

/**
 * gpuDefaultErrorCallbackEXT – The error callback used when none is provided: prints the message along with its message type.
 *
 * @code
 * const char* text = "an example of a validation message";
 * GpuStringView message = {text, strlen(text)};
 * gpuDefaultErrorCallbackEXT(NULL, VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, message);
 * @endcode
 *
 * @param queue The queue the error was raised on, or NULL.
 * @param type The message's VkDebugUtilsMessageTypeFlagsEXT.
 * @param message Human readable description of the error, not null terminated.
 */
void gpuDefaultErrorCallbackEXT(void* queue, int type, GpuStringView message);

/**
 * GpuVulkanDefault – The Vulkan objects gpuSetupDefaultVulkanEXT creates. The caller owns all of them and destroys them (in reverse order) once the queue built on top of them is gone.
 */
typedef struct GpuVulkanDefault {
	VkInstance instance NOAPI_DEFAULT(VK_NULL_HANDLE);
	VkDebugUtilsMessengerEXT messenger NOAPI_DEFAULT(VK_NULL_HANDLE); ///< VK_NULL_HANDLE unless set up with debug.
	VkSurfaceKHR surface NOAPI_DEFAULT(VK_NULL_HANDLE); ///< VK_NULL_HANDLE when set up headless.
	VkPhysicalDevice gpu NOAPI_DEFAULT(VK_NULL_HANDLE);
	VkDevice device NOAPI_DEFAULT(VK_NULL_HANDLE);
	VkQueue graphics_queue NOAPI_DEFAULT(VK_NULL_HANDLE);
	uint32_t graphics_queue_family;
} GpuVulkanDefault;

/**
 * GpuVulkanSurfaceLoaderEXT – Called by gpuSetupDefaultVulkanEXT once the instance exists, to bridge whatever windowing library is in use to a VkSurfaceKHR.
 *
 * @param instance The instance the surface has to be created from.
 * @param userdata The pointer handed to gpuSetupDefaultVulkanEXT alongside the loader.
 */
typedef VkSurfaceKHR (*GpuVulkanSurfaceLoaderEXT)(VkInstance instance, void* userdata);

/**
 * gpuSetupDefaultVulkanEXT – Create an instance, surface, physical device, device and queue with the features and extensions this backend needs, so that a program only has to supply the window.
 *
 * @code
 * GpuCStringSpan none = {NULL, 0};
 * GpuVulkanDefault vulkan;
 * char error[256];
 * bool created = gpuSetupDefaultVulkanEXT(NULL, NULL, gpuDefaultErrorCallbackEXT, VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
 * 	none, none, none, true, &vulkan, error, sizeof(error));
 * assert(created);
 * assert(vulkan.surface == VK_NULL_HANDLE); // No surface loader, so headless
 *
 * vkDestroyDevice(vulkan.device, NULL);
 * vkDestroyDebugUtilsMessengerEXT(vulkan.instance, vulkan.messenger, NULL);
 * vkDestroyInstance(vulkan.instance, NULL);
 * @endcode
 *
 * @note C++ callers can instead use the overload in noapi.hpp, which takes any callable as the surface loader and reports failure as a std::expected.
 *
 * @param surface_loader Creates the presentation surface from the new instance. NULL, or a loader that returns no surface, sets up a headless device that can still render into textures.
 * @param surface_loader_userdata Passed back to \p surface_loader untouched.
 * @param error_callback Where validation messages are reported (gpuDefaultErrorCallbackEXT is the usual choice).
 * @param severity_filter Messages below this severity are dropped.
 * @param instance_extensions Extra instance extensions to enable (those the windowing library asks for).
 * @param extra_layers Extra instance layers to enable.
 * @param device_extensions Extra device extensions to enable, on top of gpuRequiredVulkanDeviceExtensionsEXT.
 * @param debug Enable the validation layers and the debug messenger.
 * @param out_default Filled in with the created objects on success.
 * @param out_error Buffer the failure reason is written to as a null terminated string, or NULL to discard it.
 * @param out_error_capacity Size of \p out_error in bytes.
 * @return True on success; on failure nothing is created, \p out_default is left as it was, and \p out_error explains why.
 */
bool gpuSetupDefaultVulkanEXT(GpuVulkanSurfaceLoaderEXT surface_loader, void* surface_loader_userdata,
	GpuErrorCallbackEXT error_callback, VkDebugUtilsMessageSeverityFlagBitsEXT severity_filter,
	GpuCStringSpan instance_extensions, GpuCStringSpan extra_layers, GpuCStringSpan device_extensions, bool debug,
	GpuVulkanDefault* out_default, char* out_error, size_t out_error_capacity);

// ---------------------------------------------------------------------------
// Device requirements
// ---------------------------------------------------------------------------

/**
 * gpuEnableRequiredVulkanFeaturesEXT – Turns on the core features this backend needs in a feature struct of your own, so that a hand rolled device creation enables them too.
 *
 * imageCubeArray is what TEXTURE_CUBE_ARRAY costs: without it a view of several cubes can't be made.
 *
 * shaderStorageImageWriteWithoutFormat is what gpuStoreTexture costs: a shader reaches a texture by a heap index, so the format of the one it is writing is not knowable where the image is declared, and an unformatted storage image write is the feature that covers exactly that. Its read counterpart is not asked for, and gpuStoreTexture has no load beside it, because shaderStorageImageReadWithoutFormat is a feature devices in circulation decline.
 *
 * @code
 * VkPhysicalDeviceFeatures features;
 * memset(&features, 0, sizeof(features));
 * features = gpuEnableRequiredVulkanFeaturesEXT(features);
 * assert(features.shaderInt64 && features.imageCubeArray && features.shaderStorageImageWriteWithoutFormat);
 * @endcode
 *
 * @param features The features the program wants for its own sake.
 * @return \p features with this backend's requirements turned on as well.
 */
NOAPI_INLINE VkPhysicalDeviceFeatures gpuEnableRequiredVulkanFeaturesEXT(VkPhysicalDeviceFeatures features) {
	features.shaderInt64 = true;
	features.imageCubeArray = true;
	features.shaderStorageImageWriteWithoutFormat = true;
	return features;
}

/**
 * gpuEnableRequiredVulkan11FeaturesEXT – The Vulkan 1.1 features this backend needs.
 *
 * shaderDrawParameters is what SV_VertexID costs: Slang gives it the HLSL meaning, the index of the vertex within the draw, which on Vulkan is gl_VertexIndex less the draw's base vertex. Reading that base needs this feature.
 *
 * @code
 * VkPhysicalDeviceVulkan11Features features;
 * memset(&features, 0, sizeof(features));
 * features = gpuEnableRequiredVulkan11FeaturesEXT(features);
 * assert(features.sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES && features.shaderDrawParameters);
 * @endcode
 *
 * @param features The features the program wants for its own sake.
 * @return \p features with this backend's requirements turned on as well, and its sType filled in.
 */
NOAPI_INLINE VkPhysicalDeviceVulkan11Features gpuEnableRequiredVulkan11FeaturesEXT(VkPhysicalDeviceVulkan11Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	features.shaderDrawParameters = true;
	return features;
}

/**
 * gpuEnableRequiredVulkan12FeaturesEXT – The Vulkan 1.2 features this backend needs.
 *
 * shaderBufferInt64Atomics is what gpuSignalAfter costs: SIGNAL_ATOMIC_MAX and SIGNAL_ATOMIC_OR are atomic read-modify-writes of the 64 bit counter the split barrier pair is built on, which nothing narrower can express. A device without it still gets SIGNAL_ATOMIC_SET (a plain write), and the other two report themselves as unsupported through the diagnostic callback; gpuGetCapabilitiesEXT().split_barrier_signals says which of the two situations you are in.
 *
 * @code
 * VkPhysicalDeviceVulkan12Features features;
 * memset(&features, 0, sizeof(features));
 * features = gpuEnableRequiredVulkan12FeaturesEXT(features);
 * assert(features.bufferDeviceAddress && features.timelineSemaphore);
 * @endcode
 *
 * @param features The features the program wants for its own sake.
 * @return \p features with this backend's requirements turned on as well, and its sType filled in.
 */
NOAPI_INLINE VkPhysicalDeviceVulkan12Features gpuEnableRequiredVulkan12FeaturesEXT(VkPhysicalDeviceVulkan12Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	features.bufferDeviceAddress = true;
	features.timelineSemaphore = true;
	features.shaderBufferInt64Atomics = true;
	return features;
}

/**
 * gpuEnableRequiredVulkan13FeaturesEXT – The Vulkan 1.3 features this backend needs.
 *
 * @code
 * VkPhysicalDeviceVulkan13Features features;
 * memset(&features, 0, sizeof(features));
 * features = gpuEnableRequiredVulkan13FeaturesEXT(features);
 * assert(features.dynamicRendering && features.synchronization2);
 * @endcode
 *
 * @param features The features the program wants for its own sake.
 * @return \p features with this backend's requirements turned on as well, and its sType filled in.
 */
NOAPI_INLINE VkPhysicalDeviceVulkan13Features gpuEnableRequiredVulkan13FeaturesEXT(VkPhysicalDeviceVulkan13Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	features.dynamicRendering = true;
	features.synchronization2 = true;
	return features;
}

/**
 * gpuEnableRequiredVulkan14FeaturesEXT – The Vulkan 1.4 features this backend needs.
 *
 * @code
 * VkPhysicalDeviceVulkan14Features features;
 * memset(&features, 0, sizeof(features));
 * features = gpuEnableRequiredVulkan14FeaturesEXT(features);
 * assert(features.indexTypeUint8);
 * @endcode
 *
 * @param features The features the program wants for its own sake.
 * @return \p features with this backend's requirements turned on as well, and its sType filled in.
 */
NOAPI_INLINE VkPhysicalDeviceVulkan14Features gpuEnableRequiredVulkan14FeaturesEXT(VkPhysicalDeviceVulkan14Features features) {
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
	features.indexTypeUint8 = true;
	return features;
}

/**
 * gpuRequiredVulkanDeviceExtensionsEXT – The device extensions this backend can't do without, as a span of statically allocated strings.
 *
 * @code
 * GpuCStringSpan required = gpuRequiredVulkanDeviceExtensionsEXT();
 * assert(required.count > 0); // Enable each of required.ptr on a device of the program's own
 * @endcode
 */
NOAPI_INLINE GpuCStringSpan gpuRequiredVulkanDeviceExtensionsEXT(void) {
	static const char* const extensions[] = {"VK_EXT_descriptor_heap", "VK_KHR_shader_untyped_pointers", "VK_EXT_extended_dynamic_state3"};

	GpuCStringSpan out;
	out.ptr = extensions;
	out.count = sizeof(extensions) / sizeof(extensions[0]);
	return out;
}

/**
 * gpuOptionalVulkanDeviceExtensionsEXT – The device extensions the backend uses when they are there and does without when they are not, as a span of statically allocated strings.
 *
 * VK_KHR_swapchain_maintenance1 lets a present carry a fence, which is the only way to be told when the presentation engine has finished with an image. With it, a swapchain replaced by a window resize is destroyed exactly when it falls idle; without it, it is held until enough later frames have been presented that it cannot still be in use.
 *
 * VK_EXT_mesh_shader is what gpuCreateGraphicsMeshletPipeline and the two gpuDrawMeshlets calls are made of; there is no way to emulate a mesh stage without it, so where it is missing gpuGetCapabilitiesEXT().mesh_shaders reports false and those three calls decline. Only the mesh stage is asked for, not the task stage: this API dispatches mesh thread groups directly and has no amplification stage to put in front of them.
 *
 * @code
 * GpuCStringSpan optional = gpuOptionalVulkanDeviceExtensionsEXT();
 * assert(optional.count == 2);
 * @endcode
 *
 * @note A hand rolled device creation should enable these too -- alongside the matching feature bits, which vkb::PhysicalDevice::enable_extension_features_if_present or an equivalent check supplies. The backend decides what to use from what the *physical* device reports, so a device that skips one of these while its physical device supports it will be asked to do something it was not set up for.
 */
NOAPI_INLINE GpuCStringSpan gpuOptionalVulkanDeviceExtensionsEXT(void) {
	static const char* const extensions[] = {"VK_KHR_swapchain_maintenance1", "VK_EXT_mesh_shader"};

	GpuCStringSpan out;
	out.ptr = extensions;
	out.count = sizeof(extensions) / sizeof(extensions[0]);
	return out;
}

/**
 * gpuRequiredVulkanDeviceCreateInfoPnextEXT – A pNext chain turning on the feature bits those extensions come with, to be linked into your own VkDeviceCreateInfo.
 *
 * @code
 * VkDeviceCreateInfo info;
 * memset(&info, 0, sizeof(info));
 * info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
 * info.pNext = gpuRequiredVulkanDeviceCreateInfoPnextEXT();
 * assert(((const VkBaseInStructure*)info.pNext)->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT);
 * @endcode
 *
 * @note The chain is static and shared: link your own structures in front of it rather than appending to it.
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
	// gpuCreateBlendState and gpuSetBlendState rest on these: this API has no way to compile blending into each pipeline
	static VkPhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_state3_info = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
		.pNext = &untyped_pointers_info,
		.extendedDynamicState3ColorBlendEnable = true,
		.extendedDynamicState3ColorBlendEquation = true,
		.extendedDynamicState3ColorWriteMask = true
	};
	return &dynamic_state3_info;
}

// ---------------------------------------------------------------------------
// Queue and surface creation
// ---------------------------------------------------------------------------

/**
 * gpuCreateQueue – Wrap an existing Vulkan device and queue as a GpuQueue.
 *
 * The device may be one of the program's own, made with the requirements above, and works alongside the devices other queues are on.
 *
 * @code
 * GpuQueue* second = gpuCreateQueue(backend.instance, backend.gpu, backend.device, backend.graphics_queue, backend.graphics_queue_family, true, gpuDefaultCpuAllocator, NULL);
 * assert(second);
 * gpuFreeQueue(second);
 * @endcode
 *
 * @note Command buffers created from a queue may internally store pointers to it, so it is unsafe to move a queue after sub objects have been created.
 * @note The backend reaches Vulkan through volk, which this loads for \p instance, through the loader's entry points that dispatch on whichever device they are handed, so queues on any number of devices can coexist. Create them all from the one instance.
 *
 * @param instance The instance the device came from.
 * @param gpu The physical device the device was created on.
 * @param device Device all GPU objects are created on.
 * @param queue The queue work is submitted to, or VK_NULL_HANDLE to look one up (along with its family).
 * @param queue_family The family that queue belongs to; ignored when the queue is looked up.
 * @param is_graphics_queue False for a compute only queue. Looking one up needs a queue family that does compute without graphics.
 * @param allocator Host allocation hook (gpuDefaultCpuAllocator by default).
 * @param callbacks Vulkan allocation callbacks, or NULL.
 * @return The queue, or NULL if no queue could be looked up (errno VK_ERROR_FEATURE_NOT_PRESENT), the device lacks something the backend needs or an object could not be created, with errno set to the VkResult.
 */
GpuQueue* gpuCreateQueue(VkInstance instance, VkPhysicalDevice gpu, VkDevice device,
	VkQueue queue NOAPI_DEFAULT(VK_NULL_HANDLE), uint32_t queue_family NOAPI_DEFAULT(-1),
	bool is_graphics_queue NOAPI_DEFAULT(true), CpuAllocatorFunc allocator NOAPI_DEFAULT(gpuDefaultCpuAllocator),
	VkAllocationCallbacks* callbacks NOAPI_DEFAULT(NULL));

/**
 * gpuCreateSurfaceEXT – Build a swapchain for an existing VkSurfaceKHR and wrap it as a GpuSurface.
 *
 * @code
 * GpuSurfaceDescriptor desc = GPU_SURFACE_DESCRIPTOR_DEFAULT;
 * desc.texture.usage = USAGE_COLOR_ATTACHMENT;
 * GpuSurface* surface = gpuCreateSurfaceEXT(queue, backend.surface, NOAPI_REF(desc));
 * assert((surface != NULL) == (backend.surface != VK_NULL_HANDLE));
 * if(surface) gpuFreeSurfaceEXT(queue, surface);
 * @endcode
 *
 * @param queue Queue the surface presents from.
 * @param surface The platform surface to build on.
 * @param desc Requested configuration; what was actually granted is reported by gpuSurfaceGetConfigurationEXT.
 * @return The surface, or NULL with errno set if \p surface is VK_NULL_HANDLE.
 */
GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, VkSurfaceKHR surface, NOAPI_CONST_REF(GpuSurfaceDescriptor) desc);

/**
 * SURFACE_SUBOPTIMAL / SURFACE_OUT_OF_DATE – Statuses gpuSurfaceNextTextureEXT leaves in errno. A suboptimal texture is still perfectly usable (it just no longer matches the window), an out of date one means the configuration has to be redone.
 */
#ifdef __cplusplus
	constexpr static int SURFACE_SUBOPTIMAL = VK_SUBOPTIMAL_KHR;
	constexpr static int SURFACE_OUT_OF_DATE = VK_ERROR_OUT_OF_DATE_KHR;
#else
	#define SURFACE_SUBOPTIMAL VK_SUBOPTIMAL_KHR
	#define SURFACE_OUT_OF_DATE VK_ERROR_OUT_OF_DATE_KHR
#endif

NOAPI_EXTERN_C_END
