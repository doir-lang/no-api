#pragma once

// ---------------------------------------------------------------------------
// WebGPU Backend – C interface
// ---------------------------------------------------------------------------

/**
 * The backend specific half of the API: everything a program has to name to get a GpuQueue (and a GpuSurface) out of a WebGPU instance, device and surface.
 *
 * This header, like the API headers it includes, parses as both C and C++, so a C program can drive the whole backend. noapi.hpp includes it and adds the C++ only conveniences (the std::expected returning setup helper, the overloads that take a GpuWebGPUDefault directly) plus the definitions of the backend's internal objects.
 */

#include "../allocator.h"
#include "../samplers.h"
#include "../shaders.h"
#include "../surface.h"
#include "../sync.h"

#include <webgpu/webgpu.h>

NOAPI_EXTERN_C_BEGIN

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/**
 * GpuErrorCallbackEXT – Hook the backend reports device and validation errors through.
 *
 * @param queue The queue the error was raised on, or NULL when it happened before (or outside of) a queue.
 * @param type Backend specific error category (a WGPUErrorType here).
 * @param message Human readable description of the error, not null terminated.
 */
typedef void (*GpuErrorCallbackEXT)(void* queue, int type, GpuStringView message);

/**
 * gpuDefaultErrorCallbackEXT – The error callback used when none is provided: prints the message and terminates the process.
 *
 * @code
 * GpuErrorCallbackEXT report = gpuDefaultErrorCallbackEXT; // What gpuSetupDefaultWebGPUEXT is usually handed
 * assert(report);
 * @endcode
 *
 * @param queue The queue the error was raised on, or NULL.
 * @param type The error's WGPUErrorType.
 * @param message Human readable description of the error, not null terminated.
 */
void gpuDefaultErrorCallbackEXT(void* queue, int type, GpuStringView message);

/**
 * GpuWebGPUDefault – The WebGPU objects gpuSetupDefaultWebGPUEXT creates. The caller owns all of them and releases them (in reverse order) once the queue built on top of them is gone.
 */
typedef struct GpuWebGPUDefault {
	WGPUInstance instance;
	WGPUAdapter adapter;
	WGPULimits limits;
	WGPUDevice device;
	WGPUSurface surface; ///< NULL when set up headless.
} GpuWebGPUDefault;

/**
 * GpuWebGPUSurfaceLoaderEXT – Called by gpuSetupDefaultWebGPUEXT once the instance exists, to bridge whatever windowing library is in use to a WGPUSurface.
 *
 * @param instance The instance the surface has to be created from.
 * @param userdata The pointer handed to gpuSetupDefaultWebGPUEXT alongside the loader.
 */
typedef WGPUSurface (*GpuWebGPUSurfaceLoaderEXT)(WGPUInstance instance, void* userdata);

/**
 * gpuSetupDefaultWebGPUEXT – Create an instance, surface, adapter and device with the settings this backend needs, so that a program only has to supply the window.
 *
 * @code
 * GpuWebGPUDefault webgpu;
 * char error[256];
 * bool created = gpuSetupDefaultWebGPUEXT(NULL, NULL, gpuDefaultErrorCallbackEXT, true, &webgpu, error, sizeof(error));
 * assert(created);
 * assert(webgpu.surface == NULL); // No surface loader, so headless
 *
 * wgpuDeviceRelease(webgpu.device);
 * wgpuAdapterRelease(webgpu.adapter);
 * wgpuInstanceRelease(webgpu.instance);
 * @endcode
 *
 * @note C++ callers can instead use the overload in noapi.hpp, which takes any callable as the surface loader and reports failure as a std::expected.
 *
 * @param surface_loader Creates the presentation surface from the new instance. NULL, or a loader that returns no surface, sets up a headless device that can still render into textures.
 * @param surface_loader_userdata Passed back to \p surface_loader untouched.
 * @param error_callback Where device and validation errors are reported (gpuDefaultErrorCallbackEXT is the usual choice).
 * @param prefer_high_power Ask for the discrete GPU rather than the integrated one.
 * @param out_default Filled in with the created objects on success.
 * @param out_error Buffer the failure reason is written to as a null terminated string, or NULL to discard it.
 * @param out_error_capacity Size of \p out_error in bytes.
 * @return True on success; on failure nothing is created, \p out_default is left as it was, and \p out_error explains why.
 */
bool gpuSetupDefaultWebGPUEXT(GpuWebGPUSurfaceLoaderEXT surface_loader, void* surface_loader_userdata,
	GpuErrorCallbackEXT error_callback, bool prefer_high_power,
	GpuWebGPUDefault* out_default, char* out_error, size_t out_error_capacity);

// ---------------------------------------------------------------------------
// Addresses
// ---------------------------------------------------------------------------

/**
 * gpu_sampler_slot_count – How many sampler slots shaders are compiled against.
 *
 * WebGPU can't index an array of samplers, so every slot gets its own binding and shaders pick between them with a switch over the slot the lookup map handed back. Shaders are compiled against however many slots exist, so this is the 16 maxSamplersPerShaderStage guarantees rather than whatever a particular device offers.
 */
#ifdef __cplusplus
	constexpr static uint32_t gpu_sampler_slot_count = 16;
#else
	#define gpu_sampler_slot_count ((uint32_t)16)
#endif

/**
 * GpuWebGPUAddressEXT – A gpu pointer taken apart: which monobuffer it lives in, and how far into it. What gpuDecodeWebGPUAddressEXT hands back.
 */
typedef struct GpuWebGPUAddressEXT {
	uint8_t monobuffer;
	uint64_t address;
} GpuWebGPUAddressEXT;

/**
 * gpuEncodeWebGPUAddressEXT – Pack a monobuffer index and an offset into it into the gpu pointer shaders receive.
 *
 * @code
 * gpu* pointer = gpuEncodeWebGPUAddressEXT(1, 256);
 * assert(pointer);
 * assert(gpuDecodeWebGPUAddressEXT(pointer).monobuffer == 1);
 * @endcode
 *
 * @param monobuffer Which monobuffer the address is in.
 * @param address Byte offset into that monobuffer.
 */
gpu* gpuEncodeWebGPUAddressEXT(uint8_t monobuffer, uint64_t address);

/**
 * gpuDecodeWebGPUAddressEXT – The inverse of gpuEncodeWebGPUAddressEXT.
 *
 * @code
 * void* values = gpuMalloc(queue, 64, 16, MEMORY_DEFAULT);
 * GpuWebGPUAddressEXT parts = gpuDecodeWebGPUAddressEXT(gpuHostToDevicePointer(queue, values));
 * assert(gpuEncodeWebGPUAddressEXT(parts.monobuffer, parts.address) == gpuHostToDevicePointer(queue, values));
 * gpuFree(queue, values);
 * @endcode
 *
 * @param addr A gpu pointer this backend handed out.
 */
GpuWebGPUAddressEXT gpuDecodeWebGPUAddressEXT(gpu* addr);

// ---------------------------------------------------------------------------
// Queue and surface creation
// ---------------------------------------------------------------------------

/**
 * gpuCreateQueue – Wrap an existing WebGPU device (and the adapter and limits it was created from) as a GpuQueue.
 *
 * @code
 * GpuQueue* second = gpuCreateQueue(backend.adapter, backend.device, backend.limits, gpuDefaultCpuAllocator);
 * assert(second);
 * gpuFreeQueue(second);
 * @endcode
 *
 * @note Command buffers created from a queue may internally store pointers to it, so it is unsafe to move a queue after sub objects have been created.
 *
 * @param adapter Adapter the device came from.
 * @param device Device all GPU objects are created on.
 * @param limits The limits the device was created with.
 * @param allocator Host allocation hook (gpuDefaultCpuAllocator by default).
 * @return The queue. Device errors are reported through the error callback the device was created with.
 */
GpuQueue* gpuCreateQueue(WGPUAdapter adapter, WGPUDevice device, WGPULimits limits,
	CpuAllocatorFunc allocator NOAPI_DEFAULT(gpuDefaultCpuAllocator));

/**
 * gpuCreateSurfaceEXT – Configure an existing WGPUSurface for presentation and wrap it as a GpuSurface.
 *
 * @code
 * GpuSurfaceDescriptor desc = GPU_SURFACE_DESCRIPTOR_DEFAULT;
 * desc.texture.usage = USAGE_COLOR_ATTACHMENT;
 * GpuSurface* surface = gpuCreateSurfaceEXT(queue, backend.surface, NOAPI_REF(desc));
 * assert((surface != NULL) == (backend.surface != NULL));
 * if(surface) gpuFreeSurfaceEXT(queue, surface);
 * @endcode
 *
 * @param queue Queue the surface presents from.
 * @param surface The platform surface to configure.
 * @param desc Requested configuration; what was actually granted is reported by gpuSurfaceGetConfigurationEXT.
 * @return The surface, or NULL with errno set if \p surface is NULL.
 */
GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, WGPUSurface surface, NOAPI_CONST_REF(GpuSurfaceDescriptor) desc);

/**
 * SURFACE_SUBOPTIMAL / SURFACE_OUT_OF_DATE – Statuses gpuSurfaceNextTextureEXT leaves in errno. A suboptimal texture is still perfectly usable (it just no longer matches the window), an out of date one means the configuration has to be redone.
 */
#ifdef __cplusplus
	constexpr static int SURFACE_SUBOPTIMAL = WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal;
	constexpr static int SURFACE_OUT_OF_DATE = WGPUSurfaceGetCurrentTextureStatus_Outdated;
#else
	#define SURFACE_SUBOPTIMAL WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal
	#define SURFACE_OUT_OF_DATE WGPUSurfaceGetCurrentTextureStatus_Outdated
#endif

NOAPI_EXTERN_C_END
