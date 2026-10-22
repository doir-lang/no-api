#pragma once

#include "graphics.h"

// ---------------------------------------------------------------------------
// Surface Extension
// ---------------------------------------------------------------------------

/**
 * GpuSurface – Platform-specific presentation surface / swapchain object.
 *
 * A surface manages the OS window presentation backend (CAMetalLayer, Wayland surface, HWND swapchain, etc.) together with the internally owned presentable textures used for display.
 *
 * Unlike Vulkan's explicit VkSurfaceKHR + VkSwapchainKHR split, this API intentionally follows the simpler WebGPU-style model where the presentation system is represented as a single reconfigurable object.
 */
typedef struct GpuSurface GpuSurface;

/**
 * PRESENT_MODE – Swapchain presentation scheduling behavior.
 *
 * The exact implementation depends on platform compositor support and GPU driver capabilities.
 */
typedef enum PRESENT_MODE {
	/** Present immediately without waiting for vertical sync. Lowest latency but may introduce visible tearing. */
	PRESENT_MODE_IMMEDIATE,

	/** Queue frames and present in FIFO order synchronized to vblank. Guaranteed tear-free and universally supported. */
	PRESENT_MODE_FIFO,

	/** FIFO presentation while the application maintains refresh rate, but allows tearing if frames arrive late instead of stalling for another refresh interval. */
	PRESENT_MODE_FIFO_RELAXED,

	/** Low-latency triple-buffered presentation. Newly rendered frames replace older queued frames while waiting for vblank. Tear-free with lower latency than FIFO when supported. */
	PRESENT_MODE_MAILBOX,

	/** Runtime selects the best supported mode for the current platform, typically preferring MAILBOX, then FIFO_RELAXED, then FIFO. */
	PRESENT_MODE_BEST_AVAILABLE
} PRESENT_MODE;

/**
 * GPU_MAX_SURFACE_FORMATS / GPU_MAX_SURFACE_PRESENT_MODES – Capacity of the lists in GpuSurfaceCapabilities. Fixed rather than allocated so that the struct can be returned by value to a C caller with nothing to free; both are comfortably larger than the number of distinct FORMAT / PRESENT_MODE values a surface can report.
 */
#define GPU_MAX_SURFACE_FORMATS 32
#define GPU_MAX_SURFACE_PRESENT_MODES 4

/**
 * GpuSurfaceCapabilities – Queryable presentation capabilities for a surface.
 *
 * Used to determine which texture formats and presentation modes are supported by the current platform compositor and display backend.
 *
 * Both lists are ordered best first, by the same preference the backend applies when it is left to choose: formats[0] is what a FORMAT_NONE request resolves to, and presentModes[0] is what PRESENT_MODE_BEST_AVAILABLE resolves to. A surface that reports neither list as empty can always be configured with any entry in them.
 */
typedef struct GpuSurfaceCapabilities {
	/**
	 * Supported swapchain texture formats, best first. The first formatCount entries are meaningful.
	 *
	 * Typically includes formats such as FORMAT_BGRA8_SRGB, FORMAT_BGRA8_UNORM, and optionally HDR formats such as FORMAT_RGB10_A2_UNORM. Formats the rest of the API can't name, and formats only offered outside the nonlinear sRGB color space the swapchain is built for, are left out.
	 */
	FORMAT formats[GPU_MAX_SURFACE_FORMATS];
	uint32_t formatCount NOAPI_DEFAULT(0); ///< How many entries of formats are filled in.

	/** Supported presentation scheduling modes, best first. The first presentModeCount entries are meaningful. Never contains PRESENT_MODE_BEST_AVAILABLE, which is a request rather than a mode. */
	PRESENT_MODE presentModes[GPU_MAX_SURFACE_PRESENT_MODES];
	uint32_t presentModeCount NOAPI_DEFAULT(0); ///< How many entries of presentModes are filled in.

	/** True if the compositor supports transparent or alpha-composited windows, which is what a surface configured with `opaque = false` asks for. */
	bool supportsTransparency NOAPI_DEFAULT(false);
} GpuSurfaceCapabilities;

/**
 * GpuSurfaceDescriptor – Presentation surface / swapchain configuration.
 */
typedef struct GpuSurfaceDescriptor {
	/** Descriptor used to create the internally managed presentable textures. Width, height, format, sample count, and usage flags are inherited from this descriptor. Implementations may impose additional restrictions on supported formats/usages depending on platform swapchain limitations. */
	GpuTextureDesc texture NOAPI_DEFAULT({});

	/** Preferred presentation scheduling mode. The runtime may silently fall back to another supported mode if the requested mode is unavailable on the current system. */
	PRESENT_MODE presentMode NOAPI_DEFAULT(PRESENT_MODE_BEST_AVAILABLE);

	/** Hint that the presented image is fully opaque. Allows compositors to skip destination blending work on platforms that support opaque presentation surfaces. */
	bool opaque NOAPI_DEFAULT(true);
} GpuSurfaceDescriptor;

/**
 * GPU_SURFACE_DESCRIPTOR_DEFAULT – The defaults above as an initializer, for C. Keep in sync with GpuSurfaceDescriptor.
 */
#define GPU_SURFACE_DESCRIPTOR_DEFAULT { GPU_TEXTURE_DESC_DEFAULT, PRESENT_MODE_BEST_AVAILABLE, true }

// gpuCreateSurfaceEXT is declared by each backend (vulkan/noapi.h, webgpu/noapi.h), since it is built from the backend's own window surface.

NOAPI_EXTERN_C_BEGIN

/**
 * gpuFreeSurfaceEXT – Destroy a presentation surface and release all associated swapchain resources.
 *
 * The caller must ensure the GPU is no longer rendering to any acquired surface textures before destroying the surface.
 *
 * @code{.cpp}
 * GpuSurface* surface = gpuCreateSurfaceEXT(queue, backend, {.texture = {.dimensions = {640, 480, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
 * if(surface) { // NULL without a window, as on a headless device
 * 	gpuWaitIdleEXT(queue);
 * 	gpuFreeSurfaceEXT(queue, surface);
 * }
 * @endcode
 *
 * @note Does not destroy the native surface (VkSurfaceKHR, WGPUSurface) it was created from.
 *
 * @param queue GPU queue/device the surface was created against.
 * @param surface Surface to destroy.
 */
void gpuFreeSurfaceEXT(GpuQueue* queue, GpuSurface* surface);

/**
 * gpuSurfaceReconfigureEXT – Recreate/reconfigure the surface swapchain.
 *
 * Typically called after:
 * - Window resize
 * - Format changes
 * - HDR mode changes
 * - Presentation mode changes
 *
 * Any previously referenced surface textures become invalid after this call.
 *
 * @code{.cpp}
 * GpuSurface* surface = gpuCreateSurfaceEXT(queue, backend, {.texture = {.dimensions = {640, 480, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
 * if(surface) { // NULL without a window, as on a headless device
 * 	auto config = gpuSurfaceGetConfigurationEXT(surface);
 * 	config.texture.dimensions = {1280, 720, 1}; // The window was resized
 * 	gpuSurfaceReconfigureEXT(queue, surface, config);
 * 	gpuFreeSurfaceEXT(queue, surface);
 * }
 * @endcode
 *
 * @param queue GPU queue/device the surface was created against.
 * @param surface Surface to reconfigure.
 * @param desc New surface configuration; what was actually granted is reported by gpuSurfaceGetConfigurationEXT.
 */
void gpuSurfaceReconfigureEXT(GpuQueue* queue, GpuSurface* surface, NOAPI_CONST_REF(GpuSurfaceDescriptor) desc);

/**
 * gpuGetSurfaceCapabilitiesEXT – Query presentation capabilities for a surface.
 *
 * Used to determine supported formats, present modes and transparency support before configuring the surface. The surface has to exist to be asked, so choosing a format this way means creating the surface (with FORMAT_NONE, so that the backend's own preference is the fallback) and then reconfiguring it with the pick.
 *
 * @code{.cpp}
 * GpuSurface* surface = gpuCreateSurfaceEXT(queue, backend, {.texture = {.dimensions = {640, 480, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
 * if(surface) { // NULL without a window, as on a headless device
 * 	auto capabilities = gpuGetSurfaceCapabilitiesEXT(queue, surface);
 * 	auto config = gpuSurfaceGetConfigurationEXT(surface);
 * 	for(auto format: std::span(capabilities.formats, capabilities.formatCount))
 * 		if(!gpuFormatIsSrgbEXT(format)) {
 * 			config.texture.format = format;
 * 			gpuSurfaceReconfigureEXT(queue, surface, config);
 * 			break;
 * 		}
 * 	gpuFreeSurfaceEXT(queue, surface);
 * }
 * @endcode
 *
 * @param queue GPU queue/device the surface was created against.
 * @param surface Surface to query.
 * @return What the surface supports, each list ordered best first. Both lists are empty if the platform could not be asked.
 */
GpuSurfaceCapabilities gpuGetSurfaceCapabilitiesEXT(GpuQueue* queue, GpuSurface* surface);

/**
 * gpuSurfaceGetConfigurationEXT – Gets the configured properties of the surface.
 *
 * @code{.cpp}
 * GpuSurface* surface = gpuCreateSurfaceEXT(queue, backend, {.texture = {.dimensions = {640, 480, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
 * if(surface) { // NULL without a window, as on a headless device
 * 	auto config = gpuSurfaceGetConfigurationEXT(surface);
 * 	assert(config.texture.format != FORMAT_NONE); // The surface's preference, resolved
 * 	gpuFreeSurfaceEXT(queue, surface);
 * }
 * @endcode
 *
 * @param surface Surface to query.
 * @return The GpuSurfaceDescriptor last used to create or reconfigure the surface, narrowed to what the platform actually granted.
 */
GpuSurfaceDescriptor gpuSurfaceGetConfigurationEXT(const GpuSurface* surface);

/**
 * gpuSurfaceNextTextureEXT – Acquire the next presentable surface texture.
 *
 * The returned texture can be rendered into like any other render target. Ownership remains with the surface and the texture becomes invalid after gpuSurfacePresentEXT or gpuSurfaceReconfigureEXT is called.
 *
 * Returning references avoids per-frame texture handle churn and maps well to fixed swapchain backbuffers, but requires care around surface lifetime and reconfiguration invalidation.
 *
 * A presentable texture is not necessarily a first class texture. Backends that allocate textures out of their own pools (WebGPU, where the swapchain owns the image) cannot place it in the texture heap, so it cannot be named by a GpuTextureDescriptor and cannot be sampled, written as a storage texture, or used as a copy endpoint. Attaching it to a render pass always works, and gpuBlitTextureEXT is the supported bridge to and from ordinary textures: render or dispatch into a texture of your own and blit the result in at the end of the frame. Only the usages gpuSurfaceGetConfigurationEXT reports back were actually granted, which may be fewer than were requested.
 *
 * @code{.cpp}
 * GpuSurface* surface = gpuCreateSurfaceEXT(queue, backend, {.texture = {.dimensions = {640, 480, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
 * if(surface) { // NULL without a window, as on a headless device
 * 	if(auto target = gpuSurfaceNextTextureEXT(queue, surface)) {
 * 		GpuColorAttachment color = {.texture = target, .loadOp = LOAD_OP_CLEAR};
 * 		GpuRenderPassDesc pass = {.colorAttachments = {&color, 1}};
 * 		auto cmd = gpuStartCommandRecording(queue);
 * 		gpuBeginRenderPass(cmd, pass);
 * 		gpuEndRenderPass(cmd, pass);
 * 		gpuSurfacePresentEXT(queue, surface, gpuSubmit(queue, {&cmd, 1}));
 * 	}
 * 	gpuWaitIdleEXT(queue);
 * 	gpuFreeSurfaceEXT(queue, surface);
 * }
 * @endcode
 *
 * @param queue GPU queue/device the surface was created against.
 * @param surface Surface to acquire the next presentable texture from.
 * @return The texture, or NULL if none could be acquired, typically because the surface became invalid or requires reconfiguration after a resize/minimize event; errno is then SURFACE_OUT_OF_DATE or another backend status. A texture handed back with errno set to SURFACE_SUBOPTIMAL is usable, but the surface should be reconfigured soon.
 */
const GpuTexture* gpuSurfaceNextTextureEXT(GpuQueue* queue, GpuSurface* surface);

/**
 * NO_SUBMISSION_WAIT – The submission index gpuSurfacePresentEXT takes to present without waiting for a particular submission.
 */
#ifdef __cplusplus
	constexpr static uint64_t NO_SUBMISSION_WAIT = -1;
#else
	#define NO_SUBMISSION_WAIT ((uint64_t)-1)
#endif

/**
 * gpuSurfacePresentEXT – Queue the currently acquired surface texture for display.
 *
 * Presentation occurs asynchronously according to the configured PRESENT_MODE. After presentation the acquired texture becomes invalid and a new texture must be acquired before rendering the next frame.
 *
 * @code{.cpp}
 * GpuSurface* surface = gpuCreateSurfaceEXT(queue, backend, {.texture = {.dimensions = {640, 480, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
 * if(surface) { // NULL without a window, as on a headless device
 * 	if(auto target = gpuSurfaceNextTextureEXT(queue, surface)) {
 * 		GpuColorAttachment color = {.texture = target, .loadOp = LOAD_OP_CLEAR, .clearValue = {0.1f, 0.1f, 0.1f, 1}};
 * 		GpuRenderPassDesc pass = {.colorAttachments = {&color, 1}};
 * 		auto cmd = gpuStartCommandRecording(queue);
 * 		gpuBeginRenderPass(cmd, pass);
 * 		gpuEndRenderPass(cmd, pass);
 * 		gpuSurfacePresentEXT(queue, surface, gpuSubmit(queue, {&cmd, 1}));
 * 	}
 * 	gpuWaitIdleEXT(queue);
 * 	gpuFreeSurfaceEXT(queue, surface);
 * }
 * @endcode
 *
 * @param queue GPU queue/device the surface was created against.
 * @param surface Surface whose currently acquired texture should be presented.
 * @note A present that fails, typically because the surface is out of date, leaves the backend's status in errno; reconfigure the surface and carry on.
 *
 * @param wait_submission_index Submission index to wait on before presenting (as returned by gpuSubmit), or NO_SUBMISSION_WAIT to present without waiting on a specific submission.
 */
void gpuSurfacePresentEXT(GpuQueue* queue, GpuSurface* surface, uint64_t wait_submission_index NOAPI_DEFAULT(NO_SUBMISSION_WAIT));

NOAPI_EXTERN_C_END
