#pragma once

#include "graphics.hpp"

// ---------------------------------------------------------------------------
// Surface Extension
// ---------------------------------------------------------------------------

/**
 * GpuSurface
 *
 * Platform-specific presentation surface / swapchain object.
 *
 * A surface manages the OS window presentation backend (CAMetalLayer, Wayland
 * surface, HWND swapchain, etc.) together with the internally owned
 * presentable textures used for display.
 *
 * Unlike Vulkan's explicit VkSurfaceKHR + VkSwapchainKHR split, this API
 * intentionally follows the simpler WebGPU-style model where the presentation
 * system is represented as a single reconfigurable object.
 */
struct GpuSurface;

/**
 * PRESENT_MODE – Swapchain presentation scheduling behaviour.
 *
 * The exact implementation depends on platform compositor support and GPU
 * driver capabilities.
 */
enum PRESENT_MODE {
	/**
	 * Present immediately without waiting for vertical sync. Lowest latency
	 * but may introduce visible tearing.
	 */
	PRESENT_MODE_IMMEDIATE,

	/**
	 * Queue frames and present in FIFO order synchronized to vblank.
	 * Guaranteed tear-free and universally supported.
	 */
	PRESENT_MODE_FIFO,

	/**
	 * FIFO presentation while the application maintains refresh rate, but
	 * allows tearing if frames arrive late instead of stalling for another
	 * refresh interval.
	 */
	PRESENT_MODE_FIFO_RELAXED,

	/**
	 * Low-latency triple-buffered presentation. Newly rendered frames replace
	 * older queued frames while waiting for vblank. Tear-free with lower
	 * latency than FIFO when supported.
	 */
	PRESENT_MODE_MAILBOX,

	/**
	 * Runtime selects the best supported mode for the current platform,
	 * typically preferring MAILBOX, then FIFO_RELAXED, then FIFO.
	 */
	PRESENT_MODE_BEST_AVAILABLE
};

/**
 * GpuSurfaceCapabilities – Queryable presentation capabilities for a surface.
 *
 * Used to determine which texture formats and presentation modes are supported
 * by the current platform compositor and display backend.
 */
struct GpuSurfaceCapabilities {
	/**
	 * Supported swapchain texture formats.
	 *
	 * Typically includes formats such as FORMAT_RGBA8_UNORM_SRGB,
	 * FORMAT_BGRA8_UNORM_SRGB, and optionally HDR formats.
	 */
	std::span<const FORMAT> formats;

	/**
	 * Supported presentation scheduling modes.
	 */
	std::span<const PRESENT_MODE> presentModes;

	/**
	 * True if the surface/display path supports HDR presentation formats.
	 */
	bool supportsHDR = false;

	/**
	 * True if the compositor supports transparent or alpha-composited windows.
	 */
	bool supportsTransparency = false;
};

/**
 * GpuSurfaceDescriptor – Presentation surface / swapchain configuration.
 */
struct GpuSurfaceDescriptor {
	/**
	 * Descriptor used to create the internally managed presentable textures.
	 *
	 * Width, height, format, sample count, and usage flags are inherited from
	 * this descriptor. Implementations may impose additional restrictions on
	 * supported formats/usages depending on platform swapchain limitations.
	 */
	GpuTextureDescriptor texture;

	/**
	 * Preferred presentation scheduling mode.
	 *
	 * The runtime may silently fall back to another supported mode if the
	 * requested mode is unavailable on the current system.
	 */
	PRESENT_MODE presentMode = PRESENT_MODE_BEST_AVAILABLE;

	/**
	 * Hint that the presented image is fully opaque.
	 *
	 * Allows compositors to skip destination blending work on platforms that
	 * support opaque presentation surfaces.
	 */
	bool opaque = true;
};

/**
 * gpuCreateSurface – Create a platform presentation surface / swapchain.
 *
 * The exact native windowing parameters are intentionally implementation-
 * defined and may vary per platform backend:
 *
 * - Metal : CAMetalLayer*
 * - Wayland : wl_display* + wl_surface*
 * - Win32 : HINSTANCE + HWND
 * - X11 : Display* + Window
 * - etc.
 *
 * It is recommended to support gpuCreateWindowsSurface, gpuCreateWaylandSurface,
 * etc... functions to prevent a struct extensibility nightmare.
 *
 * @param queue GPU queue/device associated with the surface.
 * @param ... Platform-specific windowing/display handles.
 * @param desc Surface configuration descriptor.
 */
// GpuSurface gpuCreateSurface(/* Platform creation logic up to implementation */);

/**
 * gpuFreeSurface – Destroy a presentation surface and release all associated
 * swapchain resources.
 *
 * The caller must ensure the GPU is no longer rendering to any acquired
 * surface textures before destroying the surface.
 *
 * @param queue GPU queue/device the surface was created against.
 * @param surface Surface to destroy.
 */
void gpuFreeSurfaceEXT(GpuQueue* queue, GpuSurface* surface);

/**
 * gpuSurfaceReconfigure – Recreate/reconfigure the surface swapchain.
 *
 * Typically called after:
 * - Window resize
 * - Format changes
 * - HDR mode changes
 * - Presentation mode changes
 *
 * Any previously referenced surface textures become invalid after this call.
 *
 * @param queue GPU queue/device the surface was created against.
 * @param surface Surface to reconfigure.
 * @param desc New surface configuration.
 */
void gpuSurfaceReconfigure(GpuSurface& surface, const GpuSurfaceDescriptor& desc);

/**
 * gpuGetSurfaceCapabilities – Query presentation capabilities for a surface.
 *
 * Used to determine supported formats, present modes, transparency support,
 * and HDR support before configuring the surface.
 */
GpuSurfaceCapabilities gpuGetSurfaceCapabilities(GpuQueue& queue, GpuSurface& surface);

/**
 * gpuSurfaceGetConfigurationEXT – Gets the configured properties of the surface.
 *
 * @param surface Surface to query.
 * @return The GpuSurfaceDescriptor last used to create or reconfigure the surface.
 */
uvec2 gpuSurfaceGetSize(const GpuSurface& surface);

/**
 * gpuSurfaceGetNextTexture – Acquire the next presentable surface texture.
 *
 * The returned texture can be rendered into like any other render target.
 * Ownership remains with the surface and the texture becomes invalid after
 * gpuSurfacePresent or gpuSurfaceReconfigure is called.
 *
 * Returns std::nullopt if no texture could be acquired, typically because the
 * surface became invalid or requires reconfiguration after a resize/minimize
 * event.
 *
 * NOTE: Returning references avoids per-frame texture handle churn and maps
 * well to fixed swapchain backbuffers, but requires care around surface
 * lifetime and reconfiguration invalidation.
 *
 * @param queue GPU queue/device the surface was created against.
 * @param surface Surface to acquire the next presentable texture from.
 */
std::optional<GpuTexture&> gpuSurfaceGetNextTexture(GpuSurface& surface);

/**
 * gpuSurfacePresent – Queue the currently acquired surface texture for display.
 *
 * Presentation occurs asynchronously according to the configured PRESENT_MODE.
 * After presentation the acquired texture becomes invalid and a new texture
 * must be acquired before rendering the next frame.
 *
 * @param queue GPU queue/device the surface was created against.
 * @param surface Surface whose currently acquired texture should be presented.
 * @param wait_submission_index Submission index to wait on before presenting
 * (as returned by gpuSubmit), or NO_SUBMISSION_WAIT to present without waiting
 * on a specific submission.
 */
void gpuSurfacePresent(GpuSurface& surface);

/**
 * gpuFreeSurface – Destroy a presentation surface and release all associated
 * swapchain resources.
 *
 * The caller must ensure the GPU is no longer rendering to any acquired
 * surface textures before destroying the surface.
 */
void gpuFreeSurface(GpuSurface& surface);
