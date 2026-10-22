
#pragma once

/**
 * A minimalistic modern GPU API based on the design described in: "No Graphics API" by Sebastian Aaltonen https://www.sebastianaaltonen.com/blog/no-graphics-api
 *
 * Design philosophy:
 * - All shader data inputs are passed as a single 64-bit GPU pointer (analogous to how OS threading APIs pass a single void* to thread functions).
 * - Textures are globally accessible via a flat descriptor heap indexed by 32-bit integers; no per-draw descriptor set management is required.
 * - Memory is managed via a CUDA-style malloc/free API that returns CPU-mapped GPU pointers (UMA / PCIe ReBAR), avoiding the Vulkan object-first allocation chicken-and-egg problem.
 * - Barriers describe only producer/consumer execution stages and a small set of hazard flags. Individual resource state tracking (as in DX12 / Vulkan 1.x) is not needed on modern coherent-L2 GPUs.
 * - The entire public surface fits in ~150 lines.
 */

#include "compat.h"

#ifdef __cplusplus
	#include <algorithm>
	#include <string_view>
#endif

// ---------------------------------------------------------------------------
// Basic helpers used by the API
// ---------------------------------------------------------------------------

/**
 * uvec3 – Three unsigned integers: a texture's dimensions, or a grid of thread groups.
 */
typedef struct uvec3 { uint32_t x, y, z; } uvec3;

#ifdef __cplusplus
/**
 * string_to_bytes – Reinterpret a string_view's characters as a read-only byte span, useful for passing shader IR source text to functions expecting bytes.
 *
 * @code{.cpp}
 * auto bytes = string_to_bytes("import noapi;");
 * assert(bytes.size() == 13);
 * @endcode
 *
 * @param str String to view as bytes.
 *
 * @note C++ only sugar; a C caller builds the GpuByteSpan itself.
 */
inline std::span<const std::byte> string_to_bytes(std::string_view str) {
	return {reinterpret_cast<const std::byte*>(str.data()), str.size()};
}

/**
 * byte_span – Reinterpret a span of T elements as a read-only byte span.
 *
 * @code{.cpp}
 * const uint32_t words[2] = {1, 2};
 * assert(byte_span(std::span<const uint32_t>(words)).size() == 8);
 * @endcode
 *
 * @tparam T Element type of the input span.
 * @param span Span to view as bytes.
 */
template<typename T>
std::span<const std::byte> byte_span(std::span<const T> span) {
	return {reinterpret_cast<const std::byte*>(span.data()), span.size_bytes()};
}
#endif

// ---------------------------------------------------------------------------
// Opaque GPU object handles
// ---------------------------------------------------------------------------

/**
 * gpu – Semantically equivalent to void. Some functions will expect a gpu* instead of a void* to indicate that the data should be on the gpu.
 */
typedef struct gpu { char size_equals_one; } gpu;

/**
 * GpuQueue – Represents a GPU submission queue (graphics, compute, or copy). Work is recorded into GpuCommandBuffer objects and submitted via gpuSubmit.
 */
typedef struct GpuQueue GpuQueue;

/**
 * GpuTexture – CPU-side handle for a GPU texture allocation. Needed because the triangle rasterizer is not yet fully bindless on modern GPUs: the CPU driver must prepare render-target and depth/stencil command packets. Created by gpuCreateTexture. The 512-bit texture descriptor written into the descriptor heap is a separate GpuTextureDescriptor value (see below).
 */
typedef struct GpuTexture GpuTexture;

/**
 * GpuPipeline – Represents a compiled shader pipeline (compute, vertex+pixel, or mesh+pixel). Created by gpuCreateComputePipeline / gpuCreateGraphicsPipeline / gpuCreateGraphicsMeshletPipeline. Contains hardware-specific shader microcode. Freed with gpuFreePipeline.
 */
typedef struct GpuPipeline GpuPipeline;

/**
 * GpuCommandBuffer – A transient (one-shot) recording of GPU commands. Created by gpuStartCommandRecording and consumed by gpuSubmit. Reuse across frames is intentionally not supported: one-shot buffers simplify driver memory management (bump allocator vs. heap allocator) and avoid accidentally replaying stale work.
 */
typedef struct GpuCommandBuffer GpuCommandBuffer;

/**
 * GpuCommandBufferSpan – A read only list of command buffers, as gpuSubmit takes them: std::span<GpuCommandBuffer* const> in C++, a {ptr, count} pair in C.
 */
NOAPI_SPAN_TYPE(GpuCommandBufferSpan, GpuCommandBuffer* const);

/**
 * GpuSemaphore – A monotonically-increasing 64-bit timeline counter for GPU<->CPU and GPU<->GPU synchronization. Semantically identical to DX12 / Vulkan 1.2 timeline semaphores. A single semaphore and an ever-incrementing counter value replace the older Vulkan/Metal per-submit fence objects and N-buffering patterns. Created by gpuCreateSemaphore(initialValue); destroyed by gpuFreeSemaphore.
 */
typedef struct GpuSemaphore GpuSemaphore;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/**
 * MEMORY – GPU memory heap type for gpuMalloc.
 *
 * MEMORY_DEFAULT – CPU-mapped GPU memory (write-combined). The CPU can write directly to the returned pointer, and the GPU reads it fast via PCIe ReBAR or UMA. Optimal for small, frequently-updated data: draw arguments, per-frame uniforms, GPU pointers, and texture/buffer descriptors.
 *
 * MEMORY_GPU – GPU-private memory. The CPU cannot write to these pointers directly. Data must first be placed in MEMORY_DEFAULT (an "upload" region) and then copied with gpuMemCpy / gpuCopyToTexture. The driver can apply vendor-specific lossless compression (DCC, delta color compression) and Morton/tile swizzling for textures, maximizing bandwidth.
 *
 * MEMORY_READBACK – CPU-cached memory suitable for GPU->CPU readback. Slower for the GPU to write due to cache-coherency with the CPU. Mainly used for screenshots and virtual-texturing feedback.
 *
 * MEMORY_TEXTURE – Semantically identical to MEMORY_GPU but with internal optimizations that make the WebGPU emulation layer perform better when used for texture storage.
 *
 * MEMORY_TEXTURE_READBACK – Semantically identical to MEMORY_READBACK but with the same internal optimizations as MEMORY_TEXTURE.
 *
 * MEMORY_DESCRIPTOR_HEAP – CPU-mapped GPU memory, like MEMORY_DEFAULT, that may also be handed to gpuSetActiveTextureHeapPtr as a texture descriptor heap. Whether this is a distinct kind of memory or just MEMORY_DEFAULT under another name is a property of the device: some drivers accept descriptor heaps in any memory they accept a buffer in, and others confine them to their own memory types. gpuGetCapabilitiesEXT reports which of those a device is, via `descriptor_heap_in_any_memory`; when it is true this behaves exactly like MEMORY_DEFAULT, and when it is false only memory allocated this way can be bound as a heap in place. Allocating every heap this way is correct either way, which is why it is spelled as a memory type rather than a capability the caller has to branch on.
 */
typedef enum MEMORY { MEMORY_DEFAULT, MEMORY_GPU, MEMORY_READBACK, MEMORY_TEXTURE, MEMORY_TEXTURE_READBACK, MEMORY_DESCRIPTOR_HEAP } MEMORY;

/**
 * OP – Comparison operator used by depth tests, stencil tests, and semaphore wait conditions (gpuWaitBefore).
 */
typedef enum OP {
	OP_NEVER, ///< Test always fails.
	OP_LESS, ///< Passes if incoming < reference.
	OP_EQUAL, ///< Passes if incoming == reference.
	OP_LESS_EQUAL, ///< Passes if incoming <= reference. Default for depth test.
	OP_GREATER, ///< Passes if incoming > reference.
	OP_NOT_EQUAL, ///< Passes if incoming != reference.
	OP_GREATER_EQUAL, ///< Passes if incoming >= reference.
	OP_ALWAYS, ///< Test always passes.
} OP;

/**
 * TEXTURE – Texture dimensionality / view type. A cube has six layers per cube, so TEXTURE_CUBE takes a layerCount of 6 and TEXTURE_CUBE_ARRAY a multiple of 6.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.type = TEXTURE_CUBE, .dimensions = {16, 16, 1}, .layerCount = 6, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * assert(gpuCreateTexture(queue, desc, memory));
 * gpuFree(queue, memory);
 * @endcode
 */
typedef enum TEXTURE {
	TEXTURE_1D, ///< Not supported on WebGPU.
	TEXTURE_2D,
	TEXTURE_3D,
	TEXTURE_CUBE,
	TEXTURE_2D_ARRAY,
	TEXTURE_CUBE_ARRAY,
} TEXTURE;

/**
 * FORMAT – Pixel / texel formats. The set is the one WebGPU names, since that is the narrowest of the backends; every entry has a Vulkan equivalent. FORMAT_NONE is used to indicate "no attachment" (e.g. no depth buffer).
 *
 * The entries marked EXT ride on an optional WebGPU feature, so a device is free to reject them: the 16 bit unorm/snorm ones need `texture-formats-tier1` and FORMAT_D32_FLOAT_S8_UINT needs `depth32float-stencil8`. A Vulkan device may lack some too (FORMAT_D24_PLUS is optional there). gpuTextureSizeAlign reports a size of 0 for a format the device can't make.
 *
 * @code
 * GpuTextureDesc desc = GPU_TEXTURE_DESC_DEFAULT;
 * desc.dimensions.x = desc.dimensions.y = 64;
 * desc.format = FORMAT_R16_UNORM;
 * desc.usage = USAGE_SAMPLED;
 * if(gpuTextureSizeAlign(queue, NOAPI_REF(desc)).size == 0) desc.format = FORMAT_R16_FLOAT; // An optional format this device lacks
 * assert(gpuTextureSizeAlign(queue, NOAPI_REF(desc)).size > 0);
 * @endcode
 */
typedef enum FORMAT {
	FORMAT_NONE,

	// 1 byte per texel
	FORMAT_R8_UNORM,
	FORMAT_R8_SNORM,
	FORMAT_R8_UINT,
	FORMAT_R8_SINT,

	// 2 bytes per texel
	FORMAT_R16_UNORM, // EXT
	FORMAT_R16_SNORM, // EXT
	FORMAT_R16_UINT,
	FORMAT_R16_SINT,
	FORMAT_R16_FLOAT,
	FORMAT_RG8_UNORM,
	FORMAT_RG8_SNORM,
	FORMAT_RG8_UINT,
	FORMAT_RG8_SINT,

	// 4 bytes per texel
	FORMAT_R32_UINT,
	FORMAT_R32_SINT,
	FORMAT_R32_FLOAT,
	FORMAT_RG16_UNORM, // EXT
	FORMAT_RG16_SNORM, // EXT
	FORMAT_RG16_UINT,
	FORMAT_RG16_SINT,
	FORMAT_RG16_FLOAT,
	FORMAT_RGBA8_UNORM,
	FORMAT_RGBA8_SRGB,
	FORMAT_RGBA8_SNORM,
	FORMAT_RGBA8_UINT,
	FORMAT_RGBA8_SINT,
	// The layout most presentation surfaces prefer, and on some platforms the only one they offer
	FORMAT_BGRA8_UNORM,
	FORMAT_BGRA8_SRGB,

	// 4 bytes per texel, components packed at uneven widths
	FORMAT_RGB10_A2_UINT,
	FORMAT_RGB10_A2_UNORM,
	FORMAT_RG11B10_UFLOAT,
	FORMAT_RGB9E5_UFLOAT,

	// 8 bytes per texel
	FORMAT_RG32_UINT,
	FORMAT_RG32_SINT,
	FORMAT_RG32_FLOAT,
	FORMAT_RGBA16_UNORM, // EXT
	FORMAT_RGBA16_SNORM, // EXT
	FORMAT_RGBA16_UINT,
	FORMAT_RGBA16_SINT,
	FORMAT_RGBA16_FLOAT,

	// 16 bytes per texel
	FORMAT_RGBA32_UINT,
	FORMAT_RGBA32_SINT,
	FORMAT_RGBA32_FLOAT,

	// FORMAT_D24_PLUS is "at least 24 bits of depth", all WebGPU promises; the implementation picks the actual layout.
	FORMAT_S8_UINT,
	FORMAT_D16_UNORM,
	FORMAT_D24_PLUS,
	FORMAT_D24_PLUS_S8_UINT,
	FORMAT_D32_FLOAT,
	FORMAT_D32_FLOAT_S8_UINT, // EXT
} FORMAT;

NOAPI_EXTERN_C_BEGIN

/**
 * gpuFormatIsDepthEXT – Returns true if `format` carries a depth aspect. A stencil-only format (FORMAT_S8_UINT) is not one of these; see gpuFormatIsDepthStencilEXT.
 *
 * @code
 * assert(gpuFormatIsDepthEXT(FORMAT_D24_PLUS_S8_UINT));
 * assert(!gpuFormatIsDepthEXT(FORMAT_S8_UINT));
 * @endcode
 *
 * @param format Format to test.
 */
NOAPI_INLINE bool gpuFormatIsDepthEXT(FORMAT format) {
	switch (format) {
	case FORMAT_D16_UNORM:
	case FORMAT_D24_PLUS:
	case FORMAT_D24_PLUS_S8_UINT:
	case FORMAT_D32_FLOAT:
	case FORMAT_D32_FLOAT_S8_UINT:
		return true;
	default: return false;
	}
}

/**
 * gpuFormatIsStencilEXT – Returns true if `format` includes a stencil component.
 *
 * @code
 * assert(gpuFormatIsStencilEXT(FORMAT_D32_FLOAT_S8_UINT));
 * assert(!gpuFormatIsStencilEXT(FORMAT_D32_FLOAT));
 * @endcode
 *
 * @param format Format to test.
 */
NOAPI_INLINE bool gpuFormatIsStencilEXT(FORMAT format) {
	switch (format) {
	case FORMAT_S8_UINT:
	case FORMAT_D24_PLUS_S8_UINT:
	case FORMAT_D32_FLOAT_S8_UINT:
		return true;
	default: return false;
	}
}

/**
 * gpuFormatIsDepthStencilEXT – Returns true if `format` carries either a depth or a stencil aspect, and so can't be used where a plain color format is expected.
 *
 * @code
 * assert(gpuFormatIsDepthStencilEXT(FORMAT_S8_UINT));
 * assert(!gpuFormatIsDepthStencilEXT(FORMAT_RGBA8_UNORM));
 * @endcode
 *
 * @param format Format to test.
 */
NOAPI_INLINE bool gpuFormatIsDepthStencilEXT(FORMAT format) {
	return gpuFormatIsDepthEXT(format) || gpuFormatIsStencilEXT(format);
}

/**
 * gpuFormatIsSrgbEXT – Returns true if `format` stores sRGB encoded texels, so that a sampler decodes to linear on the way in and a render target encodes on the way out.
 *
 * @code
 * assert(gpuFormatIsSrgbEXT(FORMAT_BGRA8_SRGB));
 * assert(!gpuFormatIsSrgbEXT(FORMAT_BGRA8_UNORM));
 * @endcode
 *
 * @param format Format to test.
 */
NOAPI_INLINE bool gpuFormatIsSrgbEXT(FORMAT format) {
	return format == FORMAT_RGBA8_SRGB || format == FORMAT_BGRA8_SRGB;
}

/**
 * gpuFormatIsFilterableEXT – Returns true if a sampler may interpolate between texels of `format`. Integer formats are never filterable, a depth/stencil one only reads back through a comparison sampler, and filtering a 32 bit float image is optional on Vulkan and an opt-in feature on WebGPU, so both backends agree to never offer it.
 *
 * @code
 * assert(gpuFormatIsFilterableEXT(FORMAT_RGBA16_FLOAT));
 * assert(!gpuFormatIsFilterableEXT(FORMAT_RGBA32_FLOAT));
 * @endcode
 *
 * @param format Format to test.
 */
NOAPI_INLINE bool gpuFormatIsFilterableEXT(FORMAT format) {
	switch (format) {
	case FORMAT_R8_UINT:
	case FORMAT_R8_SINT:
	case FORMAT_R16_UINT:
	case FORMAT_R16_SINT:
	case FORMAT_RG8_UINT:
	case FORMAT_RG8_SINT:
	case FORMAT_R32_UINT:
	case FORMAT_R32_SINT:
	case FORMAT_R32_FLOAT:
	case FORMAT_RG16_UINT:
	case FORMAT_RG16_SINT:
	case FORMAT_RGBA8_UINT:
	case FORMAT_RGBA8_SINT:
	case FORMAT_RGB10_A2_UINT:
	case FORMAT_RG32_UINT:
	case FORMAT_RG32_SINT:
	case FORMAT_RG32_FLOAT:
	case FORMAT_RGBA16_UINT:
	case FORMAT_RGBA16_SINT:
	case FORMAT_RGBA32_UINT:
	case FORMAT_RGBA32_SINT:
	case FORMAT_RGBA32_FLOAT:
		return false;
	default: return !gpuFormatIsDepthStencilEXT(format);
	}
}

/**
 * gpuFormatIsStorableEXT – Returns true if a texture of `format` can be created with USAGE_STORAGE and written by gpuStoreTexture. The store takes a float4, which only float, unorm and snorm formats accept, and WGSL names storage for just these of them.
 *
 * @code
 * assert(gpuFormatIsStorableEXT(FORMAT_RGBA16_FLOAT));
 * assert(!gpuFormatIsStorableEXT(FORMAT_R32_UINT));
 * @endcode
 *
 * @param format Format to test.
 */
NOAPI_INLINE bool gpuFormatIsStorableEXT(FORMAT format) {
	switch (format) {
	case FORMAT_R32_FLOAT:
	case FORMAT_RG32_FLOAT:
	case FORMAT_RGBA8_UNORM:
	case FORMAT_RGBA8_SNORM:
	case FORMAT_RGBA16_FLOAT:
	case FORMAT_RGBA32_FLOAT:
		return true;
	default: return false;
	}
}

NOAPI_EXTERN_C_END

/**
 * TEXTURE_USAGE_FLAGS – Bitmask describing how a GpuTexture allocation will be used. The driver uses these flags to choose the appropriate memory layout and enable hardware features (DCC compression, HiZ, MSAA resolve, etc.). Combine with bitwise-OR.
 */
typedef enum TEXTURE_USAGE_FLAGS {
	USAGE_SAMPLED = 0x01, ///< Readable by texture samplers.
	USAGE_STORAGE = 0x02, ///< Read/write access from compute shaders. Only for formats gpuFormatIsStorableEXT accepts.
	USAGE_COLOR_ATTACHMENT = 0x04, ///< Rasterizer color render target.
	USAGE_DEPTH_STENCIL_ATTACHMENT = 0x08, ///< Rasterizer depth/stencil target.
	USAGE_TRANSFER_SRC = 0x10, ///< Source of copy operations.
	USAGE_TRANSFER_DST = 0x20, ///< Destination of copy operations.
} TEXTURE_USAGE_FLAGS;

/**
 * STAGE – GPU pipeline stage used in barrier producer/consumer descriptions and gpuSignalAfter / gpuWaitBefore split-barrier commands.
 *
 * Modern GPUs only require stage-level dependency tracking. Individual resource state transitions (as in Vulkan image layouts / DX12 resource states) are unnecessary on coherent-L2 architectures and are not part of this API.
 *
 * Combine multiple stages with bitwise-OR to express "any of these stages".
 */
typedef enum STAGE {
	STAGE_TRANSFER = 0x001, ///< DMA / copy engine (gpuMemCpy, gpuCopyToTexture).
	STAGE_COMPUTE = 0x002, ///< Compute shader execution.
	STAGE_VERTEX_SHADER = 0x004, ///< Vertex or mesh shader execution.
	STAGE_PIXEL_SHADER = 0x008, ///< Pixel / fragment shader execution.
	STAGE_RASTER_COLOR_OUT = 0x010, ///< Rasterizer color output (ROPs / framebuffer writes).
	STAGE_RASTER_DEPTH_OUT = 0x020, ///< Rasterizer depth/stencil output.
	STAGE_ALL = 0x03F, ///< All stages (conservative; use sparingly).
} STAGE;

/**
 * HAZARD_FLAGS – Optional bitmask passed to gpuBarrier / gpuWaitBefore to request flushing or invalidation of special non-coherent GPU caches that are not automatically flushed by every barrier.
 *
 * Most barriers between compute and raster stages do not need any flags. These flags exist because GPU hardware hides a small number of non-coherent caches (descriptor caches in texture samplers, command-processor prefetch buffers, HiZ / depth-metadata caches) that require explicit invalidation in specific scenarios.
 */
typedef enum HAZARD_FLAGS {
	/**
	 * HAZARD_DRAW_ARGUMENTS – Stall the GPU command-processor's prefetch until the producing stage has finished writing draw/dispatch argument buffers. Required when a compute dispatch writes indirect draw arguments that are consumed by a subsequent gpuDispatchIndirect or gpuDrawIndexedInstancedIndirect.
	 */
	HAZARD_DRAW_ARGUMENTS = 0x1,

	/**
	 * HAZARD_DESCRIPTORS – Invalidate texture-sampler descriptor caches. Required when a compute or CPU write updates entries in the global texture descriptor heap between uses. Normal reads/writes to GPU memory do not touch the sampler's internal descriptor cache; this flag is the only way to force a re-fetch of the updated descriptors.
	 */
	HAZARD_DESCRIPTORS = 0x2,

	/**
	 * HAZARD_DEPTH_STENCIL – Invalidate depth metadata caches (HiZ, stencil cache). Required after a compute shader writes to memory that will subsequently be consumed by the depth/stencil unit as a depth buffer.
	 */
	HAZARD_DEPTH_STENCIL = 0x4,
} HAZARD_FLAGS;

/**
 * SIGNAL – Atomic operation applied when gpuSignalAfter writes to the counter pointer in GPU memory.
 *
 * SIGNAL_ATOMIC_SET – Unconditionally write the value (non-atomic store).
 * SIGNAL_ATOMIC_MAX – Atomically update if the new value is greater (implements timeline semaphore semantics when combined with gpuWaitBefore + OP_GREATER_EQUAL).
 * SIGNAL_ATOMIC_OR – Atomically OR the value into the counter (enables multi-producer bitmask patterns: wait until all producers have set their bit).
 */
typedef enum SIGNAL { SIGNAL_ATOMIC_SET, SIGNAL_ATOMIC_MAX, SIGNAL_ATOMIC_OR } SIGNAL;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/**
 * GpuTextureDesc – Description of a GPU texture allocation. Passed to gpuTextureSizeAlign (to query the required memory footprint) and then to gpuCreateTexture (to obtain a GpuTexture CPU handle for rasterization binding).
 */
typedef struct GpuTextureDesc {
	TEXTURE type NOAPI_DEFAULT(TEXTURE_2D); ///< Dimensionality and view type.
	uvec3 dimensions NOAPI_DEFAULT({1, 1, 1}); ///< Dimensions in texels.
	uint32_t mipCount NOAPI_DEFAULT(1); ///< Number of mip levels (1 = no mips).
	uint32_t layerCount NOAPI_DEFAULT(1); ///< Array layer count (for TEXTURE_2D_ARRAY / TEXTURE_CUBE_ARRAY).
	uint32_t sampleCount NOAPI_DEFAULT(1); ///< MSAA sample count.
	FORMAT format NOAPI_DEFAULT(FORMAT_NONE); ///< Texel format.
	TEXTURE_USAGE_FLAGS usage NOAPI_DEFAULT((TEXTURE_USAGE_FLAGS)0); ///< Bitmask of TEXTURE_USAGE_FLAGS values describing intended use.
} GpuTextureDesc;

/**
 * GPU_TEXTURE_DESC_DEFAULT – The defaults above as an initializer, for C (where default member initializers don't exist). Keep in sync with GpuTextureDesc.
 */
#define GPU_TEXTURE_DESC_DEFAULT { TEXTURE_2D, {1, 1, 1}, 1, 1, 1, FORMAT_NONE, (TEXTURE_USAGE_FLAGS)0 }

/**
 * ALL_MIPS / ALL_LAYERS – GpuViewDesc counts meaning "from the base to the last level/layer".
 */
#ifdef __cplusplus
	static constexpr uint8_t ALL_MIPS = 0xff;
	static constexpr uint16_t ALL_LAYERS = 0xffff;
#else
	#define ALL_MIPS ((uint8_t)0xff)
	#define ALL_LAYERS ((uint16_t)0xffff)
#endif

/**
 * GpuViewDesc – Describes a sub-range of a GpuTexture for use in a descriptor. Passed to gpuTextureViewDescriptor / gpuRWTextureViewDescriptor to create a 512-bit descriptor blob that is stored in the global texture descriptor heap.
 */
typedef struct GpuViewDesc {
	FORMAT format NOAPI_DEFAULT(FORMAT_NONE); ///< Override format, or FORMAT_NONE to use the texture's own format.
	uint8_t baseMip NOAPI_DEFAULT(0); ///< Index of the first mip level included in the view.
	uint8_t mipCount NOAPI_DEFAULT(ALL_MIPS); ///< Number of mip levels included (ALL_MIPS = all remaining).
	uint16_t baseLayer NOAPI_DEFAULT(0); ///< First array layer / cube face included in the view.
	uint16_t layerCount NOAPI_DEFAULT(ALL_LAYERS); ///< Number of layers included (ALL_LAYERS = all remaining).
} GpuViewDesc;

/**
 * GPU_VIEW_DESC_DEFAULT – The defaults above as an initializer, for C. Keep in sync with GpuViewDesc.
 */
#define GPU_VIEW_DESC_DEFAULT { FORMAT_NONE, 0, ALL_MIPS, 0, ALL_LAYERS }

/**
 * GpuTextureSizeAlign – Returned by gpuTextureSizeAlign. Provides the byte size and required alignment for the GPU memory block that must be passed to gpuMalloc before calling gpuCreateTexture. The driver accounts for vendor-specific Morton swizzling, DCC metadata, and alignment padding.
 */
typedef struct GpuTextureSizeAlign {
	size_t size; ///< Total allocation size in bytes (including metadata).
	size_t align; ///< Required alignment in bytes for the gpuMalloc call.
} GpuTextureSizeAlign;

/**
 * GpuTextureDescriptor – A 512-bit opaque hardware-specific texture descriptor blob.
 *
 * This is the "raw descriptor" that GPUs load into scalar registers (AMD) or index into the sampler descriptor heap (Nvidia, Apple, Qualcomm). The user writes these blobs directly into a gpuMalloc'd array of GpuTextureDescriptor objects — the global texture heap. The CPU reads and writes that array through the mapped pointer it was allocated with, and no additional API object stands between the two, unlike DX12's descriptor heap copy APIs.
 *
 * Whether a shader can write the array too is a device property, reported as gpuGetCapabilitiesEXT().gpu_writable_texture_heap. Where it is true the heap is bound where it was allocated and a descriptor a shader writes is seen by the draws and dispatches after the next barrier carrying HAZARD_DESCRIPTORS. Where it is false the backend binds a snapshot taken when gpuSetActiveTextureHeapPtr was called, so a shader's writes are not visible until the heap is set again. Allocate the heap as MEMORY_DESCRIPTOR_HEAP to get the in place binding wherever the device offers it.
 *
 * 512 bits rather than the 256 an AMD image descriptor occupies, because that is not the widest one in circulation: Intel lays a sampled image out in 512 and requires it aligned to 512 as well, so a 256 bit slot could neither hold one nor sit where one may start. The width is fixed rather than queried because it is the stride of an array the program allocates and indexes itself, and an index a shader takes has to mean the same thing on every device. A backend whose descriptors are narrower leaves the rest of each slot unused and scales the index it hands the hardware.
 *
 * Created by gpuTextureViewDescriptor (sampled, read-only) or gpuRWTextureViewDescriptor (storage / read-write).
 */
typedef struct GpuTextureDescriptor { NOAPI_ALIGNAS(64) uint64_t data[8]; } GpuTextureDescriptor;

// ---------------------------------------------------------------------------
// Queue destruction
// ---------------------------------------------------------------------------

NOAPI_EXTERN_C_BEGIN

// gpuCreateQueue is declared by each backend (vulkan/noapi.h, webgpu/noapi.h), since what it is built from is the backend's own device.

/**
 * gpuFreeQueue – Destroy every object the queue allocated internally. Memory still allocated from it is freed with it, along with the textures created on that memory; free everything else created from the queue (pipelines, state objects, semaphores, surfaces) before calling it.
 *
 * @code{.cpp}
 * GpuQueue* second = gpuCreateQueue(backend);
 * assert(second);
 * void* memory = gpuMalloc(second, 64);
 * assert(memory); // Freed along with the queue
 * gpuFreeQueue(second);
 * @endcode
 *
 * @note Does not destroy the backend objects (instance, device, ...) the queue was created from; those still belong to the caller.
 *
 * @param queue Queue to destroy.
 */
void gpuFreeQueue(GpuQueue* queue);

// ---------------------------------------------------------------------------
// Capabilities and diagnostics
// ---------------------------------------------------------------------------

/**
 * GpuCapabilities – What a device and backend can actually do, for the parts of this API that cannot be provided everywhere.
 *
 * Every field answers a question the API would otherwise answer by failing at record time. A call that isn't supported reports through the diagnostic callback and does nothing rather than aborting, so a program that never reads this struct still runs -- it just renders wrong. Read it to branch before you record.
 */
typedef struct GpuCapabilities {
	/**
	 * gpuCreateGraphicsMeshletPipeline, gpuDrawMeshlets and gpuDrawMeshletsIndirect do something. False on WebGPU always, and on a Vulkan device without VK_EXT_mesh_shader.
	 *
	 * When false the pipeline creation returns NULL and the two draws report and draw nothing, so this is what to branch on before building a meshlet path at all.
	 */
	bool mesh_shaders;

	/**
	 * A pipeline can be created with GpuRasterDesc::supportDualSourceBlending, so its fragment shader's second output can feed the FACTOR_SRC1_* blend factors. Needs Vulkan's dualSrcBlend feature or WebGPU's `dual-source-blending`; where it is false such a pipeline is reported unsupported and not created.
	 */
	bool dual_source_blending;

	/**
	 * gpuSignalAfter can apply SIGNAL_ATOMIC_MAX and SIGNAL_ATOMIC_OR, which need 64 bit buffer atomics. When false those two report themselves unsupported and leave the counter alone; SIGNAL_ATOMIC_SET writes it either way.
	 */
	bool split_barrier_signals;

	/**
	 * gpuWaitBefore is a real split barrier -- work recorded between the two halves is not stalled by the pair. When false the consuming half is a full barrier from every stage, which is correct but gives up what a split barrier is for.
	 *
	 * @note Even when true, only a wait that pairs with an earlier gpuSignalAfter on the same pointer and value, recorded outside a render pass in the same command buffer, is split; anything else falls back. The fallback reports itself.
	 */
	bool split_barriers;

	/**
	 * A texture heap is bound where it was allocated, so writes a shader makes to its descriptors are seen by the draws and dispatches that follow. When false the backend binds a snapshot taken at gpuSetActiveTextureHeapPtr time, so CPU writes still land (they happen before the call) but GPU writes need another call to be picked up.
	 */
	bool gpu_writable_texture_heap;

	/**
	 * Any MEMORY_DEFAULT allocation can be bound as a texture descriptor heap, so MEMORY_DESCRIPTOR_HEAP is just another name for it. When false, heaps have to be allocated as MEMORY_DESCRIPTOR_HEAP to be bound in place.
	 */
	bool descriptor_heap_in_any_memory;

	/**
	 * How many native descriptor slots one GpuTextureDescriptor spans. A shader multiplies a heap index by this, which is what the `noapi` module's GPU_HEAP_STRIDE_RATIO is; it is here for a program that wants to size or inspect a heap itself.
	 */
	uint32_t texture_descriptor_stride_ratio;
} GpuCapabilities;

/**
 * gpuGetCapabilitiesEXT – What the queue's device and backend support.
 *
 * @code
 * GpuCapabilities capabilities = gpuGetCapabilitiesEXT(queue);
 * assert(capabilities.texture_descriptor_stride_ratio >= 1);
 * if(!capabilities.mesh_shaders) {
 * 	// Draw with gpuDrawIndexedInstanced rather than gpuDrawMeshlets
 * }
 * @endcode
 *
 * @param queue Queue to describe.
 */
GpuCapabilities gpuGetCapabilitiesEXT(const GpuQueue* queue);

/**
 * GPU_DIAGNOSTIC – What a diagnostic is telling you.
 *
 * GPU_DIAGNOSTIC_UNSUPPORTED – The call could not be honored at all and did nothing. The corresponding GpuCapabilities field is false.
 *
 * GPU_DIAGNOSTIC_EMULATED – The call was honored, but more conservatively than asked: correct, and slower or coarser. A split barrier that became a full barrier, or a descriptor heap that had to be snapshotted, reports this.
 */
typedef enum GPU_DIAGNOSTIC { GPU_DIAGNOSTIC_UNSUPPORTED, GPU_DIAGNOSTIC_EMULATED } GPU_DIAGNOSTIC;

/**
 * GpuDiagnosticCallbackEXT – Hook the backend reports unsupported and emulated calls through.
 *
 * @param queue The queue the call was recorded against.
 * @param kind Whether the call did nothing or was emulated conservatively.
 * @param message Human readable description, not null terminated.
 * @param userdata The pointer handed to gpuSetDiagnosticCallbackEXT alongside the callback.
 */
typedef void (*GpuDiagnosticCallbackEXT)(GpuQueue* queue, GPU_DIAGNOSTIC kind, GpuStringView message, void* userdata);

/**
 * gpuSetDiagnosticCallbackEXT – Route diagnostics somewhere.
 *
 * When none is set they are printed to stderr. Either way each distinct message is reported once per queue: these fire from calls made per draw, and a report per draw would bury the first one. Setting a callback starts that count over, so the new destination hears each message once too.
 *
 * @code{.cpp}
 * int emulated = 0;
 * gpuSetDiagnosticCallbackEXT(queue, [](GpuQueue*, GPU_DIAGNOSTIC kind, GpuStringView, void* userdata) {
 * 	if(kind == GPU_DIAGNOSTIC_EMULATED) ++*(int*)userdata;
 * }, &emulated);
 *
 * auto counter = gpuMalloc<uint64_t>(queue);
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuWaitBefore(cmd, STAGE_COMPUTE, gpuHostToDevicePointer(queue, counter), 1, OP_EQUAL, (HAZARD_FLAGS)0, 0x1); // No backend can split a masked wait
 * gpuFreeCommandBuffer(cmd);
 * assert(emulated == 1);
 *
 * gpuSetDiagnosticCallbackEXT(queue, nullptr, nullptr);
 * gpuFree(queue, counter);
 * @endcode
 *
 * @param queue Queue to report for.
 * @param callback Where to report, or NULL to go back to printing.
 * @param userdata Passed back to \p callback untouched.
 */
void gpuSetDiagnosticCallbackEXT(GpuQueue* queue, GpuDiagnosticCallbackEXT callback, void* userdata);

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

/**
 * gpuMalloc – Allocate GPU memory and return a CPU-accessible pointer.
 *
 * For MEMORY_DEFAULT, the returned pointer is persistently CPU-mapped (write-combined on discrete GPUs, cache-coherent on UMA). The CPU can write to it directly like ordinary heap memory. The same pointer (after conversion via gpuHostToDevicePointer) can be stored into GPU data structures or passed as a dispatch/draw argument.
 *
 * Default alignment is 16 bytes (vec4 / float4 alignment, sufficient for most data). Use the overload with an explicit alignment parameter when allocating textures or data that requires wider alignment (e.g. 256-byte constant-buffer alignment on some hardware).
 *
 * For MEMORY_GPU and MEMORY_TEXTURE the CPU has no mapping, so the GPU virtual address is returned instead (cast it to gpu*). Use gpuMemCpy / gpuCopyToTexture to transfer data via an intermediate MEMORY_DEFAULT staging region.
 *
 * @code
 * float* values = (float*)gpuMalloc(queue, 4 * sizeof(float), 16, MEMORY_DEFAULT);
 * assert(values);
 * values[0] = 1.0f;
 * gpuFree(queue, values);
 * @endcode
 *
 * @param queue The GPU queue (device) on which the memory will be used.
 * @param bytes Size of the allocation in bytes.
 * @param align Required alignment in bytes (must be a power of two). Default is 16.
 * @param memory Memory heap type (default is MEMORY_DEFAULT).
 * @return The allocation, or NULL if it could not be made, with errno set to the backend's error code.
 */
void* gpuMalloc(GpuQueue* queue, size_t bytes, size_t align NOAPI_DEFAULT(16), MEMORY memory NOAPI_DEFAULT(MEMORY_DEFAULT));

NOAPI_EXTERN_C_END
#ifdef __cplusplus
/**
 * gpuMalloc – Overload that allocates memory sized and aligned for a texture, using the result of a prior gpuTextureSizeAlign call.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * assert(memory);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @param queue The GPU queue (device) on which the memory will be used.
 * @param sizeAlign Size and alignment as returned by gpuTextureSizeAlign.
 * @param memory Memory heap type (default is MEMORY_TEXTURE).
 * @return The allocation's GPU address (for the default MEMORY_TEXTURE), or NULL if it could not be made, with errno set.
 *
 * @note C++ only; from C call gpuMalloc(queue, sizeAlign.size, sizeAlign.align, memory).
 */
inline void* gpuMalloc(GpuQueue* queue, GpuTextureSizeAlign sizeAlign, MEMORY memory = MEMORY_TEXTURE) {
	return gpuMalloc(queue, sizeAlign.size, sizeAlign.align, memory);
}

/**
 * gpuMalloc – Typed overload that allocates storage for `count` objects of type T, using sizeof(T) for the byte size and max(alignof(T), 16) for the alignment.
 *
 * @code{.cpp}
 * struct Particle { float position[3]; float mass; };
 * auto particles = gpuMalloc<Particle>(queue, 64);
 * assert(particles);
 * particles[63].mass = 1.0f;
 * gpuFree(queue, particles);
 * @endcode
 *
 * @tparam T Element type to allocate storage for.
 * @param queue The GPU queue (device) on which the memory will be used.
 * @param count Number of T elements to allocate space for (default 1).
 * @param memory Memory heap type (default is MEMORY_DEFAULT).
 * @return The allocation, or NULL if it could not be made, with errno set.
 *
 * @note C++ only.
 */
template<typename T>
T* gpuMalloc(GpuQueue* queue, size_t count = 1, MEMORY memory = MEMORY_DEFAULT) {
	return (T*)gpuMalloc(queue, sizeof(T) * count, std::max<size_t>(alignof(T), 16), memory);
}
#endif
NOAPI_EXTERN_C_BEGIN

/**
 * gpuFree – Free a GPU memory block previously returned by gpuMalloc, along with the texture created on it, if any. Behavior is undefined if the GPU still has in-flight work that reads or writes the allocation. Use gpuWaitSemaphore to drain GPU work before freeing.
 *
 * @code
 * uint32_t* counts = (uint32_t*)gpuMalloc(queue, 16 * sizeof(uint32_t), 16, MEMORY_DEFAULT);
 * assert(counts);
 * gpuFree(queue, counts);
 * @endcode
 *
 * @param queue The GPU queue (device) the memory was allocated on.
 * @param ptr Pointer returned by gpuMalloc. A pointer gpuMalloc did not return is ignored.
 */
void gpuFree(GpuQueue* queue, void* ptr);

/**
 * gpuFreeDevicePointerEXT – gpuFree for GPU virtual addresses (e.g. as returned by gpuHostToDevicePointer or a MEMORY_GPU allocation).
 *
 * @code
 * gpu* scratch = (gpu*)gpuMalloc(queue, 256, 16, MEMORY_GPU);
 * assert(scratch);
 * gpuFreeDevicePointerEXT(queue, scratch);
 * @endcode
 *
 * @note Spelled gpuFree in C++, where it is an overload of the host pointer flavour.
 *
 * @param queue The GPU queue (device) the memory was allocated on.
 * @param ptr GPU address of the start of an allocation. An address that is not one is ignored.
 */
void gpuFreeDevicePointerEXT(GpuQueue* queue, gpu* ptr);

NOAPI_EXTERN_C_END
#ifdef __cplusplus
/**
 * gpuFree – C++ spelling of gpuFreeDevicePointerEXT.
 *
 * @code{.cpp}
 * auto values = gpuMalloc<float>(queue, 4);
 * gpu* device = gpuHostToDevicePointer(queue, values);
 * gpuFree(queue, device);
 * @endcode
 *
 * @param queue The GPU queue (device) the memory was allocated on.
 * @param ptr GPU address of the start of an allocation. An address that is not one is ignored.
 */
inline void gpuFree(GpuQueue* queue, gpu* ptr) { gpuFreeDevicePointerEXT(queue, ptr); }
#endif
NOAPI_EXTERN_C_BEGIN

/**
 * gpuHostToDevicePointer – Translate a CPU-mapped GPU pointer (from MEMORY_DEFAULT gpuMalloc) into a GPU-side virtual address that can be stored in GPU data structures and dereferenced by shaders.
 *
 * This call performs a hash-map lookup in the driver and is not free. Cache the result alongside the CPU pointer in a small struct (e.g. {void* cpu, void* gpu}) rather than calling this per-frame.
 *
 * Analogous to CUDA's cudaHostGetDevicePointer and Metal 4's Buffer.gpuAddress.
 *
 * @code
 * uint32_t* counts = (uint32_t*)gpuMalloc(queue, 16 * sizeof(uint32_t), 16, MEMORY_DEFAULT);
 * gpu* device = gpuHostToDevicePointer(queue, counts);
 * assert(device);
 * assert(gpuDeviceToHostPointerEXT(queue, device) == counts);
 * gpuFree(queue, counts);
 * @endcode
 *
 * @param queue The GPU queue (device) the memory was allocated on.
 * @param ptr CPU pointer returned by gpuMalloc.
 * @return The GPU address, or NULL if \p ptr is not the start of a CPU-mapped allocation.
 */
gpu* gpuHostToDevicePointer(GpuQueue* queue, void* ptr);

/**
 * gpuDeviceToHostPointerEXT – Translate a GPU pointer to a CPU-mapped pointer.
 *
 * This call performs a hash-map lookup in the driver and is not free. Cache the result alongside the CPU pointer in a small struct (e.g. {void* cpu, void* gpu}) rather than calling this per-frame.
 *
 * @code
 * gpu* scratch = (gpu*)gpuMalloc(queue, 256, 16, MEMORY_GPU);
 * assert(gpuDeviceToHostPointerEXT(queue, scratch) == NULL);
 * gpuFreeDevicePointerEXT(queue, scratch);
 * @endcode
 *
 * @param queue The GPU queue (device) the memory was allocated on.
 * @param ptr GPU address of the start of an allocation.
 * @return The CPU mapping, or NULL if the memory is GPU only (MEMORY_GPU, MEMORY_TEXTURE) or \p ptr is not the start of an allocation.
 */
void* gpuDeviceToHostPointerEXT(GpuQueue* queue, gpu* ptr);

// ---------------------------------------------------------------------------
// Texture management
// ---------------------------------------------------------------------------

/**
 * gpuTextureSizeAlign – Query the required allocation size and alignment for a texture described by `desc`.
 *
 * Must be called before gpuMalloc so that the correct size and alignment can be passed. The returned size includes vendor-specific Morton swizzle overhead and DCC metadata. Allocate the memory as MEMORY_TEXTURE.
 *
 * This separates the memory allocation from the texture object creation, unlike the Vulkan design where a VkImage must be created first to query its memory requirements — a design flaw that forces lazy allocation patterns.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED};
 * GpuTextureSizeAlign size = gpuTextureSizeAlign(queue, desc);
 * assert(size.size > 0);
 * auto memory = (gpu*)gpuMalloc(queue, size.size, size.align, MEMORY_TEXTURE);
 * assert(memory);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @note A USAGE_STORAGE texture has to be in a format gpuFormatIsStorableEXT accepts. WebGPU also has no 1D textures and offers only 1 and 4 samples. For anything else this reports a size of 0.
 *
 * @param queue The GPU queue (device) on which the texture will be created.
 * @param desc Texture description used to determine size.
 * @return The size and alignment, or a size of 0 if the device cannot create such a texture, with errno set.
 */
GpuTextureSizeAlign gpuTextureSizeAlign(GpuQueue* queue, NOAPI_CONST_REF(GpuTextureDesc) desc);

/**
 * gpuCreateTexture – Create a CPU-side GpuTexture handle backed by memory allocated via gpuMalloc with MEMORY_TEXTURE and the size/alignment from gpuTextureSizeAlign.
 *
 * The GpuTexture object is a thin CPU-side handle required because the rasterizer is not yet fully bindless: the CPU driver must write rasterizer command packets (render-target setup, clear, resolve) that refer to vendor-specific internal texture metadata, which is not accessible through the descriptor heap.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * GpuTexture* texture = gpuCreateTexture(queue, desc, memory);
 * assert(texture);
 * gpuFree(queue, memory); // Frees the texture with it
 * @endcode
 *
 * @note Only a single texture can be created on each allocation, and the texture is freed by freeing that memory.
 * @note A description gpuTextureSizeAlign gives a size of 0 for is not created. On WebGPU a texture created with USAGE_SAMPLED must also have a format gpuFormatIsFilterableEXT accepts; any other is reported as unsupported and not created.
 *
 * @param queue The GPU queue (device) on which the texture will be created.
 * @param desc Texture description matching the one passed to gpuTextureSizeAlign.
 * @param memory GPU address of a MEMORY_TEXTURE (or MEMORY_TEXTURE_READBACK) allocation from gpuMalloc.
 * @return The texture, or NULL with errno set if \p memory is not the start of an allocation, already has a texture, or the device refuses the description.
 */
GpuTexture* gpuCreateTexture(GpuQueue* queue, NOAPI_CONST_REF(GpuTextureDesc) desc, gpu* memory);

/**
 * gpuTextureViewDescriptor – Create a read-only (sampled) 512-bit descriptor blob for a sub-range of a GpuTexture, suitable for storing in the global texture heap.
 *
 * The returned GpuTextureDescriptor value can be written directly into a CPU-mapped GpuTextureDescriptor array (the texture heap). The heap is then activated for a command buffer via gpuSetActiveTextureHeapPtr. In shaders, textures are accessed by 32-bit heap indices — no other binding API is required.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .mipCount = 7, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 *
 * auto heap = gpuMalloc<GpuTextureDescriptor>(queue, 2, MEMORY_DESCRIPTOR_HEAP);
 * heap[0] = gpuTextureViewDescriptor(queue, texture, {}); // Every mip
 * heap[1] = gpuTextureViewDescriptor(queue, texture, {.baseMip = 3, .mipCount = 1});
 *
 * gpuFree(queue, heap);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @param queue The GPU queue (device) the texture was created on.
 * @param texture Handle returned by gpuCreateTexture.
 * @param desc Mip/layer sub-range and optional format override.
 * @return The descriptor, or one of all zeros if the backend could not write it, with errno set.
 */
GpuTextureDescriptor gpuTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, NOAPI_CONST_REF(GpuViewDesc) desc);

/**
 * gpuRWTextureViewDescriptor – Create a read/write (storage image / UAV) 512-bit descriptor blob for a GpuTexture, for use in compute shaders that write to textures (gpuStoreTexture in shader code).
 *
 * Only a single mip level can be bound as an RW target at once. Usage flag USAGE_STORAGE must have been specified at texture creation time.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_STORAGE};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 * auto heap = gpuMalloc<GpuTextureDescriptor>(queue, 1, MEMORY_DESCRIPTOR_HEAP);
 * heap[0] = gpuRWTextureViewDescriptor(queue, texture, {}); // Written by shaders with gpuStoreTexture2D(0, ...)
 * gpuFree(queue, heap);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @param queue The GPU queue (device) the texture was created on.
 * @param texture Handle returned by gpuCreateTexture.
 * @param desc Mip/layer sub-range and optional format override.
 * @return The descriptor, or one of all zeros if the backend could not write it, with errno set.
 */
GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, NOAPI_CONST_REF(GpuViewDesc) desc);

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

/**
 * gpuCreateComputePipeline – Compile a compute shader into a hardware-specific GpuPipeline.
 *
 * No root signature, descriptor set layout, or push-constant layout is needed. The shader receives its data as a single 64-bit GPU pointer that the user passes to gpuDispatch / gpuDispatchIndirect.
 *
 * @code{.cpp}
 * auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("compute")] [numthreads(64, 1, 1)]
 * 	void main(uint3 thread : SV_DispatchThreadID) { gpuComputeData<uint>()[thread.x] = 0; }
 * )slang"));
 * assert(pipeline);
 * gpuFreePipeline(queue, pipeline);
 * @endcode
 *
 * @param queue The GPU queue (device) on which the pipeline will be created.
 * @param computeIR Compute shader source (Slang; see shaders.h).
 * @return The pipeline, or NULL if the shader did not compile (reported through gpuSetShaderDiagnosticCallbackEXT's callback) or the device refused it (errno set).
 */
GpuPipeline* gpuCreateComputePipeline(GpuQueue* queue, GpuByteSpan computeIR);

/**
 * gpuFreePipeline – Release a previously compiled GpuPipeline and its associated device memory. Must not be called while the GPU still has in-flight work using the pipeline.
 *
 * @code{.cpp}
 * auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("compute")] [numthreads(1, 1, 1)] void main() {}
 * )slang"));
 * assert(pipeline);
 * gpuFreePipeline(queue, pipeline);
 * @endcode
 *
 * @param queue The GPU queue (device) the pipeline was created on.
 * @param pipeline Pipeline to release.
 */
void gpuFreePipeline(GpuQueue* queue, GpuPipeline* pipeline);

// ---------------------------------------------------------------------------
// Command recording
// ---------------------------------------------------------------------------

/**
 * gpuStartCommandRecording – Begin recording a new transient command buffer on the given queue. Command buffers are one-shot (not reusable across frames). This matches Metal/WebGPU semantics and is what GPU vendors recommend for Vulkan too, since it allows the driver to use a simple bump allocator for internal command memory rather than a heap allocator.
 *
 * The returned GpuCommandBuffer is passed to all subsequent gpuDraw* / gpuDispatch* / gpuBarrier* calls until gpuSubmit consumes it.
 *
 * @code
 * GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
 * assert(cmd);
 * GpuCommandBufferSpan buffers = {&cmd, 1};
 * uint64_t submission = gpuSubmit(queue, buffers, NULL, 0);
 * assert(gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), submission, UINT64_MAX) == submission);
 * @endcode
 *
 * @note Command buffers created from a queue may internally store pointers to it, so it is unsafe to move a queue after sub objects have been created.
 *
 * @param queue Queue the command buffer will later be submitted to.
 * @return The command buffer, or NULL if the backend could not allocate one, with errno set.
 */
GpuCommandBuffer* gpuStartCommandRecording(GpuQueue* queue);

/**
 * gpuFreeCommandBuffer – Destroy a command buffer without submitting it, or one submitted with gpuSubmitNoFreeEXT once the GPU has finished with it.
 *
 * @code
 * GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
 * assert(cmd);
 * gpuFreeCommandBuffer(cmd); // Nothing was worth submitting
 * @endcode
 *
 * @note Prefer gpuSubmit, which destroys the buffers it submits while making sure they aren't destroyed before the submission.
 *
 * @param cmd The command buffer to destroy.
 */
void gpuFreeCommandBuffer(GpuCommandBuffer* cmd);

/**
 * gpuSubmit – Submit a batch of recorded command buffers to the queue for GPU execution, then destroy them. Optionally signals a timeline semaphore to a new value upon completion, enabling frame-pacing and CPU-GPU synchronization without per-submit fence objects.
 *
 * Every submission is also counted on gpuGetSubmissionSemaphoreEXT's semaphore, so the index returned here is the value to wait for there.
 *
 * @code
 * GpuSemaphore* frame = gpuCreateSemaphore(queue, 0);
 * GpuCommandBuffer* cmds[2] = {gpuStartCommandRecording(queue), gpuStartCommandRecording(queue)};
 * GpuCommandBufferSpan buffers = {cmds, 2};
 * uint64_t submission = gpuSubmit(queue, buffers, frame, 1);
 * assert(submission);
 * assert(gpuWaitSemaphore(queue, frame, 1, UINT64_MAX) == 1);
 * assert(gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), GPU_GET_VALUE, UINT64_MAX) >= submission);
 * gpuFreeSemaphore(queue, frame);
 * @endcode
 *
 * @param queue Target submission queue.
 * @param command_buffers Ordered list of command buffers to execute.
 * @param semaphore Optional timeline semaphore to signal on completion.
 * @param signal_value Value to write to the semaphore on completion (monotonically increasing). Ignored if semaphore is null.
 * @return The submission's index, which increases by one per submission, or 0 if the submission failed (errno set). The command buffers are destroyed either way.
 */
uint64_t gpuSubmit(GpuQueue* queue, GpuCommandBufferSpan command_buffers, GpuSemaphore* semaphore NOAPI_DEFAULT(NULL), uint64_t signal_value NOAPI_DEFAULT(0));

/**
 * gpuSubmitNoFreeEXT – Identical to gpuSubmit, except the submitted command buffers are not destroyed. The caller frees each with gpuFreeCommandBuffer once the GPU has finished executing it.
 *
 * @code
 * GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
 * GpuCommandBufferSpan buffers = {&cmd, 1};
 * uint64_t submission = gpuSubmitNoFreeEXT(queue, buffers, NULL, 0);
 * assert(submission);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), submission, UINT64_MAX);
 * gpuFreeCommandBuffer(cmd);
 * @endcode
 *
 * @param queue Target submission queue.
 * @param command_buffers Ordered list of command buffers to execute.
 * @param semaphore Optional timeline semaphore to signal on completion.
 * @param signal_value Value to write to the semaphore on completion (monotonically increasing). Ignored if semaphore is null.
 * @return The submission's index, or 0 if the submission failed (errno set).
 */
uint64_t gpuSubmitNoFreeEXT(GpuQueue* queue, GpuCommandBufferSpan command_buffers, GpuSemaphore* semaphore NOAPI_DEFAULT(NULL), uint64_t signal_value NOAPI_DEFAULT(0));

// ---------------------------------------------------------------------------
// Timeline semaphores (GPU <-> CPU synchronization)
// ---------------------------------------------------------------------------

/**
 * gpuCreateSemaphore – Create a timeline semaphore initialized to `initial_value`.
 *
 * A single semaphore with a monotonically-increasing 64-bit counter replaces the per-frame fence + N-buffering patterns required by older Vulkan/Metal APIs. The pattern "signal N on submit, wait for N-FRAMES_IN_FLIGHT before re-using that frame's resources" is the recommended usage for frame pacing.
 *
 * @code
 * GpuSemaphore* semaphore = gpuCreateSemaphore(queue, 5);
 * assert(semaphore);
 * assert(gpuWaitSemaphore(queue, semaphore, GPU_GET_VALUE, UINT64_MAX) == 5);
 * gpuFreeSemaphore(queue, semaphore);
 * @endcode
 *
 * @param queue The GPU queue (device) on which the semaphore will be used.
 * @param initial_value Starting value of the timeline counter.
 * @return The semaphore, or NULL if it could not be created, with errno set.
 */
GpuSemaphore* gpuCreateSemaphore(GpuQueue* queue, uint64_t initial_value);

/**
 * GPU_GET_VALUE – The value to pass gpuWaitSemaphore to read the counter rather than wait on it.
 */
#define GPU_GET_VALUE UINT64_MAX

/**
 * gpuWaitSemaphore – Block the calling CPU thread until the semaphore's GPU-side counter reaches or exceeds `value`. Used to determine when it is safe to re-use CPU-mapped GPU memory (draw argument buffers, per-frame uniform structs) from a previous frame.
 *
 * @code
 * GpuSemaphore* semaphore = gpuCreateSemaphore(queue, 3);
 * assert(gpuWaitSemaphore(queue, semaphore, 2, UINT64_MAX) == 2); // Already reached
 * assert(gpuWaitSemaphore(queue, semaphore, 4, 0) == 0); // Never signalled, so the wait times out
 * assert(gpuWaitSemaphore(queue, semaphore, 4, 2000000) == 0); // Even given 2 ms
 * assert(gpuWaitSemaphore(queue, semaphore, GPU_GET_VALUE, UINT64_MAX) == 3);
 * gpuFreeSemaphore(queue, semaphore);
 * @endcode
 *
 * @param queue The GPU queue (device) the semaphore was created on.
 * @param semaphore Timeline semaphore to wait on.
 * @param value Counter value to wait for, or GPU_GET_VALUE to return the current value immediately.
 * @param timeout Maximum time to wait, in nanoseconds (default: wait forever).
 * @return \p value once it is reached, the current value for GPU_GET_VALUE, or 0 if the timeout expired or the wait failed (errno set on failure).
 */
uint64_t gpuWaitSemaphore(GpuQueue* queue, const GpuSemaphore* semaphore, uint64_t value, uint64_t timeout NOAPI_DEFAULT(UINT64_MAX));

/**
 * gpuFreeSemaphore – Destroy a timeline semaphore object. No submission still to signal it may be in flight.
 *
 * @code
 * GpuSemaphore* semaphore = gpuCreateSemaphore(queue, 0);
 * assert(semaphore);
 * gpuFreeSemaphore(queue, semaphore);
 * @endcode
 *
 * @param queue The GPU queue (device) the semaphore was created on.
 * @param semaphore Semaphore to destroy.
 */
void gpuFreeSemaphore(GpuQueue* queue, GpuSemaphore* semaphore);

// ---------------------------------------------------------------------------
// GPU commands – data transfer
// ---------------------------------------------------------------------------

/**
 * gpuMemCpy – Record a GPU-side memory copy from src to dest.
 *
 * Both pointers must be GPU virtual addresses (either MEMORY_DEFAULT or MEMORY_GPU, converted via gpuHostToDevicePointer for the former). Use this to transfer data from a CPU-mapped staging region into MEMORY_GPU, allowing the driver to apply lossless generic buffer compression.
 *
 * @code
 * uint32_t* source = (uint32_t*)gpuMalloc(queue, 4 * sizeof(uint32_t), 16, MEMORY_DEFAULT);
 * uint32_t* destination = (uint32_t*)gpuMalloc(queue, 4 * sizeof(uint32_t), 16, MEMORY_READBACK);
 * source[3] = 4;
 * gpu* source_gpu = gpuHostToDevicePointer(queue, source);
 * gpu* destination_gpu = gpuHostToDevicePointer(queue, destination);
 *
 * GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, source_gpu);
 * gpuMemCpy(cmd, destination_gpu, source_gpu, 4 * sizeof(uint32_t));
 * gpuSyncMemoryEXT(cmd, destination_gpu);
 * GpuCommandBufferSpan buffers = {&cmd, 1};
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, buffers, NULL, 0), UINT64_MAX); // destination[3] is now 4
 *
 * gpuFree(queue, source);
 * gpuFree(queue, destination);
 * @endcode
 *
 * @param cmd Command buffer to record into.
 * @param dest Destination GPU address.
 * @param src Source GPU address.
 * @param bytes Number of bytes to copy.
 */
void gpuMemCpy(GpuCommandBuffer* cmd, gpu* dest, gpu* src, size_t bytes);

/**
 * gpuCopyToTexture – Record a copy from a linear CPU-mapped staging region into a texture's memory, applying the vendor-specific Morton swizzle and DCC compression.
 *
 * This is the canonical path for uploading texture data. After this copy, the texture is immediately usable by samplers without any additional layout transitions (unlike old Vulkan image layout transitions or GCN-era DCC decompress passes).
 *
 * The staging data is the first mip level's texels in rows, tightly packed. The other mip levels are filled by rendering, or by gpuBlitTextureEXT from the level above.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_TRANSFER_DST)};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 * auto texels = gpuMalloc<uint32_t>(queue, 64 * 64);
 * for(uint32_t i = 0; i < 64 * 64; ++i) texels[i] = 0xff0000ff; // Opaque red, as RGBA bytes
 * auto texels_gpu = gpuHostToDevicePointer(queue, texels);
 *
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, texels_gpu);
 * gpuCopyToTexture(cmd, memory, texels_gpu, texture);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 *
 * gpuFree(queue, texels);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @note Every array layer is copied, or the whole volume of a 3D texture, layer after layer in the staging data.
 *
 * @param cmd Command buffer to record into.
 * @param dest The texture's memory (from gpuMalloc with MEMORY_TEXTURE).
 * @param src GPU address of the linear texel data, usually MEMORY_DEFAULT.
 * @param texture GpuTexture handle describing the layout/format for swizzling.
 */
void gpuCopyToTexture(GpuCommandBuffer* cmd, gpu* dest, gpu* src, GpuTexture* texture);

/**
 * gpuCopyFromTexture – Record a copy from a texture back to a linear MEMORY_READBACK region (e.g. for screenshots or virtual-texturing feedback). The driver decompresses DCC and un-swizzles the Morton layout into linear rows.
 *
 * Copies the first mip level, rows tightly packed, the inverse of gpuCopyToTexture.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 * auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
 * auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
 *
 * GpuColorAttachment color = {.texture = texture, .loadOp = LOAD_OP_CLEAR, .clearValue = {0, 1, 0, 1}};
 * GpuRenderPassDesc pass = {.colorAttachments = {&color, 1}};
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuBeginRenderPass(cmd, pass);
 * gpuEndRenderPass(cmd);
 * gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
 * gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
 * gpuSyncMemoryEXT(cmd, pixels_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1})); // pixels now holds opaque green
 *
 * gpuFree(queue, pixels);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @note Depth reads back as the values the depth test compares, and a stencil as its bytes. A texture with both depth and stencil copies only its depth. WebGPU can't copy out of FORMAT_D24_PLUS or FORMAT_D24_PLUS_S8_UINT at all, since how many bits those hold is up to the implementation, and copies any other depth or stencil texture only whole, so there its rows have to be a multiple of 256 bytes; anything else is reported unsupported and not copied.
 *
 * @param cmd Command buffer to record into.
 * @param dest GPU address of a MEMORY_READBACK region (CPU can read after the submission has finished).
 * @param src The texture's memory (from gpuMalloc with MEMORY_TEXTURE).
 * @param texture GpuTexture handle.
 */
void gpuCopyFromTexture(GpuCommandBuffer* cmd, gpu* dest, gpu* src, const GpuTexture* texture);

/**
 * gpuBlitTextureEXT – Copy one texture subresource onto another by sampling it in a small internal fullscreen pass.
 *
 * Unlike the copies above this goes through the rasterizer rather than the copy engine, so the two subresources may differ in size (the source is rescaled by the sampler) and in format (the color is converted on the way out). The whole of the destination mip is written, and its previous contents are discarded. The pipeline and sampler it needs are internal to the backend and built the first time they are asked for; none of the state the command buffer had bound is disturbed.
 *
 * The source must have been created with USAGE_SAMPLED and the destination with USAGE_COLOR_ATTACHMENT. Only color formats are supported — depth/stencil textures need the copy engine or a shader written against the depth sampling intrinsics. 3D textures can't be addressed a slice at a time; blit out of a 2D array instead.
 *
 * Because it rasterizes, this is the one command here that a compute-only queue cannot record, and it must not be recorded inside a render pass — it opens one of its own.
 *
 * Does NOT insert an automatic barrier on either side. If the source was just written by a compute pass, or the destination is about to be sampled, call gpuBarrier as usual.
 *
 * @code{.cpp}
 * GpuTextureDesc source_desc = {.dimensions = {128, 128, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED};
 * GpuTextureDesc destination_desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA16_FLOAT, .usage = USAGE_COLOR_ATTACHMENT};
 * auto source_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, source_desc));
 * auto destination_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, destination_desc));
 * auto source = gpuCreateTexture(queue, source_desc, source_memory);
 * auto destination = gpuCreateTexture(queue, destination_desc, destination_memory);
 *
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuBlitTextureEXT(cmd, destination, source); // Half the size, and converted to half floats
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 *
 * gpuFree(queue, destination_memory);
 * gpuFree(queue, source_memory);
 * @endcode
 *
 * @note A 32 bit float source can't be filtered, so it is blitted with nearest sampling whatever was asked; WebGPU doesn't sample those formats at all.
 *
 * @param cmd Command buffer to record into.
 * @param destination Texture rendered into.
 * @param source Texture sampled from.
 * @param linear_filter When true the source is sampled bilinearly, otherwise the nearest texel is taken. Formats a sampler can't filter (the 32-bit float ones) always fall back to nearest.
 * @param destination_mip Mip level rendered into.
 * @param destination_slice Array layer / cube face rendered into.
 * @param source_mip Mip level sampled from.
 * @param source_slice Array layer / cube face sampled from.
 */
void gpuBlitTextureEXT(GpuCommandBuffer* cmd, GpuTexture* destination, const GpuTexture* source,
	bool linear_filter NOAPI_DEFAULT(true),
	uint32_t destination_mip NOAPI_DEFAULT(0), uint32_t destination_slice NOAPI_DEFAULT(0),
	uint32_t source_mip NOAPI_DEFAULT(0), uint32_t source_slice NOAPI_DEFAULT(0));

// ---------------------------------------------------------------------------
// GPU commands – texture heap
// ---------------------------------------------------------------------------

/**
 * gpuSetActiveTextureHeapPtr – Set the GPU pointer to the global texture descriptor heap for all subsequent draw and dispatch commands in this command buffer.
 *
 * The heap is a flat array of GpuTextureDescriptor values (each 512 bits / 64 bytes) allocated via gpuMalloc. Shaders access textures by writing a 32-bit index into their root data struct; the sampler fetches heap[index] internally.
 *
 * Changing the heap pointer may cause an internal pipeline flush on older hardware. On modern GPUs the base address is embedded per sample instruction and multiple heaps can be sampled seamlessly.
 *
 * Allocate the heap as MEMORY_DESCRIPTOR_HEAP. Where the device allows it that lets the heap be bound where it lies, which is both cheaper (nothing is copied) and what makes a descriptor written by a shader visible; see GpuTextureDescriptor. Any other memory still works, at the cost of a snapshot per call, and the backend reports that through the diagnostic callback the first time it has to take one. Because the snapshot is retaken every time the heap is set, a CPU write between two command buffers is still seen.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto heap = gpuMalloc<GpuTextureDescriptor>(queue, 1, MEMORY_DESCRIPTOR_HEAP);
 * heap[0] = gpuTextureViewDescriptor(queue, gpuCreateTexture(queue, desc, memory), {});
 * auto heap_gpu = gpuHostToDevicePointer(queue, heap);
 *
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, heap_gpu);
 * gpuSetActiveTextureHeapPtr(cmd, heap_gpu); // Shaders now sample the texture as gpuSample2D(0, ...)
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 *
 * gpuFree(queue, heap);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @param cmd Command buffer to record into.
 * @param texture_heap GPU virtual address of the first GpuTextureDescriptor in the heap.
 */
void gpuSetActiveTextureHeapPtr(GpuCommandBuffer* cmd, gpu* texture_heap);

// ---------------------------------------------------------------------------
// GPU commands – barriers and split barriers
// ---------------------------------------------------------------------------

/**
 * gpuBarrier – Insert a pipeline barrier expressing a producer-to-consumer execution dependency.
 *
 * Unlike DX12 / Vulkan 1.x, no resource list is required. The API reflects what modern coherent-L2 GPU hardware actually does: flush the tiny non-coherent caches (compute unit L0$, ROP cache, HiZ) implied by the `before` and `after` stages, and optionally invalidate special caches selected by `hazards`.
 *
 * @code
 * GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
 * gpuBarrier(cmd, STAGE_COMPUTE, STAGE_COMPUTE, (HAZARD_FLAGS)0); // One dispatch reads what the last one wrote
 * gpuBarrier(cmd, STAGE_COMPUTE, STAGE_PIXEL_SHADER, HAZARD_DESCRIPTORS); // A dispatch rewrote texture heap entries
 * gpuBarrier(cmd, (STAGE)(STAGE_RASTER_COLOR_OUT | STAGE_RASTER_DEPTH_OUT), STAGE_PIXEL_SHADER, (HAZARD_FLAGS)0); // Render targets are sampled next
 * gpuBarrier(cmd, STAGE_COMPUTE, STAGE_RASTER_DEPTH_OUT, HAZARD_DEPTH_STENCIL); // A dispatch wrote what is about to be a depth buffer
 * gpuFreeCommandBuffer(cmd);
 * @endcode
 *
 * @note A no-op on WebGPU, which orders passes and makes their writes visible on its own.
 *
 * @param cmd Command buffer to record into.
 * @param before Bitmask of STAGE values for the producing stage(s).
 * @param after Bitmask of STAGE values for the consuming stage(s).
 * @param hazards Optional HAZARD_FLAGS bitmask for special cache invalidation.
 */
void gpuBarrier(GpuCommandBuffer* cmd, STAGE before, STAGE after, HAZARD_FLAGS hazards NOAPI_DEFAULT((HAZARD_FLAGS)0));

/**
 * gpuSignalAfter – Split-barrier producer: after `before` finishes, atomically update the 64-bit counter at `ptr` using `signal` operation.
 *
 * Combine with gpuWaitBefore to implement split barriers that allow independent work to be inserted between the producer and consumer, avoiding GPU stalls. Timeline semaphore semantics (monotonic counter) are the recommended pattern, as in the example.
 *
 * The counter is real memory and the write really happens, so a shader or the CPU can read it to observe the producer's progress. SIGNAL_ATOMIC_MAX and SIGNAL_ATOMIC_OR need 64 bit buffer atomics, so they only apply where gpuGetCapabilitiesEXT().split_barrier_signals is true; elsewhere they report themselves through the diagnostic callback and leave the counter alone.
 *
 * Whether the paired wait is a true split barrier is a separate question, answered by gpuGetCapabilitiesEXT().split_barriers; see gpuWaitBefore.
 *
 * @code{.cpp}
 * auto counter = gpuMalloc<uint64_t>(queue, 1, MEMORY_READBACK);
 * auto counter_gpu = gpuHostToDevicePointer(queue, counter);
 *
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSignalAfter(cmd, STAGE_COMPUTE, counter_gpu, 7, SIGNAL_ATOMIC_SET);
 * // ... work that does not depend on the dispatches before the signal ...
 * gpuWaitBefore(cmd, STAGE_PIXEL_SHADER, counter_gpu, 7, OP_GREATER_EQUAL);
 * gpuSyncMemoryEXT(cmd, counter_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1})); // *counter is now 7
 *
 * gpuFree(queue, counter);
 * @endcode
 *
 * What can't be honored falls back and says so: a signal inside a render pass has nothing to write the counter with, a wait there can't be split, and a signal no wait pairs with ordered nothing.
 *
 * @param cmd Command buffer to record into.
 * @param before Producer stage; signal fires after this stage completes.
 * @param ptr GPU virtual address of a 64-bit counter value in GPU memory.
 * @param value Value to write / OR / max into the counter.
 * @param signal Atomic operation to apply. Only SIGNAL_ATOMIC_SET is available everywhere; the other two need split_barrier_signals, which WebGPU never has.
 */
void gpuSignalAfter(GpuCommandBuffer* cmd, STAGE before, gpu* ptr, uint64_t value, SIGNAL signal);

/**
 * gpuWaitBefore – Split-barrier consumer: stall the GPU before `after` starts until the 64-bit counter at `ptr` satisfies the comparison `op` against `value`. Optionally performs cache invalidation via `hazards` and limits which bits of the counter are compared via `mask` (e.g. for multi-producer bitmask patterns using SIGNAL_ATOMIC_OR).
 *
 * How much of that a backend can honor varies, and the honest summary is that the counter comparison is an interface no current native API implements directly:
 *
 * - A wait that pairs with an earlier gpuSignalAfter in the same command buffer, on the same \p ptr and \p value, recorded outside a render pass, becomes a real split barrier: work recorded between the two halves is not stalled by the pair. This is the pattern the example uses, and the one to write.
 * - Anything else -- a \p mask, an \p op other than OP_GREATER_EQUAL or OP_EQUAL, a counter written by a shader rather than by gpuSignalAfter, or a signal that is in another command buffer -- cannot be expressed. The wait then becomes a full barrier from every stage, which is a superset of the dependency asked for (so correct on one queue) but gives up the independent work in between. Each such fallback reports itself once through the diagnostic callback.
 *
 * - Inside a render pass Vulkan can record neither the wait nor a barrier, so there the call reports itself unsupported and does nothing; record waits before gpuBeginRenderPass. WebGPU's pass boundaries already order everything a wait could ask for.
 *
 * gpuGetCapabilitiesEXT().split_barriers is false on a backend where even the paired form cannot be split, in which case every wait is the full barrier above.
 *
 * @code{.cpp}
 * auto counter = gpuMalloc<uint64_t>(queue);
 * auto counter_gpu = gpuHostToDevicePointer(queue, counter);
 *
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSignalAfter(cmd, STAGE_COMPUTE, counter_gpu, 1, SIGNAL_ATOMIC_SET);
 * gpuWaitBefore(cmd, STAGE_COMPUTE, counter_gpu, 1, OP_GREATER_EQUAL, HAZARD_DRAW_ARGUMENTS);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 *
 * gpuFree(queue, counter);
 * @endcode
 *
 * @param cmd Command buffer to record into.
 * @param after Consumer stage; this stage stalls until the condition is met.
 * @param ptr GPU virtual address of the 64-bit counter to poll.
 * @param value Reference value for the comparison.
 * @param op Comparison operator (e.g. OP_GREATER_EQUAL for timeline semantics).
 * @param hazards Optional HAZARD_FLAGS for cache invalidation after the wait.
 * @param mask Bitmask applied to the counter before comparison (default: all bits).
 */
void gpuWaitBefore(GpuCommandBuffer* cmd, STAGE after, gpu* ptr, uint64_t value, OP op, HAZARD_FLAGS hazards NOAPI_DEFAULT((HAZARD_FLAGS)0), uint64_t mask NOAPI_DEFAULT(~(uint64_t)0));

// ---------------------------------------------------------------------------
// GPU commands – pipeline binding
// ---------------------------------------------------------------------------

/**
 * gpuSetPipeline – Bind a compiled GpuPipeline for subsequent dispatch or draw commands. This is equivalent to DX12's SetPipelineState / Vulkan's vkCmdBindPipeline, but is cheap because no descriptor set or root signature change is involved — data is passed via GPU pointers in the draw/dispatch call.
 *
 * @code{.cpp}
 * auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("compute")] [numthreads(1, 1, 1)] void main() {}
 * )slang"));
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSetPipeline(cmd, pipeline);
 * gpuFreeCommandBuffer(cmd);
 * gpuFreePipeline(queue, pipeline);
 * @endcode
 *
 * @note On WebGPU a graphics pipeline is built lazily for the depth/stencil and blend state in effect, so binding one after either changes can compile a new variant.
 *
 * @param cmd Command buffer to record into.
 * @param pipeline Pipeline to bind for subsequent dispatch/draw commands.
 */
void gpuSetPipeline(GpuCommandBuffer* cmd, const GpuPipeline* pipeline);

// ---------------------------------------------------------------------------
// GPU commands – compute dispatch
// ---------------------------------------------------------------------------

/**
 * gpuDispatch – Launch a compute shader. The shader receives `data` as its single root pointer (cast to the matching shader-side struct). Thread groups are spawned in a 3D grid of `grid_dimensions` groups.
 *
 * The bound pipeline stays bound across copies recorded between dispatches.
 *
 * @code{.cpp}
 * auto values = gpuMalloc<uint32_t>(queue, 64, MEMORY_READBACK);
 * auto values_gpu = gpuHostToDevicePointer(queue, values);
 * auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("compute")] [numthreads(64, 1, 1)]
 * 	void main(uint3 thread : SV_DispatchThreadID) { gpuComputeData<uint>()[thread.x] = thread.x * 2; }
 * )slang"));
 *
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSetPipeline(cmd, pipeline);
 * gpuDispatch(cmd, values_gpu, {1, 1, 1});
 * gpuSyncMemoryEXT(cmd, values_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1})); // values[5] is now 10
 *
 * gpuFreePipeline(queue, pipeline);
 * gpuFree(queue, values);
 * @endcode
 *
 * @param cmd Command buffer to record into.
 * @param data GPU pointer to the root data struct (see root arguments design).
 * @param grid_dimensions Thread group grid (x * y * z total groups).
 */
void gpuDispatch(GpuCommandBuffer* cmd, gpu* data, uvec3 grid_dimensions);

/**
 * gpuDispatchIndirect – Like gpuDispatch but reads the thread group dimensions from GPU memory, enabling GPU-driven compute (e.g. dispatch counts produced by culling shaders). Requires a preceding barrier with HAZARD_DRAW_ARGUMENTS if the argument buffer was written by a compute shader in the same command buffer.
 *
 * @code{.cpp}
 * auto values = gpuMalloc<uint32_t>(queue, 64, MEMORY_READBACK);
 * auto values_gpu = gpuHostToDevicePointer(queue, values);
 * auto grid = gpuMalloc<uvec3>(queue);
 * *grid = {4, 1, 1};
 * auto grid_gpu = gpuHostToDevicePointer(queue, grid);
 * auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("compute")] [numthreads(16, 1, 1)]
 * 	void main(uint3 thread : SV_DispatchThreadID) { gpuComputeData<uint>()[thread.x] = 1; }
 * )slang"));
 *
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, grid_gpu);
 * gpuSetPipeline(cmd, pipeline);
 * gpuDispatchIndirect(cmd, values_gpu, grid_gpu);
 * gpuSyncMemoryEXT(cmd, values_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1})); // All 64 values are now 1
 *
 * gpuFreePipeline(queue, pipeline);
 * gpuFree(queue, grid);
 * gpuFree(queue, values);
 * @endcode
 *
 * @param cmd Command buffer to record into.
 * @param data GPU pointer to the root data struct.
 * @param grid_dimensions_gpu GPU pointer to a uvec3 holding the group dimensions.
 */
void gpuDispatchIndirect(GpuCommandBuffer* cmd, gpu* data, gpu* grid_dimensions_gpu);

NOAPI_EXTERN_C_END
