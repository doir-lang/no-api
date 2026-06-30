
#pragma once

/**
 * A minimalistic modern GPU API based on the design described in:
 * "No Graphics API" by Sebastian Aaltonen
 * https://www.sebastianaaltonen.com/blog/no-graphics-api
 *
 * Design philosophy:
 * - All shader data inputs are passed as a single 64-bit GPU pointer (analogous
 * to how OS threading APIs pass a single void* to thread functions).
 * - Textures are globally accessible via a flat descriptor heap indexed by 32-bit
 * integers; no per-draw descriptor set management is required.
 * - Memory is managed via a CUDA-style malloc/free API that returns CPU-mapped
 * GPU pointers (UMA / PCIe ReBAR), avoiding the Vulkan object-first allocation
 * chicken-and-egg problem.
 * - Barriers describe only producer/consumer execution stages and a small set of
 * hazard flags. Individual resource state tracking (as in DX12 / Vulkan 1.x) is
 * not needed on modern coherent-L2 GPUs.
 * - The entire public surface fits in ~150 lines.
 */

#include <cstddef> // size_t
#include <cstdint> // uint8_t, uint16_t, uint32_t, uint64_t
#include <span> // std::std::span (C++20)
#include <string_view> // std::string_view (C++17)
#include <algorithm> // std::max

// ---------------------------------------------------------------------------
// Basic helpers used by the API
// ---------------------------------------------------------------------------
struct uvec3 { uint32_t x, y, z; };

/**
 * string_to_bytes – Reinterpret a string_view's characters as a read-only byte
 * span, useful for passing shader IR source text to functions expecting bytes.
 *
 * @param str String to view as bytes.
 */
inline std::span<const std::byte> string_to_bytes(std::string_view str) {
	return {reinterpret_cast<const std::byte*>(str.data()), str.size()};
}

/**
 * byte_span – Reinterpret a span of T elements as a read-only byte span.
 *
 * @tparam T Element type of the input span.
 * @param span Span to view as bytes.
 */
template<typename T>
std::span<const std::byte> byte_span(std::span<const T> span) {
	return {reinterpret_cast<const std::byte*>(span.data()), span.size_bytes()};
} 

// ---------------------------------------------------------------------------
// Opaque GPU object handles
// ---------------------------------------------------------------------------

/**
 * Semantically equivalent to void. Some functions will expect a gpu* instead of
 * a void* to indicate that the data should be on the gpu.
 */
struct gpu;

/**
 * GpuQueue
 * Represents a GPU submission queue (graphics, compute, or copy). Work is
 * recorded into GpuCommandBuffer objects and submitted via gpuSubmit.
 */
struct GpuQueue;

/**
 * GpuTexture
 * CPU-side handle for a GPU texture allocation. Needed because the triangle
 * rasterizer is not yet fully bindless on modern GPUs: the CPU driver must
 * prepare render-target and depth/stencil command packets. Created by
 * gpuCreateTexture. The 256-bit sampler descriptor written into the descriptor
 * heap is a separate GpuTextureDescriptor value (see below).
 */
struct GpuTexture;

/**
 * GpuPipeline
 * Represents a compiled shader pipeline (compute, vertex+pixel, or mesh+pixel).
 * Created by gpuCreateComputePipeline / gpuCreateGraphicsPipeline /
 * gpuCreateGraphicsMeshletPipeline. Contains hardware-specific shader microcode.
 * Freed with gpuFreePipeline.
 */
struct GpuPipeline;

/**
 * GpuCommandBuffer
 * A transient (one-shot) recording of GPU commands. Created by
 * gpuStartCommandRecording and consumed by gpuSubmit. Reuse across frames is
 * intentionally not supported: one-shot buffers simplify driver memory management
 * (bump allocator vs. heap allocator) and avoid accidentally replaying stale work.
 */
struct GpuCommandBuffer;

/**
 * GpuSemaphore
 * A monotonically-increasing 64-bit timeline counter for GPU<->CPU and
 * GPU<->GPU synchronisation. Semantically identical to DX12 / Vulkan 1.2
 * timeline semaphores. A single semaphore and an ever-incrementing counter value
 * replace the older Vulkan/Metal per-submit fence objects and N-buffering patterns.
 * Created by gpuCreateSemaphore(initialValue); destroyed by gpuDestroySemaphore.
 */
struct GpuSemaphore;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/**
 * MEMORY – GPU memory heap type for gpuMalloc.
 *
 * MEMORY_DEFAULT – CPU-mapped GPU memory (write-combined). The CPU can write
 * directly to the returned pointer, and the GPU reads it fast
 * via PCIe ReBAR or UMA. Optimal for small, frequently-updated
 * data: draw arguments, per-frame uniforms, GPU pointers, and
 * texture/buffer descriptors.
 *
 * MEMORY_GPU – GPU-private memory. The CPU cannot write to these pointers
 * directly. Data must first be placed in MEMORY_DEFAULT
 * (an "upload" region) and then copied with gpuMemCpy /
 * gpuCopyToTexture. The driver can apply vendor-specific
 * lossless compression (DCC, delta colour compression) and
 * Morton/tile swizzling for textures, maximising bandwidth.
 *
 * MEMORY_READBACK – CPU-cached memory suitable for GPU->CPU readback. Slower for
 * the GPU to write due to cache-coherency with the CPU.
 * Mainly used for screenshots and virtual-texturing feedback.
 *
 * MEMORY_TEXTURE – Semantically identical to MEMORY_GPU but with internal
 * optimizations that make the WebGPU emulation layer perform
 * better when used for texture storage.
 *
 * MEMORY_TEXTURE_READBACK – Semantically identical to MEMORY_READBACK but with the
 * same internal optimizations as MEMORY_TEXTURE.
 */
enum MEMORY { MEMORY_DEFAULT, MEMORY_GPU, MEMORY_READBACK, MEMORY_TEXTURE, MEMORY_TEXTURE_READBACK };

/**
 * OP – Comparison operator used by depth tests, stencil tests, and semaphore
 * wait conditions (gpuWaitBefore).
 */
enum OP {
	OP_NEVER, ///< Test always fails.
	OP_LESS, ///< Passes if incoming < reference.
	OP_EQUAL, ///< Passes if incoming == reference.
	OP_LESS_EQUAL, ///< Passes if incoming <= reference. Default for depth test.
	OP_GREATER, ///< Passes if incoming > reference.
	OP_NOT_EQUAL, ///< Passes if incoming != reference.
	OP_GREATER_EQUAL, ///< Passes if incoming >= reference.
	OP_ALWAYS, ///< Test always passes.
};

/**
 * TEXTURE – Texture dimensionality / view type.
 */
enum TEXTURE {
	TEXTURE_1D,
	TEXTURE_2D,
	TEXTURE_3D,
	TEXTURE_CUBE, // TODO: Can we emulate cubemaps since everything is getting merged into a big array?
	TEXTURE_2D_ARRAY,
	TEXTURE_CUBE_ARRAY,
};

/**
 * FORMAT – Pixel / texel formats. Expand as needed for your target feature set.
 * FORMAT_NONE is used to indicate "no attachment" (e.g. no depth buffer).
 */
enum FORMAT {
	FORMAT_NONE,
	FORMAT_RGBA8_UNORM,
	FORMAT_RGBA8_SRGB,
	FORMAT_RGBA16_FLOAT,
	FORMAT_RGBA32_FLOAT,
	FORMAT_RG11B10_FLOAT,
	FORMAT_RGB10_A2_UNORM,
	FORMAT_R8_UNORM,
	FORMAT_R16_FLOAT,
	FORMAT_R32_FLOAT,
	FORMAT_D16_UNORM,
	FORMAT_D24_UNORM_S8_UINT,
	FORMAT_D32_FLOAT,
	FORMAT_D32_FLOAT_S8_UINT,
	// ... extend with BC/ETC/ASTC compressed formats as required
};

/**
 * gpuFormatIsDepth – Returns true if `format` is one of the depth (or
 * depth/stencil) FORMAT values.
 *
 * @param format Format to test.
 */
inline bool gpuFormatIsDepth(FORMAT format) {
	switch (format) {
	case FORMAT_D16_UNORM:
	case FORMAT_D24_UNORM_S8_UINT:
	case FORMAT_D32_FLOAT:
	case FORMAT_D32_FLOAT_S8_UINT:
		return true;
	default: return false;
	}
}

/**
 * gpuFormatIsStencil – Returns true if `format` includes a stencil component.
 *
 * @param format Format to test.
 */
inline bool gpuFormatIsStencil(FORMAT format) {
	switch (format) {
	case FORMAT_D24_UNORM_S8_UINT:
	case FORMAT_D32_FLOAT_S8_UINT:
		return true;
	default: return false;
	}
}

/**
 * TEXTURE_USAGE_FLAGS – Bitmask describing how a GpuTexture allocation will be used.
 * The driver uses these flags to choose the appropriate memory layout and
 * enable hardware features (DCC compression, HiZ, MSAA resolve, etc.).
 * Combine with bitwise-OR.
 */
enum TEXTURE_USAGE_FLAGS {
	USAGE_SAMPLED = 0x01, ///< Readable by texture samplers.
	USAGE_STORAGE = 0x02, ///< Read/write access from compute shaders.
	USAGE_COLOR_ATTACHMENT = 0x04, ///< Rasterizer colour render target.
	USAGE_DEPTH_STENCIL_ATTACHMENT = 0x08, ///< Rasterizer depth/stencil target.
	USAGE_TRANSFER_SRC = 0x10, ///< Source of copy operations.
	USAGE_TRANSFER_DST = 0x20, ///< Destination of copy operations.
};

/**
 * STAGE – GPU pipeline stage used in barrier producer/consumer descriptions and
 * gpuSignalAfter / gpuWaitBefore split-barrier commands.
 *
 * Modern GPUs only require stage-level dependency tracking. Individual resource
 * state transitions (as in Vulkan image layouts / DX12 resource states) are
 * unnecessary on coherent-L2 architectures and are not part of this API.
 *
 * Combine multiple stages with bitwise-OR to express "any of these stages".
 */
enum STAGE {
	STAGE_TRANSFER = 0x001, ///< DMA / copy engine (gpuMemCpy, gpuCopyToTexture).
	STAGE_COMPUTE = 0x002, ///< Compute shader execution.
	STAGE_VERTEX_SHADER = 0x004, ///< Vertex or mesh shader execution.
	STAGE_PIXEL_SHADER = 0x008, ///< Pixel / fragment shader execution.
	STAGE_RASTER_COLOR_OUT = 0x010, ///< Rasterizer colour output (ROPs / framebuffer writes).
	STAGE_RASTER_DEPTH_OUT = 0x020, ///< Rasterizer depth/stencil output.
	STAGE_ALL = 0x03F, ///< All stages (conservative; use sparingly).
	// NOTE: In WebGPU will we be able to get any more granular than all?
};

/**
 * HAZARD_FLAGS – Optional bitmask passed to gpuBarrier / gpuWaitBefore to request
 * flushing or invalidation of special non-coherent GPU caches that are not
 * automatically flushed by every barrier.
 *
 * Most barriers between compute and raster stages do not need any flags. These
 * flags exist because GPU hardware hides a small number of non-coherent caches
 * (descriptor caches in texture samplers, command-processor prefetch buffers,
 * HiZ / depth-metadata caches) that require explicit invalidation in specific
 * scenarios.
 */
enum HAZARD_FLAGS {
	/**
	 * HAZARD_DRAW_ARGUMENTS – Stall the GPU command-processor's prefetch until the
	 * producing stage has finished writing draw/dispatch argument buffers.
	 * Required when a compute dispatch writes indirect draw arguments that are
	 * consumed by a subsequent gpuDispatchIndirect or gpuDrawIndexedInstancedIndirect.
	 */
	HAZARD_DRAW_ARGUMENTS = 0x1,

	/**
	 * HAZARD_DESCRIPTORS – Invalidate texture-sampler descriptor caches.
	 * Required when a compute or CPU write updates entries in the global texture
	 * descriptor heap between uses. Normal reads/writes to GPU memory do not
	 * touch the sampler's internal descriptor cache; this flag is the only way
	 * to force a re-fetch of the updated descriptors.
	 */
	HAZARD_DESCRIPTORS = 0x2,

	/**
	 * HAZARD_DEPTH_STENCIL – Invalidate depth metadata caches (HiZ, stencil
	 * cache). Required after a compute shader writes to memory that will
	 * subsequently be consumed by the depth/stencil unit as a depth buffer.
	 */
	HAZARD_DEPTH_STENCIL = 0x4,
};

/**
 * SIGNAL – Atomic operation applied when gpuSignalAfter writes to the counter
 * pointer in GPU memory.
 *
 * SIGNAL_ATOMIC_SET – Unconditionally write the value (non-atomic store).
 * SIGNAL_ATOMIC_MAX – Atomically update if the new value is greater (implements
 * timeline semaphore semantics when combined with
 * gpuWaitBefore + OP_GREATER_EQUAL).
 * SIGNAL_ATOMIC_OR – Atomically OR the value into the counter (enables
 * multi-producer bitmask patterns: wait until all
 * producers have set their bit).
 */
enum SIGNAL { SIGNAL_ATOMIC_SET, SIGNAL_ATOMIC_MAX, SIGNAL_ATOMIC_OR };

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/**
 * GpuTextureDesc – Description of a GPU texture allocation.
 * Passed to gpuTextureSizeAlign (to query the required memory footprint) and then
 * to gpuCreateTexture (to obtain a GpuTexture CPU handle for rasterization binding).
 */
struct GpuTextureDesc {
	TEXTURE type = TEXTURE_2D; ///< Dimensionality and view type.
	uvec3 dimensions = {1, 1, 1}; ///< Dimensions in texels.
	uint32_t mipCount = 1; ///< Number of mip levels (1 = no mips). // TODO: Should we view this as upload data, and then generate all missing mips (since we are storing textures as big arrays)
	uint32_t layerCount = 1; ///< Array layer count (for TEXTURE_2D_ARRAY / TEXTURE_CUBE_ARRAY).
	uint32_t sampleCount = 1; ///< MSAA sample count.
	FORMAT format = FORMAT_NONE; ///< Texel format.
	TEXTURE_USAGE_FLAGS usage = (TEXTURE_USAGE_FLAGS)0; ///< Bitmask of TEXTURE_USAGE_FLAGS values describing intended use.
};

/**
 * GpuViewDesc – Describes a sub-range of a GpuTexture for use in a descriptor.
 * Passed to gpuTextureViewDescriptor / gpuRWTextureViewDescriptor to create a
 * 256-bit descriptor blob that is stored in the global texture descriptor heap.
 *
 * ALL_MIPS / ALL_LAYERS sentinel values indicate "from base to the last level/layer".
 */
static constexpr uint8_t ALL_MIPS = 0xff;
static constexpr uint16_t ALL_LAYERS = 0xffff;

struct GpuViewDesc {
	FORMAT format = FORMAT_NONE; ///< Override format, or FORMAT_NONE to use the texture's own format.
	uint8_t baseMip = 0; ///< Index of the first mip level included in the view.
	uint8_t mipCount = ALL_MIPS; ///< Number of mip levels included (ALL_MIPS = all remaining).
	uint16_t baseLayer = 0; ///< First array layer / cube face included in the view.
	uint16_t layerCount= ALL_LAYERS; ///< Number of layers included (ALL_LAYERS = all remaining).
};

/**
 * GpuTextureSizeAlign – Returned by gpuTextureSizeAlign.
 * Provides the byte size and required alignment for the GPU memory block that must
 * be passed to gpuMalloc before calling gpuCreateTexture. The driver accounts for
 * vendor-specific Morton swizzling, DCC metadata, and alignment padding.
 */
struct GpuTextureSizeAlign {
	size_t size; ///< Total allocation size in bytes (including metadata).
	size_t align; ///< Required alignment in bytes for the gpuMalloc call.
};

/**
 * GpuTextureDescriptor – A 256-bit opaque hardware-specific texture descriptor blob.
 *
 * This is the "raw descriptor" that GPUs load into scalar registers (AMD) or index
 * into the sampler descriptor heap (Nvidia, Apple, Qualcomm). The user writes these
 * blobs directly into a gpuMalloc'd array of GpuTextureDescriptor objects — the
 * global texture heap. The GPU and CPU can both read and write this array without
 * any additional API objects, unlike DX12's descriptor heap copy APIs.
 *
 * Created by gpuTextureViewDescriptor (sampled, read-only) or
 * gpuRWTextureViewDescriptor (storage / read-write).
 */
struct GpuTextureDescriptor { uint64_t data[4]; };

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/**
 * gpuCreateQueue – Create a GPU submission queue (device and queue selection
 * details are platform-specific and omitted from this minimal API).
 */
// GpuQueue* gpuCreateQueue(/* platform-specific device/queue parameters */);

/**
 * @brief Destroy all of the objects internally allocated by a queue.
 * All subobjects of the queue should be destroyed before calling this function
 *
 * @note Won't destroy any of the platform specific objects the queue is created from.
 *
 * @param queue Queue to destroy.
 */
void gpuDestroyQueue(GpuQueue* queue);

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

/**
 * gpuMalloc – Allocate GPU memory and return a CPU-accessible pointer.
 *
 * For MEMORY_DEFAULT, the returned pointer is persistently CPU-mapped (write-combined
 * on discrete GPUs, cache-coherent on UMA). The CPU can write to it directly like
 * ordinary heap memory. The same pointer (after conversion via gpuHostToDevicePointer)
 * can be stored into GPU data structures or passed as a dispatch/draw argument.
 *
 * Default alignment is 16 bytes (vec4 / float4 alignment, sufficient for most data).
 * Use the overload with an explicit alignment parameter when allocating textures or
 * data that requires wider alignment (e.g. 256-byte constant-buffer alignment on
 * some hardware).
 *
 * For MEMORY_GPU allocations the pointer cannot be written by the CPU (a GPU side
 * pointer is returned). Use gpuMemCpy / gpuCopyToTexture to transfer data via an
 * intermediate MEMORY_DEFAULT staging region.
 *
 * @param queue The GPU queue (device) on which the memory will be used.
 * @param bytes Size of the allocation in bytes.
 * @param align Required alignment in bytes (must be a power of two). Default is 16.
 * @param memory Memory heap type (default is MEMORY_DEFAULT).
 */
void* gpuMalloc(GpuQueue* queue, size_t bytes, size_t align = 16, MEMORY memory = MEMORY_DEFAULT);

/**
 * gpuMalloc – Overload that allocates memory sized/aligned for a texture, using
 * the result of a prior gpuTextureSizeAlign call.
 *
 * @param queue The GPU queue (device) on which the memory will be used.
 * @param sizeAlign Size and alignment as returned by gpuTextureSizeAlign.
 * @param memory Memory heap type (default is MEMORY_TEXTURE).
 */
inline void* gpuMalloc(GpuQueue* queue, GpuTextureSizeAlign sizeAlign, MEMORY memory = MEMORY_TEXTURE) {
	return gpuMalloc(queue, sizeAlign.size, sizeAlign.align, memory);
}

/**
 * gpuMalloc – Typed overload that allocates storage for `count` objects of type
 * T, using sizeof(T) for the byte size and max(alignof(T), 16) for the alignment.
 *
 * @tparam T Element type to allocate storage for.
 * @param queue The GPU queue (device) on which the memory will be used.
 * @param count Number of T elements to allocate space for (default 1).
 * @param memory Memory heap type (default is MEMORY_DEFAULT).
 */
template<typename T>
T* gpuMalloc(GpuQueue* queue, size_t count = 1, MEMORY memory = MEMORY_DEFAULT) {
	return (T*)gpuMalloc(queue, sizeof(T) * count, std::max<size_t>(alignof(T), 16), memory);
}

/**
 * gpuFree – Free a GPU memory block previously returned by gpuMalloc.
 * Behaviour is undefined if the GPU still has in-flight work that reads or writes
 * the allocation. Use gpuWaitSemaphore to drain GPU work before freeing.
 *
 * @param queue The GPU queue (device) the memory was allocated on.
 * @param ptr CPU pointer returned by gpuMalloc. Must not be null.
 */
void gpuFree(GpuQueue* queue, void* ptr);

/**
 * gpuFree – Overload of gpuFree for GPU virtual addresses (e.g. as returned by
 * gpuHostToDevicePointer or a MEMORY_GPU allocation).
 *
 * @param queue The GPU queue (device) the memory was allocated on.
 * @param ptr GPU pointer returned by gpuMalloc. Must not be null.
 */
void gpuFree(GpuQueue* queue, gpu* ptr);

/**
 * gpuHostToDevicePointer – Translate a CPU-mapped GPU pointer (from MEMORY_DEFAULT
 * gpuMalloc) into a GPU-side virtual address that can be stored in GPU data
 * structures and dereferenced by shaders.
 *
 * This call performs a hash-map lookup in the driver and is not
 * free. Cache the result alongside the CPU pointer in a small struct
 * (e.g. {void* cpu, void* gpu}) rather than calling this per-frame.
 *
 * Analogous to CUDA's cudaHostGetDevicePointer and Metal 4's Buffer.gpuAddress.
 *
 * @param queue The GPU queue (device) the memory was allocated on.
 * @param ptr CPU pointer returned by gpuMalloc. Must not be null.
 */
gpu* gpuHostToDevicePointer(GpuQueue* queue, void* ptr);

/**
 * gpuDeviceToHostPointer – Translate a GPU pointer to a CPU-mapped pointer.
 * Will return nullptr if the backing memory is GPU only.
 *
 * This call performs a hash-map lookup in the driver and is not
 * free. Cache the result alongside the CPU pointer in a small struct
 * (e.g. {void* cpu, void* gpu}) rather than calling this per-frame.
 *
 * @param queue The GPU queue (device) the memory was allocated on.
 * @param ptr GPU pointer to translate. Must not be null.
 */
void* gpuDeviceToHostPointer(GpuQueue* queue, gpu* ptr);

// ---------------------------------------------------------------------------
// Texture management
// ---------------------------------------------------------------------------

/**
 * gpuTextureSizeAlign – Query the required allocation size and alignment for a
 * texture described by `desc`.
 *
 * Must be called before gpuMalloc so that the correct size and alignment can be
 * passed. The returned size includes vendor-specific Morton swizzle overhead and
 * DCC metadata. The allocation must be in MEMORY_GPU for optimal compressed layout.
 *
 * This separates the memory allocation from the texture object creation, unlike
 * the Vulkan design where a VkImage must be created first to query its memory
 * requirements — a design flaw that forces lazy allocation patterns.
 *
 * @param queue The GPU queue (device) on which the texture will be created.
 * @param desc Texture description used to determine size.
 */
GpuTextureSizeAlign gpuTextureSizeAlign(GpuQueue* queue, const GpuTextureDesc& desc);

/**
 * gpuCreateTexture – Create a CPU-side GpuTexture handle backed by an existing GPU
 * memory pointer (allocated via gpuMalloc with MEMORY_GPU and the size/alignment
 * from gpuTextureSizeAlign).
 *
 * The GpuTexture object is a thin CPU-side handle required because the rasterizer
 * is not yet fully bindless: the CPU driver must write rasterizer command packets
 * (render-target setup, clear, resolve) that refer to vendor-specific internal
 * texture metadata, which is not accessible through the 256-bit descriptor heap.
 *
 * @note It is assumed that only a single image is bound to each memory allocation.
 *
 * @param queue The GPU queue (device) on which the texture will be created.
 * @param desc Texture description matching the one passed to gpuTextureSizeAlign.
 * @param memory MEMORY_TEXTURE/MEMORY_TEXTURE_READBACK pointer from gpuMalloc (GPU virtual address).
 */
GpuTexture* gpuCreateTexture(GpuQueue* queue, const GpuTextureDesc& desc, gpu* memory);

/**
 * gpuTextureViewDescriptor – Create a read-only (sampled) 256-bit descriptor blob
 * for a sub-range of a GpuTexture, suitable for storing in the global texture heap.
 *
 * The returned GpuTextureDescriptor value can be written directly into a CPU-mapped
 * GpuTextureDescriptor array (the texture heap). The heap is then activated for a
 * command buffer via gpuSetActiveTextureHeapPtr. In shaders, textures are accessed
 * by 32-bit heap indices — no other binding API is required.
 *
 * @param queue The GPU queue (device) the texture was created on.
 * @param texture Handle returned by gpuCreateTexture.
 * @param desc Mip/layer sub-range and optional format override.
 */
GpuTextureDescriptor gpuTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc);

/**
 * gpuRWTextureViewDescriptor – Create a read/write (storage image / UAV) 256-bit
 * descriptor blob for a GpuTexture, for use in compute shaders that write to
 * textures (TextureRW in shader code).
 *
 * Only a single mip level can be bound as an RW target at once. Usage flag
 * USAGE_STORAGE must have been specified at texture creation time.
 *
 * @param queue The GPU queue (device) the texture was created on.
 * @param texture Handle returned by gpuCreateTexture.
 * @param desc Mip/layer sub-range and optional format override.
 */
GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc);

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

/**
 * gpuCreateComputePipeline – Compile a compute shader from its intermediate
 * representation (IR) into a hardware-specific GpuPipeline.
 *
 * No root signature, descriptor set layout, or push-constant layout is needed.
 * The shader receives its data as a single 64-bit GPU pointer that the user passes
 * to gpuDispatch / gpuDispatchIndirect.
 *
 * @param queue The GPU queue (device) on which the pipeline will be created.
 * @param computeIR Platform IR blob (e.g. DXIL, SPIR-V, or Metal AIR).
 */
GpuPipeline* gpuCreateComputePipeline(GpuQueue* queue, std::span<const std::byte> computeIR);

/**
 * gpuDestroyPipeline – Release a previously compiled GpuPipeline and its associated
 * device memory. Must not be called while the GPU still has in-flight work using
 * the pipeline.
 *
 * @param queue The GPU queue (device) the pipeline was created on.
 * @param pipeline Pipeline to release.
 */
void gpuDestroyPipeline(GpuQueue* queue, GpuPipeline* pipeline);

// ---------------------------------------------------------------------------
// Command recording
// ---------------------------------------------------------------------------

/**
 * gpuStartCommandRecording – Begin recording a new transient command buffer on
 * the given queue. Command buffers are one-shot (not reusable across frames). This
 * matches Metal/WebGPU semantics and is what GPU vendors recommend for Vulkan too,
 * since it allows the driver to use a simple bump allocator for internal command
 * memory rather than a heap allocator.
 *
 * The returned GpuCommandBuffer is passed to all subsequent gpuCmd* / gpuDraw* /
 * gpuDispatch* / gpuBarrier* calls until gpuSubmit consumes it.
 *
 * @note CommandBuffers created from a queue may internally store pointers to it.
 * Thus it is unsafe to move a Queue after sub objects have been created.
 *
 * @param queue Queue the command buffer will later be submitted to.
 */
GpuCommandBuffer* gpuStartCommandRecording(GpuQueue* queue);

/**
 * @brief Destroys the command buffers.
 * This function is intended to be used on buffers that it turns out won't be submitted.
 *
 * @note You should prefer using gpuSubmit which will automatically destroy the submitted
 * buffers while ensuring they aren't destroyed before submission.
 * 
 * @param cmd The command buffer to destroy
 */
void gpuFreeCommandBuffer(GpuCommandBuffer* cmd);

/**
 * gpuSubmit – Submit a batch of recorded command buffers to the queue for GPU
 * execution. Optionally signals a timeline semaphore to a new value upon completion,
 * enabling frame-pacing and CPU-GPU synchronisation without per-submit fence objects.
 * Returns a monotonically increasing submission number.
 *
 * Note that submitted command buffers are then destroyed unless the NoDestroy variant is used
 *
 * @param queue Target submission queue.
 * @param command_buffers Ordered list of command buffers to execute.
 * @param semaphore Optional timeline semaphore to signal on completion.
 * @param signal_value Value to write to the semaphore on completion (monotonically
 * increasing). Ignored if semaphore is null.
 */
uint64_t gpuSubmit(GpuQueue* queue, std::span<GpuCommandBuffer*> command_buffers, GpuSemaphore* semaphore = nullptr, uint64_t signal_value = 0);

/**
 * gpuSubmitNoFree – Identical to gpuSubmit, except the submitted command
 * buffers are NOT destroyed after submission. The caller is responsible for
 * eventually calling gpuDestoryCommandBuffer on each buffer once the GPU has
 * finished executing it.
 *
 * @param queue Target submission queue.
 * @param commandBuffers Ordered list of command buffers to execute.
 * @param semaphore Optional timeline semaphore to signal on completion.
 * @param signal_value Value to write to the semaphore on completion (monotonically
 * increasing). Ignored if semaphore is null.
 */
uint64_t gpuSubmitNoFree(GpuQueue* queue, std::span<GpuCommandBuffer*> commandBuffers, GpuSemaphore* semaphore = nullptr, uint64_t signal_value = 0);

// ---------------------------------------------------------------------------
// Timeline semaphores (GPU <-> CPU synchronisation)
// ---------------------------------------------------------------------------

/**
 * gpuCreateSemaphore – Create a timeline semaphore initialised to `initial_value`.
 *
 * A single semaphore with a monotonically-increasing 64-bit counter replaces
 * the per-frame fence + N-buffering patterns required by older Vulkan/Metal APIs.
 * The pattern "signal N on submit, wait for N-FRAMES_IN_FLIGHT before re-using
 * that frame's resources" is the recommended usage for frame pacing.
 *
 * @param queue The GPU queue (device) on which the semaphore will be used.
 * @param initial_value Starting value of the timeline counter.
 */
GpuSemaphore* gpuCreateSemaphore(GpuQueue* queue, uint64_t initial_value);

#define GPU_GET_VALUE UINT64_MAX

/**
 * gpuWaitSemaphore – Block the calling CPU thread until the semaphore's GPU-side
 * counter reaches or exceeds `value`. Used to determine when it is safe to
 * re-use CPU-mapped GPU memory (draw argument buffers, per-frame uniform structs)
 * from a previous frame.
 *
 * @note if \p value is set to GPU_GET_VALUE instead of waiting the current value of the semaphore will be returned
 *
 * @param queue The GPU queue (device) the semaphore was created on.
 * @param semaphore Timeline semaphore to wait on.
 * @param value Counter value to wait for, or GPU_GET_VALUE to return the current value immediately.
 * @param timeout Maximum time to wait, in implementation-defined units (default: wait forever).
 */
uint64_t gpuWaitSemaphore(GpuQueue* queue, const GpuSemaphore* semaphore, uint64_t value, uint64_t timeout = UINT64_MAX);

/**
 * gpuFreeSemaphore – Destroy a timeline semaphore object.
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
 * Both pointers must be GPU virtual addresses (either MEMORY_DEFAULT or
 * MEMORY_GPU, converted via gpuHostToDevicePointer for the former). Use this to
 * transfer data from a CPU-mapped staging region into MEMORY_GPU, allowing the
 * driver to apply lossless generic buffer compression.
 *
 * @param cmd Command buffer to record into.
 * @param dest Destination GPU address.
 * @param src Source GPU address.
 * @param bytes Number of bytes to copy.
 * @param no_offsets When true it skips calculating offsets into buffers for the gpu*'s
 */
void gpuMemCpy(GpuCommandBuffer* cmd, gpu* dest, gpu* src, size_t bytes, bool no_offsets = false);

/**
 * gpuCopyToTexture – Record a copy from a linear CPU-mapped staging region into a
 * MEMORY_GPU texture allocation, applying the vendor-specific Morton swizzle and
 * DCC compression.
 *
 * This is the canonical path for uploading texture data. After this copy, the
 * texture is immediately usable by samplers without any additional layout
 * transitions (unlike old Vulkan image layout transitions or GCN-era DCC
 * decompress passes).
 *
 * @param cmd Command buffer to record into.
 * @param dest GPU-only texture memory pointer (from gpuMalloc MEMORY_GPU).
 * @param src CPU-mapped (MEMORY_DEFAULT) GPU pointer to linear texture data.
 * @param texture GpuTexture handle describing the layout/format for swizzling.
 * @param no_offsets When true it skips calculating offsets into buffers for the gpu*'s
 */
void gpuCopyToTexture(GpuCommandBuffer* cmd, gpu* dest, gpu* src, GpuTexture* texture, bool no_offsets = false);

/**
 * gpuCopyFromTexture – Record a copy from a MEMORY_GPU texture back to a linear
 * MEMORY_READBACK region (e.g. for screenshots or virtual-texturing feedback).
 * The driver decompresses DCC and un-swizzles the Morton layout into linear rows.
 *
 * @param cmd Command buffer to record into.
 * @param dest MEMORY_READBACK GPU pointer (CPU can read after semaphore wait).
 * @param src GPU-only texture memory pointer.
 * @param texture GpuTexture handle.
 * @param no_offsets When true it skips calculating offsets into buffers for the gpu*'s
 */
void gpuCopyFromTexture(GpuCommandBuffer* cmd, gpu* dest, gpu* src, const GpuTexture* texture, bool no_offsets = false);

// ---------------------------------------------------------------------------
// GPU commands – texture heap
// ---------------------------------------------------------------------------

/**
 * gpuSetActiveTextureHeapPtr – Set the GPU pointer to the global texture descriptor
 * heap for all subsequent draw and dispatch commands in this command buffer.
 *
 * The heap is a flat array of GpuTextureDescriptor values (each 256 bits / 32 bytes)
 * allocated via gpuMalloc. Shaders access textures by writing a 32-bit index into
 * their root data struct; the sampler fetches heap[index] internally.
 *
 * Changing the heap pointer may cause an internal pipeline flush on older hardware.
 * On modern GPUs the base address is embedded per sample instruction and multiple
 * heaps can be sampled seamlessly.
 *
 * @param cmd Command buffer to record into.
 * @param texture_heap GPU virtual address of the first GpuTextureDescriptor in the heap.
 * @param no_offsets When true it skips calculating offsets into buffers for the gpu*'s
 */
void gpuSetActiveTextureHeapPtr(GpuCommandBuffer* cmd, gpu* texture_heap, bool no_offsets = false);

// ---------------------------------------------------------------------------
// GPU commands – barriers and split barriers
// ---------------------------------------------------------------------------

/**
 * gpuBarrier – Insert a pipeline barrier expressing a producer-to-consumer
 * execution dependency.
 *
 * Unlike DX12 / Vulkan 1.x, no resource list is required. The API reflects what
 * modern coherent-L2 GPU hardware actually does: flush the tiny non-coherent caches
 * (compute unit L0$, ROP cache, HiZ) implied by the `before` and `after` stages,
 * and optionally invalidate special caches selected by `hazards`.
 *
 * Common usage patterns:
 *
 * // Compute → compute UAV barrier (most common; no hazard flags needed)
 * gpuBarrier(cb, STAGE_COMPUTE, STAGE_COMPUTE);
 *
 * // Compute writes texture descriptor heap → subsequent sampling
 * gpuBarrier(cb, STAGE_COMPUTE, STAGE_PIXEL_SHADER, HAZARD_DESCRIPTORS);
 *
 * // Render target → texture sample (ROP cache is flushed automatically)
 * gpuBarrier(cb, STAGE_RASTER_COLOR_OUT | STAGE_RASTER_DEPTH_OUT, STAGE_PIXEL_SHADER);
 *
 * @param cmd Command buffer to record into.
 * @param before Bitmask of STAGE values for the producing stage(s).
 * @param after Bitmask of STAGE values for the consuming stage(s).
 * @param hazards Optional HAZARD_FLAGS bitmask for special cache invalidation.
 */
void gpuBarrier(GpuCommandBuffer* cmd, STAGE before, STAGE after, HAZARD_FLAGS hazards = (HAZARD_FLAGS)0);

/**
 * gpuSignalAfter – Split-barrier producer: after `before` finishes, atomically
 * update the 64-bit counter at `ptr` using `signal` operation.
 *
 * Combine with gpuWaitBefore to implement split barriers that allow independent
 * work to be inserted between the producer and consumer, avoiding GPU stalls.
 * Timeline semaphore semantics (monotonic counter) are the recommended pattern:
 *
 * gpuSignalAfter(cb, STAGE_COMPUTE, counterGpu, N, SIGNAL_ATOMIC_MAX);
 * // ... independent work that doesn't depend on the compute output ...
 * gpuWaitBefore(cb, STAGE_PIXEL_SHADER, counterGpu, N, OP_GREATER_EQUAL);
 *
 * @param cmd Command buffer to record into.
 * @param before Producer stage; signal fires after this stage completes.
 * @param ptr GPU virtual address of a 64-bit counter value in GPU memory.
 * @param value Value to write / OR / max into the counter.
 * @param signal Atomic operation to apply.
 */
void gpuSignalAfter(GpuCommandBuffer* cmd, STAGE before, gpu* ptr, uint64_t value, SIGNAL signal);

/**
 * gpuWaitBefore – Split-barrier consumer: stall the GPU before `after` starts
 * until the 64-bit counter at `ptr` satisfies the comparison `op` against
 * `value`. Optionally performs cache invalidation via `hazards` and limits which
 * bits of the counter are compared via `mask` (e.g. for multi-producer bitmask
 * patterns using SIGNAL_ATOMIC_OR).
 *
 * @param cmd Command buffer to record into.
 * @param after Consumer stage; this stage stalls until the condition is met.
 * @param ptr GPU virtual address of the 64-bit counter to poll.
 * @param value Reference value for the comparison.
 * @param op Comparison operator (e.g. OP_GREATER_EQUAL for timeline semantics).
 * @param hazards Optional HAZARD_FLAGS for cache invalidation after the wait.
 * @param mask Bitmask applied to the counter before comparison (default: all bits).
 */
void gpuWaitBefore(GpuCommandBuffer* cmd, STAGE after, gpu* ptr, uint64_t value, OP op, HAZARD_FLAGS hazards = (HAZARD_FLAGS)0, uint64_t mask = ~uint64_t(0));

// ---------------------------------------------------------------------------
// GPU commands – pipeline binding
// ---------------------------------------------------------------------------

/**
 * gpuSetPipeline – Bind a compiled GpuPipeline for subsequent dispatch or draw
 * commands. This is equivalent to DX12's SetPipelineState / Vulkan's
 * vkCmdBindPipeline, but is cheap because no descriptor set or root signature
 * change is involved — data is passed via GPU pointers in the draw/dispatch call.
 *
 * NOTE: On WebGPU this call is not cheap!
 *
 * @param cmd Command buffer to record into.
 * @param pipeline Pipeline to bind for subsequent dispatch/draw commands.
 */
void gpuSetPipeline(GpuCommandBuffer* cmd, const GpuPipeline* pipeline);

// ---------------------------------------------------------------------------
// GPU commands – compute dispatch
// ---------------------------------------------------------------------------

/**
 * gpuDispatch – Launch a compute shader. The shader receives `data` as its
 * single root pointer (cast to the matching shader-side struct). Thread groups are
 * spawned in a 3D grid of `grid_dimensions` groups.
 *
 * @param cmd Command buffer to record into.
 * @param data GPU pointer to the root data struct (see root arguments design).
 * @param grid_dimensions Thread group grid (x * y * z total groups).
 */
void gpuDispatch(GpuCommandBuffer* cmd, gpu* data, uvec3 grid_dimensions);

/**
 * gpuDispatchIndirect – Like gpuDispatch but reads the thread group dimensions from
 * GPU memory, enabling GPU-driven compute (e.g. dispatch counts produced by
 * culling shaders). Requires a preceding barrier with HAZARD_DRAW_ARGUMENTS if the
 * argument buffer was written by a compute shader in the same command buffer.
 *
 * @param cmd Command buffer to record into.
 * @param data GPU pointer to the root data struct.
 * @param grid_dimensions_gpu GPU pointer to a uvec3 holding the group dimensions.
 * @param no_offsets When true it skips calculating offsets into buffers for the gpu*'s
 */
void gpuDispatchIndirect(GpuCommandBuffer* cmd, gpu* data, gpu* grid_dimensions_gpu, bool no_offsets = false);
