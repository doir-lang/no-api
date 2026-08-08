#pragma once

/**
 * A minimalistic modern GPU API based on the design described in:
 * "No Graphics API" by Sebastian Aaltonen
 * https://www.sebastianaaltonen.com/blog/no-graphics-api
 *
 * This file contains the graphics specific public API while compute.hpp contains
 * all of the shared code used for both graphics and compute.
 *
 * NOTE: The original API did not include any API for swapchain/surface support.
 * Thus I have taken the liberty of lifting a WebGPU inspired API for surfaces.
 */

#include "compute.hpp"

#include <optional>

// ---------------------------------------------------------------------------
// Basic math types used in the API
// ---------------------------------------------------------------------------
struct uvec2 { uint32_t x, y; };
struct ivec2 { int32_t x, y; };

// ---------------------------------------------------------------------------
// Opaque GPU object handles
// ---------------------------------------------------------------------------

/**
 * GpuDepthStencilState
 * A pre-baked depth/stencil configuration object. Separating this state from the
 * PSO (as Metal does) reduces pipeline permutations. Applied per-command-buffer
 * via gpuSetDepthStencilState. Created by gpuCreateDepthStencilState.
 */
struct GpuDepthStencilState;

/**
 * GpuBlendState
 * A pre-baked alpha-blend configuration object. On desktop GPUs that expose
 * fixed-function blend hardware this can be applied dynamically without
 * recompiling the PSO, significantly cutting blend-mode permutations.
 * On mobile TBDR GPUs blending is burned into the pixel shader microcode; users
 * may instead use framebuffer-fetch intrinsics and author a parametrised formula.
 * Created by gpuCreateBlendState; requires a device feature flag.
 */
struct GpuBlendState;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/**
 * CULL – Triangle-facing culling mode for rasterisation.
 * CULL_CCW – Cull counter-clockwise (back-face) triangles (typical default).
 * CULL_CW – Cull clockwise triangles.
 * CULL_ALL – Cull all triangles (useful for depth-only / shadow-map prepass
 * that only needs depth from the rasterizer's early-Z).
 * CULL_NONE – No culling; both faces are rasterized.
 */
enum CULL { CULL_CCW, CULL_CW, CULL_ALL, CULL_NONE };

/**
 * DEPTH_FLAGS – Bitmask controlling depth-buffer access in GpuDepthStencilDesc.
 * DEPTH_READ – The depth buffer is tested (early-Z / depth test enabled).
 * DEPTH_WRITE – Passing fragments update the depth buffer.
 * Combine with bitwise-OR, e.g. DEPTH_READ | DEPTH_WRITE.
 */
enum DEPTH_FLAGS { DEPTH_READ = 0x1, DEPTH_WRITE = 0x2 };

/**
 * STENCIL_OP – Action taken on the stencil buffer depending on depth/stencil
 * test outcomes.
 */
enum STENCIL_OP {
	STENCIL_OP_KEEP, ///< Keep the current stencil value unchanged.
	STENCIL_OP_ZERO, ///< Set stencil to 0.
	STENCIL_OP_REPLACE, ///< Replace with the reference value.
	STENCIL_OP_INCR_SAT, ///< Increment, clamping at the maximum value.
	STENCIL_OP_DECR_SAT, ///< Decrement, clamping at 0.
	STENCIL_OP_INVERT, ///< Bitwise-invert the stencil value.
	STENCIL_OP_INCR_WRAP, ///< Increment, wrapping to 0 on overflow.
	STENCIL_OP_DECR_WRAP, ///< Decrement, wrapping on underflow.
};

/**
 * BLEND – RGB / alpha blend equation applied by the fixed-function blender
 * (or emulated via framebuffer-fetch on mobile TBDR GPUs).
 * result = src * srcFactor <BLEND> dst * dstFactor
 */
enum BLEND {
	BLEND_ADD, ///< result = src*srcF + dst*dstF (standard alpha blend)
	BLEND_SUBTRACT, ///< result = src*srcF - dst*dstF
	BLEND_REV_SUBTRACT, ///< result = dst*dstF - src*srcF
	BLEND_MIN, ///< result = min(src, dst) (factors ignored)
	BLEND_MAX, ///< result = max(src, dst) (factors ignored)
};

/**
 * FACTOR – Blend scale factors applied to source and destination colours/alphas.
 * Maps 1:1 to DX12/Vulkan blend factor enumerations.
 */
enum FACTOR {
	FACTOR_ZERO, ///< 0
	FACTOR_ONE, ///< 1
	FACTOR_SRC_COLOR, ///< Source RGB
	FACTOR_ONE_MINUS_SRC_COLOR, ///< 1 - source RGB
	FACTOR_DST_COLOR, ///< Destination RGB
	FACTOR_ONE_MINUS_DST_COLOR, ///< 1 - destination RGB
	FACTOR_SRC_ALPHA, ///< Source alpha
	FACTOR_ONE_MINUS_SRC_ALPHA, ///< 1 - source alpha
	FACTOR_DST_ALPHA, ///< Destination alpha
	FACTOR_ONE_MINUS_DST_ALPHA, ///< 1 - destination alpha
	FACTOR_SRC1_COLOR, ///< Second source RGB (dual-source blending)
	FACTOR_ONE_MINUS_SRC1_COLOR, ///< 1 - second source RGB
	FACTOR_SRC1_ALPHA, ///< Second source alpha (dual-source blending)
	FACTOR_ONE_MINUS_SRC1_ALPHA, ///< 1 - second source alpha
};

/**
 * TOPOLOGY – Primitive assembly topology for the rasterizer.
 * TOPOLOGY_TRIANGLE_LIST – Every 3 vertices form an independent triangle.
 * TOPOLOGY_TRIANGLE_STRIP – Vertices share edges; (n-2) triangles from n vertices.
 * TOPOLOGY_TRIANGLE_FAN – Vertices fan around the first vertex.
 * NOTE: Supporting points and lines will increase the API surface... but we would only gain an optional line width parameter
 */
enum TOPOLOGY {
	TOPOLOGY_TRIANGLE_LIST,
	TOPOLOGY_TRIANGLE_STRIP,
	// TOPOLOGY_TRIANGLE_FAN, // Not supported by WebGPU
};

enum INDEX_TYPE_EXT {
	INDEX_TYPE_UINT8,
	INDEX_TYPE_UINT16,
	INDEX_TYPE_UINT32
};

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/**
 * Stencil – Per-face stencil operation description used inside GpuDepthStencilDesc.
 */
struct GpuStencil {
	OP test = OP_ALWAYS; ///< Stencil comparison function.
	STENCIL_OP failOp = STENCIL_OP_KEEP; ///< Action when stencil test fails.
	STENCIL_OP passOp = STENCIL_OP_KEEP; ///< Action when both tests pass.
	STENCIL_OP depthFailOp = STENCIL_OP_KEEP; ///< Action when stencil passes but depth fails.
	uint8_t reference = 0; ///< Stencil reference value.
};

/**
 * GpuDepthStencilDesc – Full description of depth/stencil test and write behavior.
 *
 * This is intentionally separated from the PSO (unlike DX12, where most of these
 * fields are baked into the pipeline object). Separating depth/stencil state
 * reduces the number of PSO permutations and allows changing it cheaply at
 * draw-call time via gpuSetDepthStencilState, matching the actual GPU command
 * packet granularity.
 *
 * All fields have sensible defaults: by default depth testing is disabled and
 * stencil operations are all KEEP / ALWAYS.
 */
struct GpuDepthStencilDesc {
	DEPTH_FLAGS depthMode = (DEPTH_FLAGS)0; ///< Bitmask: DEPTH_READ, DEPTH_WRITE, or both.
	OP depthTest = OP_ALWAYS; ///< Depth comparison function (e.g. OP_LESS_EQUAL).
	float depthBias = 0.0f; ///< Constant depth value added to each fragment.
	float depthBiasSlopeFactor = 0.0f; ///< Depth bias scaled by triangle slope (shadow mapping).
	float depthBiasClamp = 0.0f; ///< Maximum magnitude of the slope-scaled depth bias.
	uint8_t stencilReadMask = 0xff; ///< Mask ANDed with the stencil buffer before comparison.
	uint8_t stencilWriteMask = 0xff; ///< Mask ANDed with written stencil values.
	GpuStencil stencilFront; ///< Stencil ops for front-facing (CCW) triangles.
	GpuStencil stencilBack; ///< Stencil ops for back-facing (CW) triangles.
};

/**
 * GpuBlendDesc – Alpha blending configuration for a single color render target.
 *
 * Default values implement opaque (no-blending) output: src * 1 + dst * 0 = src.
 * A standard "over" alpha-blend would be:
 * colorOp = BLEND_ADD, srcColorFactor = FACTOR_SRC_ALPHA,
 * dstColorFactor = FACTOR_ONE_MINUS_SRC_ALPHA, ... etc.
 *
 * On desktop GPUs with fixed-function blend hardware this state can be applied
 * dynamically via gpuSetBlendState. On mobile TBDR GPUs it is typically compiled
 * into the pixel shader; consider framebuffer-fetch + a parametrized blend formula
 * instead to avoid PSO permutations.
 */
struct GpuBlendDesc {
	BLEND colorOp = BLEND_ADD; ///< Blend equation for RGB channels.
	FACTOR srcColorFactor = FACTOR_ONE; ///< Scale factor applied to the source (incoming) color.
	FACTOR dstColorFactor = FACTOR_ZERO; ///< Scale factor applied to the destination (existing) color.
	BLEND alphaOp = BLEND_ADD; ///< Blend equation for the alpha channel.
	FACTOR srcAlphaFactor = FACTOR_ONE; ///< Scale factor applied to the source alpha.
	FACTOR dstAlphaFactor = FACTOR_ZERO; ///< Scale factor applied to the destination alpha.
	uint8_t colorWriteMask = 0xf; ///< Per-channel write enable: bit 0=R, 1=G, 2=B, 3=A.
	// TODO: Should we just mandate this be done in the pixel shader?
};

/**
 * ColorTarget – Render target format and write mask entry in GpuRasterDesc.
 *
 * writeMask mirrors the DX12 / Vulkan concept of per-render-target write masks.
 * Baking it into the PSO lets the shader compiler dead-code-eliminate color
 * outputs that are masked off, saving ALU and export bandwidth. This is distinct
 * from GpuBlendDesc::colorWriteMask, which applies when a dynamic GpuBlendState is
 * in use.
 */
struct GpuColorTarget {
	FORMAT format = FORMAT_NONE; ///< Pixel format of this render target attachment.
	uint8_t writeMask = 0xf; ///< Bitmask: bit 0=R, 1=G, 2=B, 3=A. 0xf = write all channels.
};

/**
 * GpuRasterDesc – Minimal rasterizer state baked into a graphics PSO.
 *
 * The goal is to keep this struct small to minimize PSO permutations. States that
 * change frequently (depth/stencil behavior, blend modes on capable hardware) are
 * moved out into separate dynamically-applied state objects.
 *
 * The following fields must be baked because they affect the generated shader
 * microcode or the fundamental rasterizer command stream:
 * topology, cull, alphaToCoverage, supportDualSourceBlending, sampleCount,
 * depthFormat, stencilFormat, colorTargets, and (optionally) blendstate.
 */
struct GpuRasterDesc {
	///< Primitive assembly mode. Affects vertex grouping in the rasterizer.
	TOPOLOGY topology = TOPOLOGY_TRIANGLE_LIST;

	///< Triangle facing cull mode.
	CULL cull = CULL_NONE;

	///< When true, MSAA coverage is derived from the pixel shader's output alpha.
	///< Useful for alpha-tested foliage rendered into an MSAA buffer.
	bool alphaToCoverage = false;

	///< When true, the shader compiler enables the second pixel-shader color
	///< output (SV_Color1) for use as the second blend source. Only valid when
	///< a single color target is used. Requires blendstate to reference
	///< FACTOR_SRC1_* blend factors.
	bool supportDualSourceBlending = false;
	// TODO: WebGPU supports?

	///< MSAA sample count (1, 2, 4, 8). Must match the resolve and depth targets.
	uint8_t sampleCount = 1;

	///< Depth attachment format, or FORMAT_NONE if no depth buffer is used.
	FORMAT depthFormat = FORMAT_NONE;

	///< Stencil attachment format, or FORMAT_NONE. On most hardware the depth
	///< and stencil share the same memory allocation (e.g. FORMAT_D24_UNORM_S8_UINT).
	FORMAT stencilFormat = FORMAT_NONE;

	///< List of color render target formats and write masks. Maximum is
	///< hardware-defined (typically 8). An empty std::span means no color output
	///< (e.g. depth-only shadow pass).
	std::span<GpuColorTarget> colorTargets = {};

	///< Optional pointer to an embedded (baked) blend state. When non-null the
	///< blend equation is compiled into the PSO, allowing the driver to dead-code-
	///< eliminate color exports on the mobile shader path. When null, blending
	///< must be applied dynamically via gpuSetBlendState (requires device feature).
	std::optional<GpuBlendDesc> blendstate = std::nullopt;
	// TODO: Should we require one by default?
};

// ---------------------------------------------------------------------------
// Render pass descriptions
// ---------------------------------------------------------------------------

/**
 * LOAD_OP – Operation performed on an attachment when a render pass begins.
 *
 * On TBDR (tile-based deferred rendering) GPUs this controls whether the tile
 * memory must be populated from system memory before rasterization begins.
 *
 * LOAD_OP_LOAD – Preserve the previous attachment contents.
 * LOAD_OP_CLEAR – Clear the attachment to the specified clear value.
 * LOAD_OP_DONT_CARE – Previous contents are undefined and may be discarded.
 */
enum LOAD_OP {
	LOAD_OP_LOAD,
	LOAD_OP_CLEAR,
	LOAD_OP_DONT_CARE,
};

/**
 * STORE_OP – Operation performed on an attachment when a render pass ends.
 *
 * On TBDR GPUs this determines whether tile memory is written back to system
 * memory after rasterization completes.
 *
 * STORE_OP_STORE – Preserve the rendered contents after the pass.
 * STORE_OP_DONT_CARE – Final contents are undefined and may be discarded.
 */
enum STORE_OP {
	STORE_OP_STORE,
	STORE_OP_DONT_CARE,
};

/**
 * ClearColor – RGBA floating-point clear value for color attachments.
 */
struct ClearColor {
	float r = 0.0f;
	float g = 0.0f;
	float b = 0.0f;
	float a = 1.0f;
};

/**
 * GpuColorAttachment – Color render target binding used by GpuRenderPassDesc.
 */
struct GpuColorAttachment {
	/**
	* Texture subresource used as the render target.
	* Must have been created with TEXTURE_USAGE_RENDER_TARGET.
	*/
	const GpuTexture* texture;

	/**
	* Mip level to render into.
	*/
	uint32_t mipLevel = 0;

	/**
	* Array layer or 3D slice to render into.
	*/
	uint32_t slice = 0;

	/**
	* Optional MSAA resolve target.
	*
	* When texture is multisampled and resolveTexture is valid, the rasterized
	* image is automatically resolved into this texture at the end of the pass.
	*/
	const GpuTexture* resolveTexture = nullptr;

	/**
	* Load operation performed at render pass begin.
	*/
	LOAD_OP loadOp = LOAD_OP_LOAD;

	/**
	* Store operation performed at render pass end.
	*/
	STORE_OP storeOp = STORE_OP_STORE;

	/**
	* Clear value used when loadOp == LOAD_OP_CLEAR.
	*/
	ClearColor clearValue = {};
};

/**
 * GpuDepthAttachment – Depth/stencil attachment binding used by
 * GpuRenderPassDesc.
 *
 * Depth and stencil load/store operations are separated because some APIs
 * (Vulkan/WebGPU/Metal) expose them independently.
 */
struct GpuDepthStencilAttachment {
	/**
	* Texture subresource used as the depth/stencil target.
	* Must have been created with TEXTURE_USAGE_DEPTH_STENCIL.
	*/
	const GpuTexture* texture;

	/**
	 * Mip level to render into.
	 */
	uint32_t mipLevel = 0;

	/**
	 * Array layer or 3D slice to render into.
	 */
	uint32_t slice = 0;

	/**
	 * Depth load operation
	 */
	LOAD_OP loadOp = LOAD_OP_LOAD;
	/**
	 * Depth store operation
	 */
	STORE_OP storeOp = STORE_OP_STORE;

	/**
	 * Clear depth value used when depthLoadOp == LOAD_OP_CLEAR.
	 */
	double clearValue = 1.0f;
};

/**
 * GpuRenderPassDesc – Full render pass attachment configuration.
 *
 * This describes the render targets bound for rasterization along with the
 * attachment load/store behavior required to efficiently map onto both IMR
 * desktop GPUs and mobile TBDR architectures.
 *
 * Unlike Vulkan render passes, this object is lightweight and intended to be
 * specified directly at command recording time, more closely matching Metal
 * and WebGPU.
 */
struct GpuRenderPassDesc {
	/**
	 * Optional depth attachment.
	 */
	std::optional<GpuDepthStencilAttachment> depthAttachment = std::nullopt;

	/**
	 * Optional stencil attachment.
	 */
	std::optional<GpuDepthStencilAttachment> stencilAttachment = std::nullopt;

	/**
	 * List of color render targets.
	 *
	 * An empty span is valid for depth-only rendering passes such as shadow maps.
	 */
	std::span<const GpuColorAttachment> colorAttachments = {};
};

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/**
 * gpuCreateGraphicsPipeline – Compile a vertex + pixel shader pipeline.
 *
 * Vertex buffer bindings are replaced by raw pointer loads from the root data struct.
 * The index buffer remains special (passed to gpuDrawIndexed*) because index
 * deduplication hardware still provides significant performance benefits. Blend state
 * can optionally be embedded in GpuRasterDesc::blendstate; otherwise it is applied
 * dynamically via gpuSetBlendState.
 *
 * @param queue The GPU queue (device) on which the pipeline will be created.
 * @param vertexIR Vertex shader IR blob (SPIRV on Vulkan, WGSL on WebGPU)
 * @param fragmentIR Pixel shader IR blob (SPIRV on Vulkan, WGSL on WebGPU)
 * @param desc Rasterizer, format, and optional embedded blend state.
 */
GpuPipeline* gpuCreateGraphicsPipeline(GpuQueue* queue, std::span<const std::byte> vertexIR, std::span<const std::byte> fragmentIR, const GpuRasterDesc& desc);

/**
 * gpuCreateGraphicsMeshletPipeline – Compile a mesh shader + pixel shader pipeline.
 *
 * Mesh shaders bypass the index deduplication hardware and post-transform cache.
 * All vertex deduplication is performed offline into compact meshlets. On mobile
 * TBDR GPUs, mesh shaders are generally not supported because per-triangle tile
 * binning requires a vertex-shader-granularity primitive stream.
 *
 * @param queue The GPU queue (device) on which the pipeline will be created.
 * @param meshletIR Mesh shader IR blob (SPIRV on Vulkan, WGSL on WebGPU)
 * @param fragmentIR Pixel shader IR blob (SPIRV on Vulkan, WGSL on WebGPU)
 * @param desc Rasterizer, format, and optional embedded blend state.
 */
// GpuPipeline gpuCreateGraphicsMeshletPipeline(GpuQueue* queue, std::span<const std::byte> meshletIR, std::span<const std::byte> fragmentIR, GpuRasterDesc desc);
// NOTE: WebGPU doesn't yet support mesh shaders!

// ---------------------------------------------------------------------------
// Separate state objects
// ---------------------------------------------------------------------------

/**
 * gpuCreateDepthStencilState – Bake a depth/stencil configuration into a reusable
 * state object. Applied per-command-buffer via gpuSetDepthStencilState before
 * draw calls. Separating depth/stencil state from the PSO (unlike DX12) reduces
 * permutations and avoids costly PSO recompiles when only depth bias or the stencil
 * reference changes.
 *
 * @param queue The GPU queue (device) on which the state object will be created.
 * @param desc Depth/stencil test and write configuration.
 */
GpuDepthStencilState* gpuCreateDepthStencilState(GpuQueue* queue, const GpuDepthStencilDesc& desc);

/**
 * gpuCreateBlendState – Bake a blend configuration into a reusable state object.
 * Applied per-command-buffer via gpuSetBlendState. Only valid on devices that
 * expose fixed-function blend hardware independently of the PSO (check device
 * feature flags). On mobile TBDR GPUs, prefer embedding the blend state in the PSO
 * or using framebuffer-fetch intrinsics for a parametrised blend formula.
 *
 * @param queue The GPU queue (device) on which the state object will be created.
 * @param desc Blend equation and factor configuration.
 */
GpuBlendState* gpuCreateBlendState(GpuQueue* queue, const GpuBlendDesc& desc);

/**
 * gpuFreeDepthStencilState – Release a GpuDepthStencilState object.
 *
 * @param queue The GPU queue (device) the state object was created on.
 * @param state Depth/stencil state object to release.
 */
void gpuFreeDepthStencilState(GpuQueue* queue, GpuDepthStencilState* state);

/**
 * gpuFreeBlendState – Release a GpuBlendState object.
 *
 * @param queue The GPU queue (device) the state object was created on.
 * @param state Blend state object to release.
 */
void gpuFreeBlendState(GpuQueue* queue, GpuBlendState* state);

// ---------------------------------------------------------------------------
// GPU commands – state binding
// ---------------------------------------------------------------------------

/**
 * gpuSetDepthStencilState – Apply a pre-baked GpuDepthStencilState for subsequent
 * draw calls. Emits a small hardware command packet to configure the depth/stencil
 * unit without touching the shader microcode or requiring a PSO recompile.
 *
* NOTE: On WebGPU this usually involves a PSO recompile!
 *
 * @param cmd Command buffer to record into.
 * @param state Depth/stencil state object to apply, created by gpuCreateDepthStencilState.
 */
void gpuSetDepthStencilState(GpuCommandBuffer* cmd, const GpuDepthStencilState* state);

/**
 * gpuSetBlendState – Apply a pre-baked GpuBlendState for subsequent draw calls on
 * desktop GPUs with dynamic blend hardware. Requires the corresponding device
 * feature flag. On mobile TBDR GPUs this call is unsupported; blend state must be
 * embedded in the PSO or implemented via framebuffer-fetch intrinsics.
 *
* NOTE: On WebGPU this usually involves a PSO recompile!
 *
 * @param cmd Command buffer to record into.
 * @param state Blend state object to apply, created by gpuCreateBlendState.
 */
void gpuSetBlendState(GpuCommandBuffer* cmd, const GpuBlendState* state);

/**
 * gpuSetViewportEXT – Set the viewport transform used to map clip-space
 * coordinates onto the render target.
 *
 * @note Set automatically by gpuBeginRenderPass to cover the full render target;
 * call this afterward to override it.
 *
 * @param cmd Command buffer to record into.
 * @param extent Width and height of the viewport in pixels.
 * @param origin Top-left corner of the viewport in pixels (default: {0, 0}).
 * @param depth_min Minimum depth range value (default: 0).
 * @param depth_max Maximum depth range value (default: 1).
 */
void gpuSetViewportEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin = {0, 0}, float depth_min = 0, float depth_max = 1);

/**
 * gpuSetScissorRectEXT – Set the scissor rectangle that clips rasterizer output.
 *
 * @note Set automatically by gpuBeginRenderPass to cover the full render target;
 * call this afterward to override it.
 *
 * @param cmd Command buffer to record into.
 * @param extent Width and height of the scissor rectangle in pixels.
 * @param origin Top-left corner of the scissor rectangle in pixels (default: {0, 0}).
 */
void gpuSetScissorRectEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin = {0, 0});

// ---------------------------------------------------------------------------
// GPU commands – render passes
// ---------------------------------------------------------------------------

/**
 * gpuBeginRenderPass – Configure the rasterizer for a set of render targets and
 * begin recording rasterization commands.
 *
 * Performs the hardware-specific render-target setup and (on TBDR GPUs) triggers
 * the tile load operations described by the attachment load ops. Fast-clear
 * elimination is handled transparently by the driver when the clear color changes.
 *
 * Does NOT insert an automatic barrier before the pass. If a previous pass wrote
 * to any of these targets via compute shaders, the user must call gpuBarrier
 * with appropriate HAZARD_DEPTH_STENCIL or HAZARD_DESCRIPTORS flags beforehand.
 *
 * @param cmd Command buffer to record into.
 * @param desc Render target attachments and their load/store behavior.
 */
void gpuBeginRenderPass(GpuCommandBuffer* cmd, const GpuRenderPassDesc& desc);

/**
 * gpuEndRenderPass – End the current render pass and trigger (on TBDR GPUs) tile
 * store operations for attachments whose storeOp is STORE_OP_STORE.
 *
 * Does NOT insert an automatic barrier after the pass. If subsequent passes need
 * to read these render targets as textures, the user must call gpuBarrier with
 * STAGE_RASTER_COLOR_OUT / STAGE_RASTER_DEPTH_OUT as the producer stage.
 *
 * @param cmd The command buffer to bind against.
 * @param desc (optionally) the same render pass descriptor that was passed to gpuBeginRenderPass (transitions the images to a more optimal presentation layout if provided)
 */
void gpuEndRenderPass(GpuCommandBuffer* cmd, std::optional<const GpuRenderPassDesc> desc = {});

// ---------------------------------------------------------------------------
// GPU commands – rasterizer draw calls
// ---------------------------------------------------------------------------

/**
 * gpuDrawIndexedInstanced – Draw indexed geometry with instancing.
 *
 * The vertex shader receives `vertexData` as its root data pointer and the pixel
 * shader receives `fragmentData`. Passing the same pointer for both is valid if the
 * two shaders share their input struct. Vertex buffers are replaced by raw pointer
 * loads from the vertex data struct; only the index buffer is special (hardware
 * index deduplication still provides a 4-6x vertex shading reduction in practice).
 *
 * @param cmd Command buffer to record into.
 * @param vertex_data GPU pointer to the vertex shader root data struct.
 * @param fragment_data GPU pointer to the pixel shader root data struct.
 * @param indices GPU pointer to the index buffer (uint16 or uint32 array).
 * @param index_count Number of indices to draw.
 * @param instance_count Number of instances (1 for non-instanced geometry).
 * @param index_type expected type of the bound indices
 * @param no_offsets When true it skips calculating offsets into buffers for the gpu*'s
 * @param no_index_buffer_changes When true reuses the last value in interal index buffer (skips copying any changed indices)
 * 
 */
void gpuDrawIndexedInstanced(GpuCommandBuffer* cmd,
 gpu* vertex_data, gpu* fragment_data,
 gpu* indices, uint32_t index_count, uint32_t instance_count, 
 INDEX_TYPE_EXT index_type = INDEX_TYPE_UINT32, bool no_offsets = false, bool no_index_buffer_changes = false);

/**
 * gpuDrawIndexedInstancedIndirect – GPU-driven indexed instanced draw. Reads the
 * draw arguments (index count, instance count, etc.) from GPU memory, enabling
 * the GPU to cull and compact draw lists without CPU round-trips.
 *
 * Both the root data pointers and the draw arguments are GPU pointers, making this
 * a fully indirect draw — an improvement over existing APIs (DX12 ExecuteIndirect,
 * Vulkan vkCmdDrawIndexedIndirect) which require the CPU to supply the root data.
 *
 * @param cmd Command buffer to record into.
 * @param vertex_data GPU pointer to the vertex shader root data struct.
 * @param fragment_data GPU pointer to the pixel shader root data struct.
 * @param indices GPU pointer to the index buffer.
 * @param args GPU pointer to an indirect draw argument struct
 * @param index_type expected type of the bound indices
 * @param no_offsets When true it skips calculating offsets into buffers for the gpu*'s
 * @param no_index_buffer_changes When true reuses the last value in interal index buffer (skips copying any changed indices)
 */
void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer* cmd,
 gpu* vertex_data, gpu* fragment_data,
 gpu* indices, gpu* args, 
 INDEX_TYPE_EXT index_type = INDEX_TYPE_UINT32, bool no_offsets = false, bool no_index_buffer_changes = false);

// TODO: Why does this one not take an index buffer?
// /**
//  * gpuDrawIndexedInstancedIndirectMulti – Multi-draw indirect with per-draw root data.
//  *
//  * Allows the GPU to drive an entire batch of draw calls, each with its own root
//  * data structs, avoiding the per-draw CPU overhead of vkCmdDrawIndexedIndirectCount
//  * (which shares a single descriptor set for all draws). This enables clean, efficient
//  * per-draw material/texture switching with no hacks or shader indirections.
//  *
//  * A stride of 0 for vertex or pixel data means the same pointer is replicated for
//  * every draw (akin to a broadcast), reducing memory if all draws share parameters.
//  *
//  * @param cmd Command buffer to record into.
//  * @param dataVxGpu GPU pointer to the array of vertex shader root data structs.
//  * @param vxStride Stride between vertex data entries in bytes (0 = broadcast first).
//  * @param dataPxGpu GPU pointer to the array of pixel shader root data structs.
//  * @param pxStride Stride between pixel data entries in bytes (0 = broadcast first).
//  * @param argsGpu GPU pointer to an array of indirect draw argument structs.
//  * @param drawCountGpu GPU pointer to a uint32 holding the actual draw count.
//  * @param index_type expected type of the bound indices
//  * @param no_offsets When true it skips calculating offsets into buffers for the gpu*'s
//  * @param no_index_buffer_changes When true reuses the last value in interal index buffer (skips copying any changed indices)
//  */
// void gpuDrawIndexedInstancedIndirectMulti(GpuCommandBuffer* cmd,
//  gpu* dataVxGpu, uint32_t vxStride,
//  gpu* dataPxGpu, uint32_t pxStride,
//  gpu* argsGpu, gpu* drawCountGpu, 
//  INDEX_TYPE_EXT index_type = INDEX_TYPE_UINT32, bool no_offsets = false, bool no_index_buffer_changes = false);

/**
 * gpuDrawMeshlets – Launch a mesh shader pass, dispatching a 3D grid of mesh
 * shader thread groups. Each group outputs a self-contained meshlet (up to 256
 * vertices, 128 triangles on AMD; 126 vertices, 64 triangles on Nvidia). Offline
 * vertex deduplication eliminates the need for an index deduplication unit.
 *
 * @note WebGPU doesn't yet support mesh shaders!
 *
 * @param cmd Command buffer to record into.
 * @param meshlet_data GPU pointer to the mesh shader root data struct.
 * @param fragment_data GPU pointer to the pixel shader root data struct.
 * @param dim Thread group grid dimensions (typically x = meshlet count).
 */
void gpuDrawMeshlets(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, uvec3 dim);
// 

/**
 * gpuDrawMeshletsIndirect – GPU-driven mesh shader dispatch. Reads the thread
 * group grid dimensions from GPU memory, enabling the GPU to cull and compact
 * meshlet lists without CPU round-trips.
 *
 * @note WebGPU doesn't yet support mesh shaders!
 *
 * @param cmd Command buffer to record into.
 * @param meshlet_data GPU pointer to the mesh shader root data struct.
 * @param fragment_data GPU pointer to the pixel shader root data struct.
 * @param dim GPU pointer to a uvec3 containing group dimensions.
 * @param no_offsets When true it skips calculating offsets into buffers for the gpu*'s
 */
void gpuDrawMeshletsIndirect(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, gpu* dim, bool no_offsets = false);
