#pragma once

/**
 * A minimalistic modern GPU API based on the design described in: "No Graphics API" by Sebastian Aaltonen https://www.sebastianaaltonen.com/blog/no-graphics-api
 *
 * This file contains the graphics specific public API while compute.h contains all of the shared code used for both graphics and compute.
 *
 * The original API did not include any API for swapchain/surface support, so surface.h lifts a WebGPU inspired one.
 */

#include "compute.h"

// ---------------------------------------------------------------------------
// Basic math types used in the API
// ---------------------------------------------------------------------------

/**
 * uvec2 – Two unsigned integers: a width and a height.
 */
typedef struct uvec2 { uint32_t x, y; } uvec2;

/**
 * ivec2 – Two signed integers: a position in pixels.
 */
typedef struct ivec2 { int32_t x, y; } ivec2;

// ---------------------------------------------------------------------------
// Opaque GPU object handles
// ---------------------------------------------------------------------------

/**
 * GpuDepthStencilState – A pre-baked depth/stencil configuration object. Separating this state from the PSO (as Metal does) reduces pipeline permutations. Applied per-command-buffer via gpuSetDepthStencilState. Created by gpuCreateDepthStencilState.
 */
typedef struct GpuDepthStencilState GpuDepthStencilState;

/**
 * GpuBlendState – A pre-baked alpha-blend configuration object. On desktop GPUs that expose fixed-function blend hardware this can be applied dynamically without recompiling the PSO, significantly cutting blend-mode permutations. On mobile TBDR GPUs blending is burned into the pixel shader microcode; users may instead use framebuffer-fetch intrinsics and author a parametrised formula. Created by gpuCreateBlendState.
 *
 * @note Dynamic blending is something the device has to provide (on Vulkan it is VK_EXT_extended_dynamic_state3, which the backend requires), rather than something a program asks about and branches on -- a device without it cannot run this API at all.
 */
typedef struct GpuBlendState GpuBlendState;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/**
 * CULL – Triangle-facing culling mode for rasterisation.
 *
 * CULL_CCW – Cull counter-clockwise (back-face) triangles (typical default).
 * CULL_CW – Cull clockwise triangles.
 * CULL_ALL – Cull all triangles (useful for depth-only / shadow-map prepass that only needs depth from the rasterizer's early-Z).
 * CULL_NONE – No culling; both faces are rasterized.
 */
typedef enum CULL { CULL_CCW, CULL_CW, CULL_ALL, CULL_NONE } CULL;

/**
 * DEPTH_FLAGS – Bitmask controlling depth-buffer access in GpuDepthStencilDesc. Combine with bitwise-OR, e.g. DEPTH_READ | DEPTH_WRITE.
 *
 * DEPTH_READ – The depth buffer is tested (early-Z / depth test enabled).
 * DEPTH_WRITE – Passing fragments update the depth buffer.
 */
typedef enum DEPTH_FLAGS { DEPTH_READ = 0x1, DEPTH_WRITE = 0x2 } DEPTH_FLAGS;

/**
 * STENCIL_OP – Action taken on the stencil buffer depending on depth/stencil test outcomes.
 */
typedef enum STENCIL_OP {
	STENCIL_OP_KEEP, ///< Keep the current stencil value unchanged.
	STENCIL_OP_ZERO, ///< Set stencil to 0.
	STENCIL_OP_REPLACE, ///< Replace with the reference value.
	STENCIL_OP_INCR_SAT, ///< Increment, clamping at the maximum value.
	STENCIL_OP_DECR_SAT, ///< Decrement, clamping at 0.
	STENCIL_OP_INVERT, ///< Bitwise-invert the stencil value.
	STENCIL_OP_INCR_WRAP, ///< Increment, wrapping to 0 on overflow.
	STENCIL_OP_DECR_WRAP, ///< Decrement, wrapping on underflow.
} STENCIL_OP;

/**
 * BLEND – RGB / alpha blend equation applied by the fixed-function blender (or emulated via framebuffer-fetch on mobile TBDR GPUs): result = src * srcFactor <BLEND> dst * dstFactor.
 */
typedef enum BLEND {
	BLEND_ADD, ///< result = src*srcF + dst*dstF (standard alpha blend)
	BLEND_SUBTRACT, ///< result = src*srcF - dst*dstF
	BLEND_REV_SUBTRACT, ///< result = dst*dstF - src*srcF
	BLEND_MIN, ///< result = min(src, dst) (factors ignored)
	BLEND_MAX, ///< result = max(src, dst) (factors ignored)
} BLEND;

/**
 * FACTOR – Blend scale factors applied to source and destination colours/alphas. Maps 1:1 to DX12/Vulkan blend factor enumerations.
 */
typedef enum FACTOR {
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
} FACTOR;

/**
 * TOPOLOGY – Primitive assembly topology for the rasterizer. Triangle fans are left out because WebGPU has none; points and lines because they would only add a line width.
 *
 * TOPOLOGY_TRIANGLE_LIST – Every 3 vertices form an independent triangle.
 * TOPOLOGY_TRIANGLE_STRIP – Vertices share edges; (n-2) triangles from n vertices.
 */
typedef enum TOPOLOGY {
	TOPOLOGY_TRIANGLE_LIST,
	TOPOLOGY_TRIANGLE_STRIP,
} TOPOLOGY;

/**
 * INDEX_TYPE_EXT – The width of each entry in an index buffer.
 */
typedef enum INDEX_TYPE_EXT {
	INDEX_TYPE_UINT8, ///< Not supported on WebGPU.
	INDEX_TYPE_UINT16,
	INDEX_TYPE_UINT32
} INDEX_TYPE_EXT;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/**
 * GpuStencil – Per-face stencil operation description used inside GpuDepthStencilDesc.
 */
typedef struct GpuStencil {
	OP test NOAPI_DEFAULT(OP_ALWAYS); ///< Stencil comparison function.
	STENCIL_OP failOp NOAPI_DEFAULT(STENCIL_OP_KEEP); ///< Action when stencil test fails.
	STENCIL_OP passOp NOAPI_DEFAULT(STENCIL_OP_KEEP); ///< Action when both tests pass.
	STENCIL_OP depthFailOp NOAPI_DEFAULT(STENCIL_OP_KEEP); ///< Action when stencil passes but depth fails.
	uint8_t reference NOAPI_DEFAULT(0); ///< Stencil reference value.
} GpuStencil;

/**
 * GPU_STENCIL_DEFAULT – The defaults above as an initializer, for C. Keep in sync with GpuStencil.
 */
#define GPU_STENCIL_DEFAULT { OP_ALWAYS, STENCIL_OP_KEEP, STENCIL_OP_KEEP, STENCIL_OP_KEEP, 0 }

/**
 * GpuDepthStencilDesc – Full description of depth/stencil test and write behavior.
 *
 * This is intentionally separated from the PSO (unlike DX12, where most of these fields are baked into the pipeline object). Separating depth/stencil state reduces the number of PSO permutations and allows changing it cheaply at draw-call time via gpuSetDepthStencilState, matching the actual GPU command packet granularity.
 *
 * All fields have sensible defaults: by default depth testing is disabled and stencil operations are all KEEP / ALWAYS.
 */
typedef struct GpuDepthStencilDesc {
	DEPTH_FLAGS depthMode NOAPI_DEFAULT((DEPTH_FLAGS)0); ///< Bitmask: DEPTH_READ, DEPTH_WRITE, or both.
	OP depthTest NOAPI_DEFAULT(OP_ALWAYS); ///< Depth comparison function (e.g. OP_LESS_EQUAL).
	float depthBias NOAPI_DEFAULT(0.0f); ///< Constant depth value added to each fragment.
	float depthBiasSlopeFactor NOAPI_DEFAULT(0.0f); ///< Depth bias scaled by triangle slope (shadow mapping).
	float depthBiasClamp NOAPI_DEFAULT(0.0f); ///< Maximum magnitude of the slope-scaled depth bias.
	uint8_t stencilReadMask NOAPI_DEFAULT(0xff); ///< Mask ANDed with the stencil buffer before comparison.
	uint8_t stencilWriteMask NOAPI_DEFAULT(0xff); ///< Mask ANDed with written stencil values.
	GpuStencil stencilFront NOAPI_DEFAULT(GPU_STENCIL_DEFAULT); ///< Stencil ops for front-facing (CCW) triangles.
	GpuStencil stencilBack NOAPI_DEFAULT(GPU_STENCIL_DEFAULT); ///< Stencil ops for back-facing (CW) triangles.
} GpuDepthStencilDesc;

/**
 * GPU_DEPTH_STENCIL_DESC_DEFAULT – The defaults above as an initializer, for C. Keep in sync with GpuDepthStencilDesc.
 */
#define GPU_DEPTH_STENCIL_DESC_DEFAULT { (DEPTH_FLAGS)0, OP_ALWAYS, 0.0f, 0.0f, 0.0f, 0xff, 0xff, GPU_STENCIL_DEFAULT, GPU_STENCIL_DEFAULT }

/**
 * GpuBlendDesc – Alpha blending configuration for a single color render target.
 *
 * Default values implement opaque (no-blending) output: src * 1 + dst * 0 = src. A standard "over" alpha-blend sets srcColorFactor = FACTOR_SRC_ALPHA and dstColorFactor = FACTOR_ONE_MINUS_SRC_ALPHA.
 *
 * On desktop GPUs with fixed-function blend hardware this state can be applied dynamically via gpuSetBlendState. On mobile TBDR GPUs it is typically compiled into the pixel shader; consider framebuffer-fetch + a parametrized blend formula instead to avoid PSO permutations.
 */
typedef struct GpuBlendDesc {
	BLEND colorOp NOAPI_DEFAULT(BLEND_ADD); ///< Blend equation for RGB channels.
	FACTOR srcColorFactor NOAPI_DEFAULT(FACTOR_ONE); ///< Scale factor applied to the source (incoming) color.
	FACTOR dstColorFactor NOAPI_DEFAULT(FACTOR_ZERO); ///< Scale factor applied to the destination (existing) color.
	BLEND alphaOp NOAPI_DEFAULT(BLEND_ADD); ///< Blend equation for the alpha channel.
	FACTOR srcAlphaFactor NOAPI_DEFAULT(FACTOR_ONE); ///< Scale factor applied to the source alpha.
	FACTOR dstAlphaFactor NOAPI_DEFAULT(FACTOR_ZERO); ///< Scale factor applied to the destination alpha.
	uint8_t colorWriteMask NOAPI_DEFAULT(0xf); ///< Per-channel write enable: bit 0=R, 1=G, 2=B, 3=A.
} GpuBlendDesc;

/**
 * GPU_BLEND_DESC_DEFAULT – The defaults above as an initializer, for C. Keep in sync with GpuBlendDesc.
 */
#define GPU_BLEND_DESC_DEFAULT { BLEND_ADD, FACTOR_ONE, FACTOR_ZERO, BLEND_ADD, FACTOR_ONE, FACTOR_ZERO, 0xf }

/**
 * GpuOptionalBlendDesc – An optionally present GpuBlendDesc: std::optional<GpuBlendDesc> in C++, a {has_value, value} pair in C.
 */
NOAPI_OPTIONAL_TYPE(GpuOptionalBlendDesc, GpuBlendDesc);

/**
 * GpuColorTarget – Render target format and write mask entry in GpuRasterDesc.
 *
 * writeMask mirrors the DX12 / Vulkan concept of per-render-target write masks. Baking it into the PSO lets the shader compiler dead-code-eliminate color outputs that are masked off, saving ALU and export bandwidth. This is distinct from GpuBlendDesc::colorWriteMask, which applies when a dynamic GpuBlendState is in use.
 */
typedef struct GpuColorTarget {
	FORMAT format NOAPI_DEFAULT(FORMAT_NONE); ///< Pixel format of this render target attachment.
	uint8_t writeMask NOAPI_DEFAULT(0xf); ///< Bitmask: bit 0=R, 1=G, 2=B, 3=A. 0xf = write all channels.
} GpuColorTarget;

/**
 * GPU_COLOR_TARGET_DEFAULT – The defaults above as an initializer, for C. Keep in sync with GpuColorTarget.
 */
#define GPU_COLOR_TARGET_DEFAULT { FORMAT_NONE, 0xf }

/**
 * GpuColorTargetSpan – A read only list of color render targets: std::span<const GpuColorTarget> in C++, a {ptr, count} pair in C.
 */
NOAPI_SPAN_TYPE(GpuColorTargetSpan, const GpuColorTarget);

/**
 * GpuRasterDesc – Minimal rasterizer state baked into a graphics PSO.
 *
 * The goal is to keep this struct small to minimize PSO permutations. States that change frequently (depth/stencil behavior, blend modes on capable hardware) are moved out into separate dynamically-applied state objects.
 *
 * The following fields must be baked because they affect the generated shader microcode or the fundamental rasterizer command stream: topology, cull, alphaToCoverage, supportDualSourceBlending, sampleCount, depthFormat, stencilFormat, colorTargets, and (optionally) blendstate.
 */
typedef struct GpuRasterDesc {
	/** Primitive assembly mode. Affects vertex grouping in the rasterizer. */
	TOPOLOGY topology NOAPI_DEFAULT(TOPOLOGY_TRIANGLE_LIST);

	/** Triangle facing cull mode. */
	CULL cull NOAPI_DEFAULT(CULL_NONE);

	/** When true, MSAA coverage is derived from the pixel shader's output alpha. Useful for alpha-tested foliage rendered into an MSAA buffer. */
	bool alphaToCoverage NOAPI_DEFAULT(false);

	/** When true, the shader compiler enables the second pixel-shader color output (SV_Color1) for use as the second blend source. Only valid when a single color target is used. Requires blendstate to reference FACTOR_SRC1_* blend factors. */
	bool supportDualSourceBlending NOAPI_DEFAULT(false);

	/** MSAA sample count (1, 2, 4, 8). Must match the resolve and depth targets. */
	uint8_t sampleCount NOAPI_DEFAULT(1);

	/** Depth attachment format, or FORMAT_NONE if no depth buffer is used. */
	FORMAT depthFormat NOAPI_DEFAULT(FORMAT_NONE);

	/** Stencil attachment format, or FORMAT_NONE. On most hardware the depth and stencil share the same memory allocation (e.g. FORMAT_D24_PLUS_S8_UINT). */
	FORMAT stencilFormat NOAPI_DEFAULT(FORMAT_NONE);

	/** List of color render target formats and write masks. Maximum is hardware-defined (typically 8). An empty span means no color output (e.g. depth-only shadow pass). */
	GpuColorTargetSpan colorTargets NOAPI_DEFAULT({});

	/** Optional embedded (baked) blend state. When present the blend equation is compiled into the PSO, allowing the driver to dead-code-eliminate color exports on the mobile shader path. When absent, blending is applied dynamically via gpuSetBlendState. */
	GpuOptionalBlendDesc blendstate NOAPI_DEFAULT({});
} GpuRasterDesc;

/**
 * GPU_RASTER_DESC_DEFAULT – The defaults above as an initializer, for C. Keep in sync with GpuRasterDesc.
 */
#define GPU_RASTER_DESC_DEFAULT { TOPOLOGY_TRIANGLE_LIST, CULL_NONE, false, false, 1, FORMAT_NONE, FORMAT_NONE, {NULL, 0}, {false, GPU_BLEND_DESC_DEFAULT} }

// ---------------------------------------------------------------------------
// Render pass descriptions
// ---------------------------------------------------------------------------

/**
 * LOAD_OP – Operation performed on an attachment when a render pass begins.
 *
 * On TBDR (tile-based deferred rendering) GPUs this controls whether the tile memory must be populated from system memory before rasterization begins.
 *
 * LOAD_OP_LOAD – Preserve the previous attachment contents.
 * LOAD_OP_CLEAR – Clear the attachment to the specified clear value.
 * LOAD_OP_DONT_CARE – Previous contents are undefined and may be discarded.
 */
typedef enum LOAD_OP {
	LOAD_OP_LOAD,
	LOAD_OP_CLEAR,
	LOAD_OP_DONT_CARE,
} LOAD_OP;

/**
 * STORE_OP – Operation performed on an attachment when a render pass ends.
 *
 * On TBDR GPUs this determines whether tile memory is written back to system memory after rasterization completes.
 *
 * STORE_OP_STORE – Preserve the rendered contents after the pass.
 * STORE_OP_DONT_CARE – Final contents are undefined and may be discarded.
 */
typedef enum STORE_OP {
	STORE_OP_STORE,
	STORE_OP_DONT_CARE,
} STORE_OP;

/**
 * ClearColor – RGBA floating-point clear value for color attachments.
 */
typedef struct ClearColor {
	float r NOAPI_DEFAULT(0.0f);
	float g NOAPI_DEFAULT(0.0f);
	float b NOAPI_DEFAULT(0.0f);
	float a NOAPI_DEFAULT(1.0f);
} ClearColor;

/**
 * GPU_CLEAR_COLOR_DEFAULT – The defaults above as an initializer, for C. Keep in sync with ClearColor.
 */
#define GPU_CLEAR_COLOR_DEFAULT { 0.0f, 0.0f, 0.0f, 1.0f }

/**
 * GpuColorAttachment – Color render target binding used by GpuRenderPassDesc.
 */
typedef struct GpuColorAttachment {
	/** Texture subresource used as the render target. Must have been created with USAGE_COLOR_ATTACHMENT. */
	const GpuTexture* texture;

	/** Mip level to render into. */
	uint32_t mipLevel NOAPI_DEFAULT(0);

	/** Array layer or 3D slice to render into. */
	uint32_t slice NOAPI_DEFAULT(0);

	/** Optional MSAA resolve target. When texture is multisampled and resolveTexture is valid, the rasterized image is automatically resolved into this texture at the end of the pass. */
	const GpuTexture* resolveTexture NOAPI_DEFAULT(NULL);

	/** Load operation performed at render pass begin. */
	LOAD_OP loadOp NOAPI_DEFAULT(LOAD_OP_LOAD);

	/** Store operation performed at render pass end. */
	STORE_OP storeOp NOAPI_DEFAULT(STORE_OP_STORE);

	/** Clear value used when loadOp == LOAD_OP_CLEAR. */
	ClearColor clearValue NOAPI_DEFAULT({});
} GpuColorAttachment;

/**
 * GPU_COLOR_ATTACHMENT_DEFAULT – The defaults above as an initializer, for C. Keep in sync with GpuColorAttachment.
 */
#define GPU_COLOR_ATTACHMENT_DEFAULT { NULL, 0, 0, NULL, LOAD_OP_LOAD, STORE_OP_STORE, GPU_CLEAR_COLOR_DEFAULT }

/**
 * GpuColorAttachmentSpan – A read only list of color attachments: std::span<const GpuColorAttachment> in C++, a {ptr, count} pair in C.
 */
NOAPI_SPAN_TYPE(GpuColorAttachmentSpan, const GpuColorAttachment);

/**
 * GpuDepthStencilAttachment – Depth/stencil attachment binding used by GpuRenderPassDesc.
 *
 * Depth and stencil load/store operations are separated because some APIs (Vulkan/WebGPU/Metal) expose them independently.
 */
typedef struct GpuDepthStencilAttachment {
	/** Texture subresource used as the depth/stencil target. Must have been created with USAGE_DEPTH_STENCIL_ATTACHMENT. */
	const GpuTexture* texture;

	/** Mip level to render into. */
	uint32_t mipLevel NOAPI_DEFAULT(0);

	/** Array layer or 3D slice to render into. */
	uint32_t slice NOAPI_DEFAULT(0);

	/** Load operation performed at render pass begin. */
	LOAD_OP loadOp NOAPI_DEFAULT(LOAD_OP_LOAD);

	/** Store operation performed at render pass end. */
	STORE_OP storeOp NOAPI_DEFAULT(STORE_OP_STORE);

	/** Clear value used when loadOp == LOAD_OP_CLEAR: a depth, or for the stencil attachment a stencil value. */
	double clearValue NOAPI_DEFAULT(1.0f);
} GpuDepthStencilAttachment;

/**
 * GPU_DEPTH_STENCIL_ATTACHMENT_DEFAULT – The defaults above as an initializer, for C. Keep in sync with GpuDepthStencilAttachment.
 */
#define GPU_DEPTH_STENCIL_ATTACHMENT_DEFAULT { NULL, 0, 0, LOAD_OP_LOAD, STORE_OP_STORE, 1.0 }

/**
 * GpuOptionalDepthStencilAttachment – An optionally present GpuDepthStencilAttachment: std::optional<GpuDepthStencilAttachment> in C++, a {has_value, value} pair in C.
 */
NOAPI_OPTIONAL_TYPE(GpuOptionalDepthStencilAttachment, GpuDepthStencilAttachment);

/**
 * GpuRenderPassDesc – Full render pass attachment configuration.
 *
 * This describes the render targets bound for rasterization along with the attachment load/store behavior required to efficiently map onto both IMR desktop GPUs and mobile TBDR architectures.
 *
 * Unlike Vulkan render passes, this object is lightweight and intended to be specified directly at command recording time, more closely matching Metal and WebGPU.
 */
typedef struct GpuRenderPassDesc {
	/** Optional depth attachment. */
	GpuOptionalDepthStencilAttachment depthAttachment NOAPI_DEFAULT({});

	/** Optional stencil attachment. */
	GpuOptionalDepthStencilAttachment stencilAttachment NOAPI_DEFAULT({});

	/** List of color render targets. An empty span is valid for depth-only rendering passes such as shadow maps. */
	GpuColorAttachmentSpan colorAttachments NOAPI_DEFAULT({});
} GpuRenderPassDesc;

/**
 * GPU_RENDER_PASS_DESC_DEFAULT – The defaults above as an initializer, for C. Keep in sync with GpuRenderPassDesc.
 */
#define GPU_RENDER_PASS_DESC_DEFAULT { \
	{false, GPU_DEPTH_STENCIL_ATTACHMENT_DEFAULT}, \
	{false, GPU_DEPTH_STENCIL_ATTACHMENT_DEFAULT}, \
	{NULL, 0} \
}

/**
 * GpuOptionalRenderPassDesc – An optionally present GpuRenderPassDesc: std::optional<GpuRenderPassDesc> in C++, a {has_value, value} pair in C.
 */
NOAPI_OPTIONAL_TYPE(GpuOptionalRenderPassDesc, GpuRenderPassDesc);

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

NOAPI_EXTERN_C_BEGIN

/**
 * gpuCreateGraphicsPipeline – Compile a vertex + pixel shader pipeline.
 *
 * Vertex buffer bindings are replaced by raw pointer loads from the root data struct. The index buffer remains special (passed to gpuDrawIndexed*) because index deduplication hardware still provides significant performance benefits. Blend state can optionally be embedded in GpuRasterDesc::blendstate; otherwise it is applied dynamically via gpuSetBlendState.
 *
 * @code{.cpp}
 * auto shader = string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
 * 	[shader("fragment")] float4 fragmentMain() : SV_Target { return gpuFragmentData<float4>()[0]; }
 * )slang");
 * GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
 * auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {
 * 	.colorTargets = {&target, 1},
 * 	.blendstate = GpuBlendDesc{.srcColorFactor = FACTOR_SRC_ALPHA, .dstColorFactor = FACTOR_ONE_MINUS_SRC_ALPHA},
 * });
 * assert(pipeline);
 * assert(!gpuCreateGraphicsPipeline(queue, string_to_bytes("not a shader"), shader, {.colorTargets = {&target, 1}}));
 * gpuFreePipeline(queue, pipeline);
 * @endcode
 *
 * @param queue The GPU queue (device) on which the pipeline will be created.
 * @param vertexIR Vertex shader source (Slang; see shaders.h). May be the same module as \p fragmentIR, since entry points are found by stage.
 * @param fragmentIR Pixel shader source (Slang; see shaders.h).
 * @param desc Rasterizer, format, and optional embedded blend state.
 * @return The pipeline, or NULL if a shader did not compile (reported through gpuSetShaderDiagnosticCallbackEXT's callback) or the device refused it (errno set).
 */
GpuPipeline* gpuCreateGraphicsPipeline(GpuQueue* queue, GpuByteSpan vertexIR, GpuByteSpan fragmentIR, NOAPI_CONST_REF(GpuRasterDesc) desc);

/**
 * gpuCreateGraphicsMeshletPipeline – Compile a mesh shader + pixel shader pipeline.
 *
 * Mesh shaders bypass the index deduplication hardware and post-transform cache. All vertex deduplication is performed offline into compact meshlets. On mobile TBDR GPUs, mesh shaders are generally not supported because per-triangle tile binning requires a vertex-shader-granularity primitive stream.
 *
 * The mesh shader replaces the vertex stage rather than joining it, so the pipeline has no vertex input and no index buffer: the entry point is marked [shader("mesh")] and reads its root data through the `noapi` module's gpuMeshletData<T>(), which is the same pointer gpuVertexData<T>() would hand back. Draw one with gpuDrawMeshlets or gpuDrawMeshletsIndirect; GpuRasterDesc::topology is ignored, since the shader declares its own output topology.
 *
 * @code{.cpp}
 * auto shader = string_to_bytes(R"slang(
 * 	import noapi;
 * 	struct Vertex { float4 position : SV_Position; }
 * 	[shader("mesh")] [outputtopology("triangle")] [numthreads(1, 1, 1)]
 * 	void meshMain(out vertices Vertex corners[3], out indices uint3 triangles[1]) {
 * 		SetMeshOutputCounts(3, 1);
 * 		corners[0].position = float4(-1, -1, 0, 1); corners[1].position = float4(3, -1, 0, 1); corners[2].position = float4(-1, 3, 0, 1);
 * 		triangles[0] = uint3(0, 1, 2);
 * 	}
 * 	[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
 * )slang");
 * GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
 * auto pipeline = gpuCreateGraphicsMeshletPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
 * assert((pipeline != nullptr) == gpuGetCapabilitiesEXT(queue).mesh_shaders);
 * if(pipeline) gpuFreePipeline(queue, pipeline);
 * @endcode
 *
 * @note This is an optional feature. It is available only where gpuGetCapabilitiesEXT().mesh_shaders is true -- a Vulkan device with VK_EXT_mesh_shader, and never on WebGPU, which has no mesh shaders in any form. Where it is false this reports through the diagnostic callback and returns NULL, without compiling anything, so a program can either branch on the capability or branch on the NULL.
 *
 * @param queue The GPU queue (device) on which the pipeline will be created.
 * @param meshletIR Mesh shader source (Slang; see shaders.h). May be the same module as \p fragmentIR, since entry points are found by stage.
 * @param fragmentIR Pixel shader source (Slang; see shaders.h).
 * @param desc Rasterizer, format, and optional embedded blend state.
 * @return The pipeline, or NULL if it could not be created (including because the device has no mesh shaders).
 */
GpuPipeline* gpuCreateGraphicsMeshletPipeline(GpuQueue* queue, GpuByteSpan meshletIR, GpuByteSpan fragmentIR, NOAPI_CONST_REF(GpuRasterDesc) desc);

// ---------------------------------------------------------------------------
// Separate state objects
// ---------------------------------------------------------------------------

/**
 * gpuCreateDepthStencilState – Bake a depth/stencil configuration into a reusable state object. Applied per-command-buffer via gpuSetDepthStencilState before draw calls. Separating depth/stencil state from the PSO (unlike DX12) reduces permutations and avoids costly PSO recompiles when only depth bias or the stencil reference changes.
 *
 * @code{.cpp}
 * auto state = gpuCreateDepthStencilState(queue, {.depthMode = DEPTH_FLAGS(DEPTH_READ | DEPTH_WRITE), .depthTest = OP_LESS_EQUAL});
 * assert(state);
 * gpuFreeDepthStencilState(queue, state);
 * @endcode
 *
 * @param queue The GPU queue (device) on which the state object will be created.
 * @param desc Depth/stencil test and write configuration.
 * @return The state object, or NULL if it could not be allocated.
 */
GpuDepthStencilState* gpuCreateDepthStencilState(GpuQueue* queue, NOAPI_CONST_REF(GpuDepthStencilDesc) desc);

/**
 * gpuCreateBlendState – Bake a blend configuration into a reusable state object. Applied per-command-buffer via gpuSetBlendState. On mobile TBDR GPUs, prefer embedding the blend state in the PSO or using framebuffer-fetch intrinsics for a parametrised blend formula.
 *
 * @code{.cpp}
 * auto additive = gpuCreateBlendState(queue, {.srcColorFactor = FACTOR_ONE, .dstColorFactor = FACTOR_ONE});
 * assert(additive);
 * gpuFreeBlendState(queue, additive);
 * @endcode
 *
 * @param queue The GPU queue (device) on which the state object will be created.
 * @param desc Blend equation and factor configuration.
 * @return The state object, or NULL if it could not be allocated.
 */
GpuBlendState* gpuCreateBlendState(GpuQueue* queue, NOAPI_CONST_REF(GpuBlendDesc) desc);

/**
 * gpuFreeDepthStencilState – Release a GpuDepthStencilState object. Command buffers recorded with it may still be in flight.
 *
 * @code
 * GpuDepthStencilDesc desc = GPU_DEPTH_STENCIL_DESC_DEFAULT;
 * desc.depthMode = DEPTH_READ;
 * GpuDepthStencilState* state = gpuCreateDepthStencilState(queue, NOAPI_REF(desc));
 * gpuFreeDepthStencilState(queue, state);
 * @endcode
 *
 * @param queue The GPU queue (device) the state object was created on.
 * @param state Depth/stencil state object to release.
 */
void gpuFreeDepthStencilState(GpuQueue* queue, GpuDepthStencilState* state);

/**
 * gpuFreeBlendState – Release a GpuBlendState object. Command buffers recorded with it may still be in flight.
 *
 * @code
 * GpuBlendDesc desc = GPU_BLEND_DESC_DEFAULT;
 * desc.colorOp = BLEND_MAX;
 * GpuBlendState* state = gpuCreateBlendState(queue, NOAPI_REF(desc));
 * gpuFreeBlendState(queue, state);
 * @endcode
 *
 * @param queue The GPU queue (device) the state object was created on.
 * @param state Blend state object to release.
 */
void gpuFreeBlendState(GpuQueue* queue, GpuBlendState* state);

// ---------------------------------------------------------------------------
// GPU commands – state binding
// ---------------------------------------------------------------------------

/**
 * gpuSetDepthStencilState – Apply a pre-baked GpuDepthStencilState for subsequent draw calls. Emits a small hardware command packet to configure the depth/stencil unit without touching the shader microcode or requiring a PSO recompile.
 *
 * @code{.cpp}
 * auto shader = string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0.5, 1); }
 * 	[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
 * )slang");
 * GpuTextureDesc color_desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
 * GpuTextureDesc depth_desc = {.dimensions = {64, 64, 1}, .format = FORMAT_D32_FLOAT, .usage = USAGE_DEPTH_STENCIL_ATTACHMENT};
 * auto color_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, color_desc));
 * auto depth_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, depth_desc));
 * auto color = gpuCreateTexture(queue, color_desc, color_memory);
 * auto depth = gpuCreateTexture(queue, depth_desc, depth_memory);
 * GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
 * auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.depthFormat = FORMAT_D32_FLOAT, .colorTargets = {&target, 1}});
 * auto closer = gpuCreateDepthStencilState(queue, {.depthMode = DEPTH_READ, .depthTest = OP_LESS}); // 0.5 is not closer than the 0.25 cleared
 * auto indices = gpuMalloc<uint32_t>(queue, 3);
 * for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
 * auto indices_gpu = gpuHostToDevicePointer(queue, indices);
 * auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
 * auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
 *
 * GpuColorAttachment color_attachment = {.texture = color, .loadOp = LOAD_OP_CLEAR};
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, indices_gpu);
 * gpuBeginRenderPass(cmd, {.depthAttachment = GpuDepthStencilAttachment{.texture = depth, .loadOp = LOAD_OP_CLEAR, .clearValue = 0.25}, .colorAttachments = {&color_attachment, 1}});
 * gpuSetPipeline(cmd, pipeline);
 * gpuSetDepthStencilState(cmd, closer);
 * gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
 * gpuEndRenderPass(cmd);
 * gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
 * gpuCopyFromTexture(cmd, pixels_gpu, color_memory, color);
 * gpuSyncMemoryEXT(cmd, pixels_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 * assert(pixels[0] == 0xff000000); // Still the clear color: the triangle failed the depth test
 *
 * gpuFreeDepthStencilState(queue, closer);
 * gpuFreePipeline(queue, pipeline);
 * gpuFree(queue, pixels);
 * gpuFree(queue, indices);
 * gpuFree(queue, depth_memory);
 * gpuFree(queue, color_memory);
 * @endcode
 *
 * @note On WebGPU the depth/stencil state is part of the pipeline, so the next draw may compile a new variant of the bound one.
 *
 * @param cmd Command buffer to record into.
 * @param state Depth/stencil state object to apply, created by gpuCreateDepthStencilState.
 */
void gpuSetDepthStencilState(GpuCommandBuffer* cmd, const GpuDepthStencilState* state);

/**
 * gpuSetBlendState – Apply a pre-baked GpuBlendState for subsequent draw calls, overriding the blend state the bound pipeline was created with for as long as it is set.
 *
 * @code{.cpp}
 * auto shader = string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
 * 	[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 0, 0, 1); }
 * )slang");
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 * GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
 * auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
 * auto additive = gpuCreateBlendState(queue, {.srcColorFactor = FACTOR_ONE, .dstColorFactor = FACTOR_ONE});
 * auto indices = gpuMalloc<uint32_t>(queue, 3);
 * for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
 * auto indices_gpu = gpuHostToDevicePointer(queue, indices);
 * auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
 * auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
 *
 * GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR, .clearValue = {0, 0, 1, 1}};
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, indices_gpu);
 * gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
 * gpuSetPipeline(cmd, pipeline);
 * gpuSetBlendState(cmd, additive);
 * gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
 * gpuEndRenderPass(cmd);
 * gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
 * gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
 * gpuSyncMemoryEXT(cmd, pixels_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 * assert(pixels[0] == 0xffff00ff); // Red added onto the blue it was cleared to
 *
 * gpuFreeBlendState(queue, additive);
 * gpuFreePipeline(queue, pipeline);
 * gpuFree(queue, pixels);
 * gpuFree(queue, indices);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @note On WebGPU the blend state is part of the pipeline, so the next draw may compile a new variant of the bound one.
 *
 * @param cmd Command buffer to record into.
 * @param state Blend state object to apply, created by gpuCreateBlendState.
 */
void gpuSetBlendState(GpuCommandBuffer* cmd, const GpuBlendState* state);

/**
 * gpuSetViewportEXT – Set the viewport transform used to map clip-space coordinates onto the render target.
 *
 * @code{.cpp}
 * auto shader = string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
 * 	[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
 * )slang");
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 * GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
 * auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
 * auto indices = gpuMalloc<uint32_t>(queue, 3);
 * for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
 * auto indices_gpu = gpuHostToDevicePointer(queue, indices);
 * auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
 * auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
 *
 * GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, indices_gpu);
 * gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
 * gpuSetViewportEXT(cmd, {32, 64}); // The left half
 * gpuSetPipeline(cmd, pipeline);
 * gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
 * gpuEndRenderPass(cmd);
 * gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
 * gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
 * gpuSyncMemoryEXT(cmd, pixels_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 * assert(pixels[0] == 0xffffffff && pixels[63] == 0xff000000);
 *
 * gpuFreePipeline(queue, pipeline);
 * gpuFree(queue, pixels);
 * gpuFree(queue, indices);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @note Set automatically by gpuBeginRenderPass to cover the full render target; call this afterward to override it.
 *
 * @param cmd Command buffer to record into.
 * @param extent Width and height of the viewport in pixels.
 * @param origin Top-left corner of the viewport in pixels (default: {0, 0}).
 * @param depth_min Minimum depth range value (default: 0).
 * @param depth_max Maximum depth range value (default: 1).
 */
void gpuSetViewportEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin NOAPI_DEFAULT({0, 0}), float depth_min NOAPI_DEFAULT(0), float depth_max NOAPI_DEFAULT(1));

/**
 * gpuSetScissorRectEXT – Set the scissor rectangle that clips rasterizer output.
 *
 * @code{.cpp}
 * auto shader = string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
 * 	[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
 * )slang");
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 * GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
 * auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
 * auto indices = gpuMalloc<uint32_t>(queue, 3);
 * for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
 * auto indices_gpu = gpuHostToDevicePointer(queue, indices);
 * auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
 * auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
 *
 * GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, indices_gpu);
 * gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
 * gpuSetScissorRectEXT(cmd, {32, 64}, {32, 0}); // Only the right half is drawn
 * gpuSetPipeline(cmd, pipeline);
 * gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
 * gpuEndRenderPass(cmd);
 * gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
 * gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
 * gpuSyncMemoryEXT(cmd, pixels_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 * assert(pixels[0] == 0xff000000 && pixels[63] == 0xffffffff);
 *
 * gpuFreePipeline(queue, pipeline);
 * gpuFree(queue, pixels);
 * gpuFree(queue, indices);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @note Set automatically by gpuBeginRenderPass to cover the full render target; call this afterward to override it.
 *
 * @param cmd Command buffer to record into.
 * @param extent Width and height of the scissor rectangle in pixels.
 * @param origin Top-left corner of the scissor rectangle in pixels (default: {0, 0}).
 */
void gpuSetScissorRectEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin NOAPI_DEFAULT({0, 0}));

// ---------------------------------------------------------------------------
// GPU commands – render passes
// ---------------------------------------------------------------------------

/**
 * gpuBeginRenderPass – Configure the rasterizer for a set of render targets and begin recording rasterization commands.
 *
 * Performs the hardware-specific render-target setup and (on TBDR GPUs) triggers the tile load operations described by the attachment load ops. Fast-clear elimination is handled transparently by the driver when the clear color changes.
 *
 * Does NOT insert an automatic barrier before the pass. If a previous pass wrote to any of these targets via compute shaders, the user must call gpuBarrier with appropriate HAZARD_DEPTH_STENCIL or HAZARD_DESCRIPTORS flags beforehand.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 * auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
 * auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
 *
 * GpuColorAttachment clear = {.texture = texture, .loadOp = LOAD_OP_CLEAR, .clearValue = {1, 0, 0, 1}};
 * GpuColorAttachment keep = {.texture = texture, .loadOp = LOAD_OP_LOAD};
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuBeginRenderPass(cmd, {.colorAttachments = {&clear, 1}});
 * gpuEndRenderPass(cmd);
 * gpuBeginRenderPass(cmd, {.colorAttachments = {&keep, 1}}); // Picks up where the last pass left off
 * gpuEndRenderPass(cmd);
 * gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
 * gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
 * gpuSyncMemoryEXT(cmd, pixels_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 * assert(pixels[0] == 0xff0000ff);
 *
 * gpuFree(queue, pixels);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @param cmd Command buffer to record into.
 * @param desc Render target attachments and their load/store behavior.
 */
void gpuBeginRenderPass(GpuCommandBuffer* cmd, NOAPI_CONST_REF(GpuRenderPassDesc) desc);

/**
 * gpuEndRenderPass – End the current render pass and trigger (on TBDR GPUs) tile store operations for attachments whose storeOp is STORE_OP_STORE.
 *
 * Does NOT insert an automatic barrier after the pass. If subsequent passes need to read these render targets as textures, the user must call gpuBarrier with STAGE_RASTER_COLOR_OUT / STAGE_RASTER_DEPTH_OUT as the producer stage.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_COLOR_ATTACHMENT};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 *
 * GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
 * GpuRenderPassDesc pass = {.colorAttachments = {&attachment, 1}};
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuBeginRenderPass(cmd, pass);
 * gpuEndRenderPass(cmd, pass); // As a surface texture would be ended, ready to present
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 *
 * gpuFree(queue, memory);
 * @endcode
 *
 * @param cmd The command buffer to bind against.
 * @param desc Optionally, the same render pass descriptor that was passed to gpuBeginRenderPass, which lets the backend leave the attachments in a layout better suited to presentation.
 */
void gpuEndRenderPass(GpuCommandBuffer* cmd, GpuOptionalRenderPassDesc desc NOAPI_DEFAULT({}));

// ---------------------------------------------------------------------------
// GPU commands – rasterizer draw calls
// ---------------------------------------------------------------------------

/**
 * gpuDrawIndexedInstanced – Draw indexed geometry with instancing.
 *
 * The vertex shader receives `vertexData` as its root data pointer and the pixel shader receives `fragmentData`. Passing the same pointer for both is valid if the two shaders share their input struct. Vertex buffers are replaced by raw pointer loads from the vertex data struct; only the index buffer is special (hardware index deduplication still provides a 4-6x vertex shading reduction in practice).
 *
 * @code{.cpp}
 * auto shader = string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
 * 	[shader("fragment")] float4 fragmentMain() : SV_Target { return gpuFragmentData<float4>()[0]; }
 * )slang");
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 * GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
 * auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
 *
 * struct Color { float r, g, b, a; };
 * auto color = gpuMalloc<Color>(queue);
 * *color = {1, 0, 1, 1};
 * auto indices = gpuMalloc<uint16_t>(queue, 4);
 * for(uint16_t i = 0; i < 3; ++i) indices[i] = i;
 * auto color_gpu = gpuHostToDevicePointer(queue, color);
 * auto indices_gpu = gpuHostToDevicePointer(queue, indices);
 * auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
 * auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
 *
 * GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, color_gpu);
 * gpuSyncMemoryEXT(cmd, indices_gpu);
 * gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
 * gpuSetPipeline(cmd, pipeline);
 * gpuDrawIndexedInstanced(cmd, color_gpu, color_gpu, indices_gpu, 3, 1, INDEX_TYPE_UINT16);
 * gpuEndRenderPass(cmd);
 * gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
 * gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
 * gpuSyncMemoryEXT(cmd, pixels_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 * assert(pixels[0] == 0xffff00ff);
 *
 * gpuFreePipeline(queue, pipeline);
 * gpuFree(queue, pixels);
 * gpuFree(queue, indices);
 * gpuFree(queue, color);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @param cmd Command buffer to record into.
 * @param vertex_data GPU pointer to the vertex shader root data struct.
 * @param fragment_data GPU pointer to the pixel shader root data struct.
 * @param indices GPU pointer to the index buffer, whose entries are \p index_type wide.
 * @param index_count Number of indices to draw.
 * @param instance_count Number of instances (1 for non-instanced geometry).
 * @param index_type Width of each index.
 */
void gpuDrawIndexedInstanced(GpuCommandBuffer* cmd,
	gpu* vertex_data, gpu* fragment_data,
	gpu* indices, uint32_t index_count, uint32_t instance_count,
	INDEX_TYPE_EXT index_type NOAPI_DEFAULT(INDEX_TYPE_UINT32));

/**
 * gpuDrawIndexedInstancedIndirect – GPU-driven indexed instanced draw. Reads the draw arguments (index count, instance count, etc.) from GPU memory, enabling the GPU to cull and compact draw lists without CPU round-trips.
 *
 * Both the root data pointers and the draw arguments are GPU pointers, making this a fully indirect draw — an improvement over existing APIs (DX12 ExecuteIndirect, Vulkan vkCmdDrawIndexedIndirect) which require the CPU to supply the root data.
 *
 * @code{.cpp}
 * auto shader = string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
 * 	[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(0, 0, 1, 1); }
 * )slang");
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 * GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
 * auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
 *
 * struct DrawArguments { uint32_t index_count, instance_count, first_index; int32_t vertex_offset; uint32_t first_instance; };
 * auto arguments = gpuMalloc<DrawArguments>(queue);
 * *arguments = {3, 1, 0, 0, 0};
 * auto indices = gpuMalloc<uint32_t>(queue, 3);
 * for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
 * auto arguments_gpu = gpuHostToDevicePointer(queue, arguments);
 * auto indices_gpu = gpuHostToDevicePointer(queue, indices);
 * auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
 * auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
 *
 * GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, arguments_gpu);
 * gpuSyncMemoryEXT(cmd, indices_gpu);
 * gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
 * gpuSetPipeline(cmd, pipeline);
 * gpuDrawIndexedInstancedIndirect(cmd, indices_gpu, indices_gpu, indices_gpu, arguments_gpu);
 * gpuEndRenderPass(cmd);
 * gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
 * gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
 * gpuSyncMemoryEXT(cmd, pixels_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 * assert(pixels[0] == 0xffff0000);
 *
 * gpuFreePipeline(queue, pipeline);
 * gpuFree(queue, pixels);
 * gpuFree(queue, indices);
 * gpuFree(queue, arguments);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @note Every argument struct between \p args and the end of its allocation is drawn. On Vulkan more than one needs the device's multiDrawIndirect feature.
 *
 * @param cmd Command buffer to record into.
 * @param vertex_data GPU pointer to the vertex shader root data struct.
 * @param fragment_data GPU pointer to the pixel shader root data struct.
 * @param indices GPU pointer to the index buffer.
 * @param args GPU pointer to the draw arguments: index count, instance count and first index as uint32_t, then an int32_t vertex offset and a uint32_t first instance.
 * @param index_type Width of each index.
 */
void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer* cmd,
	gpu* vertex_data, gpu* fragment_data,
	gpu* indices, gpu* args,
	INDEX_TYPE_EXT index_type NOAPI_DEFAULT(INDEX_TYPE_UINT32));

/**
 * gpuDrawMeshlets – Launch a mesh shader pass, dispatching a 3D grid of mesh shader thread groups. Each group outputs a self-contained meshlet (up to 256 vertices, 128 triangles on AMD; 126 vertices, 64 triangles on Nvidia). Offline vertex deduplication eliminates the need for an index deduplication unit.
 *
 * Requires a pipeline from gpuCreateGraphicsMeshletPipeline to be bound.
 *
 * @code{.cpp}
 * auto shader = string_to_bytes(R"slang(
 * 	import noapi;
 * 	struct Vertex { float4 position : SV_Position; }
 * 	[shader("mesh")] [outputtopology("triangle")] [numthreads(1, 1, 1)]
 * 	void meshMain(out vertices Vertex corners[3], out indices uint3 triangles[1]) {
 * 		SetMeshOutputCounts(3, 1);
 * 		corners[0].position = float4(-1, -1, 0, 1); corners[1].position = float4(3, -1, 0, 1); corners[2].position = float4(-1, 3, 0, 1);
 * 		triangles[0] = uint3(0, 1, 2);
 * 	}
 * 	[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
 * )slang");
 * GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
 * auto pipeline = gpuCreateGraphicsMeshletPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
 * if(pipeline) {
 * 	GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
 * 	auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * 	auto texture = gpuCreateTexture(queue, desc, memory);
 * 	auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
 * 	auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
 *
 * 	GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
 * 	auto cmd = gpuStartCommandRecording(queue);
 * 	gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
 * 	gpuSetPipeline(cmd, pipeline);
 * 	gpuDrawMeshlets(cmd, pixels_gpu, pixels_gpu, {1, 1, 1});
 * 	gpuEndRenderPass(cmd);
 * 	gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
 * 	gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
 * 	gpuSyncMemoryEXT(cmd, pixels_gpu);
 * 	gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 * 	assert(pixels[0] == 0xffffffff);
 *
 * 	gpuFreePipeline(queue, pipeline);
 * 	gpuFree(queue, pixels);
 * 	gpuFree(queue, memory);
 * }
 * @endcode
 *
 * @note Optional, alongside the pipeline it draws with: where gpuGetCapabilitiesEXT().mesh_shaders is false this reports through the diagnostic callback and draws nothing. WebGPU has no mesh shaders at all.
 *
 * @param cmd Command buffer to record into.
 * @param meshlet_data GPU pointer to the mesh shader root data struct.
 * @param fragment_data GPU pointer to the pixel shader root data struct.
 * @param dim Thread group grid dimensions (typically x = meshlet count).
 */
void gpuDrawMeshlets(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, uvec3 dim);

/**
 * gpuDrawMeshletsIndirect – GPU-driven mesh shader dispatch. Reads the thread group grid dimensions from GPU memory, enabling the GPU to cull and compact meshlet lists without CPU round-trips.
 *
 * Requires a pipeline from gpuCreateGraphicsMeshletPipeline to be bound. Every uvec3 between \p dim and the end of its allocation is dispatched, matching gpuDrawIndexedInstancedIndirect.
 *
 * @code{.cpp}
 * auto shader = string_to_bytes(R"slang(
 * 	import noapi;
 * 	struct Vertex { float4 position : SV_Position; }
 * 	[shader("mesh")] [outputtopology("triangle")] [numthreads(1, 1, 1)]
 * 	void meshMain(out vertices Vertex corners[3], out indices uint3 triangles[1]) {
 * 		SetMeshOutputCounts(3, 1);
 * 		corners[0].position = float4(-1, -1, 0, 1); corners[1].position = float4(3, -1, 0, 1); corners[2].position = float4(-1, 3, 0, 1);
 * 		triangles[0] = uint3(0, 1, 2);
 * 	}
 * 	[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
 * )slang");
 * GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
 * auto pipeline = gpuCreateGraphicsMeshletPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
 * if(pipeline) {
 * 	GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
 * 	auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * 	auto texture = gpuCreateTexture(queue, desc, memory);
 * 	auto grid = gpuMalloc<uvec3>(queue);
 * 	*grid = {1, 1, 1};
 * 	auto grid_gpu = gpuHostToDevicePointer(queue, grid);
 * 	auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
 * 	auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
 *
 * 	GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
 * 	auto cmd = gpuStartCommandRecording(queue);
 * 	gpuSyncMemoryEXT(cmd, grid_gpu);
 * 	gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
 * 	gpuSetPipeline(cmd, pipeline);
 * 	gpuDrawMeshletsIndirect(cmd, grid_gpu, grid_gpu, grid_gpu);
 * 	gpuEndRenderPass(cmd);
 * 	gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
 * 	gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
 * 	gpuSyncMemoryEXT(cmd, pixels_gpu);
 * 	gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 * 	assert(pixels[0] == 0xffffffff);
 *
 * 	gpuFreePipeline(queue, pipeline);
 * 	gpuFree(queue, pixels);
 * 	gpuFree(queue, grid);
 * 	gpuFree(queue, memory);
 * }
 * @endcode
 *
 * @note Optional, alongside the pipeline it draws with: where gpuGetCapabilitiesEXT().mesh_shaders is false this reports through the diagnostic callback and draws nothing. WebGPU has no mesh shaders at all.
 *
 * @param cmd Command buffer to record into.
 * @param meshlet_data GPU pointer to the mesh shader root data struct.
 * @param fragment_data GPU pointer to the pixel shader root data struct.
 * @param dim GPU pointer to a uvec3 containing group dimensions.
 */
void gpuDrawMeshletsIndirect(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, gpu* dim);

NOAPI_EXTERN_C_END
