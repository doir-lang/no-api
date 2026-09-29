// A photograph on a rectangle: what it takes to get a texture from a file onto the screen.
//
// Compared with 0_basic's triangle, everything new here is on the texture side:
//
// - gpuTextureSizeAlign + gpuMalloc + gpuCreateTexture, which is the three step allocation the API
//   uses instead of creating an image and then asking it what memory it wants
// - gpuCopyToTexture out of a staging allocation, and gpuBlitTextureEXT to build the mip chain it
//   cannot write (see upload_texture in common.hpp)
// - the texture heap: an ordinary gpuMalloc'd array of 512 bit descriptor blobs, made active for a
//   command buffer with gpuSetActiveTextureHeapPtr, which shaders index by a plain uint. There is
//   no other binding call for a texture anywhere in this file.
// - gpuSetEnabledSamplersEXT, the fixed set of samplers a shader picks between by describing the
//   one it wants rather than by binding a sampler object
//
// The one thing the rectangle has to do for itself is pick its mip level. Nothing in this API
// samples with implicit derivatives -- gpuSample takes the level as an argument -- so the fragment
// shader works out how many texels its pixel covers and asks for that level, which is what keeps
// the photo from aliasing as the window shrinks.

#include "common.hpp"

#include "stb_image.h"

using namespace test;

// The photo is resampled to this before it is uploaded; see resample in common.hpp for why it is a
// power of two, and why it is nowhere near the 6160x4640 the file actually holds.
static constexpr uint32_t TEXTURE_SIZE = 1024;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

static const std::string slang_rectangle = R"slang(
import noapi;

// One entry of the vertex array, matching Vertex on the C++ side: four floats, {x, y, u, v}
struct Vertex : IGpuLoadable {
	float2 position;
	float2 uv;

	static const uint gpuStride = 16;

	static Vertex gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		Vertex out;
		out.position = from.next<float2>();
		out.uv = from.next<float2>();
		return out;
	}
}

// The root data struct both stages are handed, matching RectData on the C++ side. `albedo` is the
// whole of what binds the texture: an index into the array gpuSetActiveTextureHeapPtr made active.
struct RectData : IGpuLoadable {
	GpuPtr<Vertex> vertices;
	float2 fit; // Scales the unit rectangle so the photo keeps its aspect ratio in the window
	float2 textureSize; // The albedo texture's size in texels, which mip selection needs
	uint albedo;

	// 8 + 8 + 8 + 4, rounded up to the 8 the pointer field aligns the host struct to
	static const uint gpuStride = 32;

	static RectData gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		RectData out;
		out.vertices = from.next<GpuPtr<Vertex> >();
		out.fit = from.next<float2>();
		out.textureSize = from.next<float2>();
		out.albedo = from.next<uint>();
		return out;
	}
}

struct Varyings {
	float4 position : SV_Position;
	float2 uv : TEXCOORD0;
}

// The sampler the host enabled with gpuSetEnabledSamplersEXT, described rather than bound. Clamped
// instead of repeated so the bilinear filter along the photo's border doesn't reach around to the
// opposite edge, which is visible as a one pixel seam once the rectangle is letterboxed.
GpuSamplerDesc albedoSampler() {
	return GpuSamplerDesc(ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP,
		FILTER_LINEAR, FILTER_LINEAR, FILTER_LINEAR);
}

/**
 * The mip level a pixel covering `size * ddx(uv)` by `size * ddy(uv)` texels wants: the log2 of the
 * longer of the two footprints, which is the level where one texel matches one pixel.
 *
 * A conventional API would do this inside Sample. Here the only sampling entry point takes the
 * level explicitly (gpuSample's `mip`), so it is written out -- and being written out, it is also
 * where a shader would bias or clamp it.
 */
float mipForFootprint(float2 uv, float2 size) {
	let dx = ddx(uv) * size;
	let dy = ddy(uv) * size;
	return 0.5 * log2(max(dot(dx, dx), dot(dy, dy)));
}

[shader("vertex")]
Varyings vertexMain(uint index : SV_VertexID) {
	let data = *gpuVertexData<RectData>();
	let vertex = data.vertices[index];

	Varyings output;
	output.position = float4(vertex.position * data.fit, 0.0, 1.0);
	output.uv = vertex.uv;
	return output;
}

[shader("fragment")]
float4 fragmentMain(Varyings varyings) : SV_Target {
	let data = *gpuFragmentData<RectData>();
	let mip = mipForFootprint(varyings.uv, data.textureSize);
	// Layer 0: the view in the heap entry describes a plain 2D texture, so there is only one
	return gpuSample(data.albedo, gpuGetSamplerIndex(albedoSampler()), float3(varyings.uv, 0.0), 0u, mip);
}
)slang";

// ---------------------------------------------------------------------------
// Host side layouts
// ---------------------------------------------------------------------------

struct Vertex {
	float x, y;
	float u, v;
};

// Keep in sync with RectData in the shader above, including its stride
struct RectData {
	gpu* vertices;
	float fit[2];
	float texture_size[2];
	uint32_t albedo;
	uint32_t _padding;
};
static_assert(sizeof(RectData) == 32, "RectData::gpuStride in the shader has to match");

// The sampler the shader's albedoSampler() describes. gpuGetSamplerIndex matches them by the
// packed value, so the two only have to agree field for field, not slot for slot.
static constexpr GpuSamplerDesc ALBEDO_SAMPLER {
	.address_mode_u = ADDRESS_MODE_CLAMP,
	.address_mode_v = ADDRESS_MODE_CLAMP,
	.address_mode_w = ADDRESS_MODE_CLAMP,
};

// ---------------------------------------------------------------------------

/**
 * load_photo -- Read the jpeg, resample it to TEXTURE_SIZE square, and report the aspect ratio it
 * had before that so the rectangle can put it back.
 *
 * Squaring it rather than preserving the aspect keeps the texture a power of two in both axes,
 * which is what the rectangle then undoes; see resample in common.hpp.
 *
 * @param out_aspect Filled in with the original image's width over its height.
 */
static std::vector<uint8_t> load_photo(float& out_aspect) {
	constexpr auto path = TEST_ASSET_DIR "/pexels-70588695-35341664.jpg";

	int width, height, channels;
	auto pixels = stbi_load(path, &width, &height, &channels, 4);
	if(!pixels) throw std::runtime_error(std::string("Failed to load ") + path + ": " + stbi_failure_reason());

	out_aspect = float(width) / float(height);
	std::println("photo: {}x{} ({} channels), resampled to {}x{}", width, height, channels, TEXTURE_SIZE, TEXTURE_SIZE);

	auto resampled = resample<4>(std::span<const uint8_t>(pixels, size_t(width) * height * 4),
		uint32_t(width), uint32_t(height), TEXTURE_SIZE, TEXTURE_SIZE);
	stbi_image_free(pixels);
	return resampled;
}

TEST_MAIN({
	App app;
	open_window(app, "NoAPI Texture");
	create_device(app);

	//
	// The photo, as a mipped 2D texture. USAGE_COLOR_ATTACHMENT is there on top of USAGE_SAMPLED
	// because the mip chain is rasterized rather than copied -- gpuCopyToTexture only writes level
	// 0 -- so every level below the first is a render target for one blit. See upload_texture.
	//
	float photo_aspect = 1.0f;
	auto photo = load_photo(photo_aspect);

	auto albedo = create_texture(app.queue, GpuTextureDesc{
		.dimensions = {TEXTURE_SIZE, TEXTURE_SIZE, 1},
		.mipCount = mip_count_for(TEXTURE_SIZE, TEXTURE_SIZE),
		.format = FORMAT_RGBA8_UNORM,
		.usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_DST),
	});
	upload_texture(app.queue, albedo, std::as_bytes(std::span(photo)));
	std::println("albedo: {} mip levels", albedo.desc.mipCount);

	//
	// The texture heap. Just an array in gpu memory holding one 256 bit descriptor per texture; the
	// shader's `albedo` field is an index into it and nothing else binds anything.
	//
	auto heap = gpuMalloc<GpuTextureDescriptor>(app.queue, 1);
	heap[0] = gpuTextureViewDescriptor(app.queue, albedo.texture, GpuViewDesc{});
	auto heap_gpu = gpuHostToDevicePointer(app.queue, heap);
	gpuSyncMemoryEXT(app.queue, heap_gpu);

	//
	// Geometry. Two triangles over a unit rectangle, with v = 0 on the edge that ends up at the top
	// of the screen, because the upload put the image's first row at v = 0.
	//
	auto vertices = gpuMalloc<Vertex>(app.queue, 4);
	vertices[0] = {-1.0f,  1.0f, 0.0f, 0.0f};
	vertices[1] = { 1.0f,  1.0f, 1.0f, 0.0f};
	vertices[2] = { 1.0f, -1.0f, 1.0f, 1.0f};
	vertices[3] = {-1.0f, -1.0f, 0.0f, 1.0f};

	auto indices = gpuMalloc<uint32_t>(app.queue, 6);
	constexpr uint32_t quad[] = {0, 1, 2, 0, 2, 3};
	memcpy(indices, quad, sizeof(quad));

	auto data = gpuMalloc<RectData>(app.queue);
	*data = RectData{
		.vertices = gpuHostToDevicePointer(app.queue, vertices),
		.fit = {1.0f, 1.0f},
		.texture_size = {float(TEXTURE_SIZE), float(TEXTURE_SIZE)},
		.albedo = 0,
	};
	auto data_gpu = gpuHostToDevicePointer(app.queue, data);
	auto indices_gpu = gpuHostToDevicePointer(app.queue, indices);

	gpuSyncMemoryEXT(app.queue, data->vertices);
	gpuSyncMemoryEXT(app.queue, indices_gpu);

	//
	// Pipeline. One module supplies both stages, found by stage rather than by name.
	//
	GpuColorTarget target { .format = app.surface_format };
	auto pipeline = gpuCreateGraphicsPipeline(app.queue, string_to_bytes(slang_rectangle), string_to_bytes(slang_rectangle), GpuRasterDesc{
		.colorTargets = {&target, 1},
	});
	assert(pipeline && "Pipeline creation failed");

	run(app, [&](App& app, GpuCommandBuffer* cmd, const GpuTexture* backbuffer) {
		// Letterbox the photo: whichever axis the window has to spare is the one that shrinks
		auto window_aspect = app.aspect();
		if(window_aspect > photo_aspect) {
			data->fit[0] = photo_aspect / window_aspect;
			data->fit[1] = 1.0f;
		} else {
			data->fit[0] = 1.0f;
			data->fit[1] = window_aspect / photo_aspect;
		}
		// Both of these record a copy, so they have to happen before the render pass opens: the
		// heap is snapshotted out of the buffer it lives in, and gpuSyncMemoryEXT pushes the root
		// data the frame just rewrote
		gpuSyncMemoryEXT(cmd, data_gpu);
		gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
		gpuSetEnabledSamplersEXT(cmd, {&ALBEDO_SAMPLER, 1});

		GpuColorAttachment color {
			.texture = backbuffer,
			.loadOp = LOAD_OP_CLEAR,
			.clearValue = {0.02f, 0.02f, 0.03f, 1.0f},
		};
		GpuRenderPassDesc pass { .colorAttachments = {&color, 1} };

		gpuBeginRenderPass(cmd, pass);
		gpuSetPipeline(cmd, pipeline);
		gpuDrawIndexedInstanced(cmd, data_gpu, data_gpu, indices_gpu, 6, 1);
		gpuEndRenderPass(cmd, pass);
	}, [&](App& app) {
		gpuFreePipeline(app.queue, pipeline);
		gpuFree(app.queue, albedo.memory);
		gpuFree(app.queue, heap);
		gpuFree(app.queue, vertices);
		gpuFree(app.queue, indices);
		gpuFree(app.queue, data);
	});

	return 0;
})
