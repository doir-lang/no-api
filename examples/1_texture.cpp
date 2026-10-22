#include "common.hpp"

#include "stb_image.h"

using namespace test;

static constexpr uint32_t TEXTURE_SIZE = 1024;


static const std::string slang_rectangle = R"slang(
import noapi;

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

struct RectData : IGpuLoadable {
	GpuPtr<Vertex> vertices;
	float2 fit;
	float2 textureSize;
	uint albedo;

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

GpuSamplerDesc albedoSampler() {
	return GpuSamplerDesc(ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, FILTER_LINEAR, FILTER_LINEAR, FILTER_LINEAR);
}

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
	return gpuSample(data.albedo, gpuGetSamplerIndex(albedoSampler()), float3(varyings.uv, 0.0), 0u, mip);
}
)slang";


struct Vertex {
	float x, y;
	float u, v;
};

struct RectData {
	gpu* vertices;
	std::array<float, 2> fit;
	std::array<float, 2> texture_size;
	uint32_t albedo;
	uint32_t _padding;
};
static_assert(sizeof(RectData) == 32, "RectData::gpuStride in the shader has to match");

static constexpr GpuSamplerDesc ALBEDO_SAMPLER {
	.address_mode_u = ADDRESS_MODE_CLAMP,
	.address_mode_v = ADDRESS_MODE_CLAMP,
	.address_mode_w = ADDRESS_MODE_CLAMP,
};


static std::vector<uint8_t> load_photo(float& out_aspect) {
	constexpr auto path = EXAMPLE_ASSET_DIR "/pexels-70588695-35341664.jpg";

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

	auto heap = gpuMalloc<GpuTextureDescriptor>(app.queue, 1, MEMORY_DESCRIPTOR_HEAP);
	heap[0] = gpuTextureViewDescriptor(app.queue, albedo.texture, GpuViewDesc{});
	auto heap_gpu = gpuHostToDevicePointer(app.queue, heap);
	gpuSyncMemoryEXT(app.queue, heap_gpu);

	auto vertices = gpuMalloc<Vertex>(app.queue, 4);
	vertices[0] = {-1.0f,  1.0f, 0.0f, 0.0f};
	vertices[1] = { 1.0f,  1.0f, 1.0f, 0.0f};
	vertices[2] = { 1.0f, -1.0f, 1.0f, 1.0f};
	vertices[3] = {-1.0f, -1.0f, 0.0f, 1.0f};

	auto indices = gpuMalloc<uint32_t>(app.queue, 6);
	constexpr std::array<uint32_t, 6> quad = {0, 1, 2, 0, 2, 3};
	memcpy(indices, quad.data(), sizeof(quad));

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

	GpuColorTarget target { .format = app.surface_format };
	auto pipeline = gpuCreateGraphicsPipeline(app.queue, string_to_bytes(slang_rectangle), string_to_bytes(slang_rectangle), GpuRasterDesc{
		.colorTargets = {&target, 1},
	});
	assert(pipeline && "Pipeline creation failed");

	run(app, [&](App& app, GpuCommandBuffer* cmd, const GpuTexture* backbuffer) {
		auto window_aspect = app.aspect();
		if(window_aspect > photo_aspect) {
			data->fit[0] = photo_aspect / window_aspect;
			data->fit[1] = 1.0f;
		} else {
			data->fit[0] = 1.0f;
			data->fit[1] = window_aspect / photo_aspect;
		}
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
