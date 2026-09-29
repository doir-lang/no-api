#include "common.hpp"

using namespace test;

static constexpr uint32_t TEXTURE_SIZE = 1024;

static constexpr uint32_t GENERATE_GROUP = 8;


static const std::string slang_generate = R"slang(
import noapi;

static const uint MAX_ITERATIONS = 256u;

struct GenerateData : IGpuLoadable {
	uint target;
	uint2 size;
	float2 center;
	float scale;

	static const uint gpuStride = 24;

	static GenerateData gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		GenerateData out;
		out.target = from.next<uint>();
		out.size = from.next<uint2>();
		out.center = from.next<float2>();
		out.scale = from.next<float>();
		return out;
	}
}

float escape(float2 c) {
	float2 z = float2(0.0);
	for(uint i = 0u; i < MAX_ITERATIONS; ++i) {
		z = float2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;

		let radius = dot(z, z);
		if(radius > 4.0) {
			let smoothed = float(i) + 1.0 - log2(0.5 * log2(radius));
			return smoothed / float(MAX_ITERATIONS);
		}
	}
	return 1.0;
}

float3 palette(float t) {
	return 0.5 + 0.5 * cos(6.28318530718 * (t + float3(0.0, 0.33, 0.67)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 thread : SV_DispatchThreadID) {
	let data = *gpuComputeData<GenerateData>();
	if(any(thread.xy >= data.size)) return;

	let uv = (float2(thread.xy) + 0.5) / float2(data.size);
	let c = data.center + (uv * 2.0 - 1.0) * data.scale;

	let t = escape(c);
	let colour = t >= 1.0 ? float3(0.0) : palette(pow(t, 0.35) * 3.0);

	gpuStoreTexture2D(data.target, thread.xy, float4(colour, 1.0));
}
)slang";

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


struct GenerateData {
	uint32_t target;
	std::array<uint32_t, 2> size;
	std::array<float, 2> center;
	float scale;
};
static_assert(sizeof(GenerateData) == 24, "GenerateData::gpuStride in the shader has to match");

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

static constexpr uint32_t ALBEDO_SAMPLED = 0;
static constexpr uint32_t ALBEDO_STORAGE = 1;


TEST_MAIN({
	App app;
	open_window(app, "NoAPI Compute Texture");
	create_device(app);

	auto albedo = create_texture(app.queue, GpuTextureDesc{
		.dimensions = {TEXTURE_SIZE, TEXTURE_SIZE, 1},
		.mipCount = mip_count_for(TEXTURE_SIZE, TEXTURE_SIZE),
		.format = FORMAT_RGBA8_UNORM,
		.usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_STORAGE | USAGE_COLOR_ATTACHMENT),
	});
	std::println("albedo: {}x{}, {} mip levels", TEXTURE_SIZE, TEXTURE_SIZE, albedo.desc.mipCount);

	auto heap = gpuMalloc<GpuTextureDescriptor>(app.queue, 2, MEMORY_DESCRIPTOR_HEAP);
	heap[ALBEDO_SAMPLED] = gpuTextureViewDescriptor(app.queue, albedo.texture, GpuViewDesc{});
	heap[ALBEDO_STORAGE] = gpuRWTextureViewDescriptor(app.queue, albedo.texture, GpuViewDesc{.mipCount = 1});
	auto heap_gpu = gpuHostToDevicePointer(app.queue, heap);
	gpuSyncMemoryEXT(app.queue, heap_gpu);

	auto generate_data = gpuMalloc<GenerateData>(app.queue);
	*generate_data = GenerateData{
		.target = ALBEDO_STORAGE,
		.size = {TEXTURE_SIZE, TEXTURE_SIZE},
		.center = {-0.6f, 0.0f},
		.scale = 1.4f,
	};
	auto generate_data_gpu = gpuHostToDevicePointer(app.queue, generate_data);
	gpuSyncMemoryEXT(app.queue, generate_data_gpu);

	auto generate = gpuCreateComputePipeline(app.queue, string_to_bytes(slang_generate));
	assert(generate && "Compute pipeline creation failed");

	{
		auto groups = (TEXTURE_SIZE + GENERATE_GROUP - 1) / GENERATE_GROUP;

		auto cmd = gpuStartCommandRecording(app.queue);
		gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
		gpuSetPipeline(cmd, generate);
		gpuDispatch(cmd, generate_data_gpu, {groups, groups, 1});
		gpuBarrier(cmd, STAGE_COMPUTE, STAGE_PIXEL_SHADER);
		blit_mip_chain(cmd, albedo);

		auto submission = gpuSubmit(app.queue, {&cmd, 1});
		gpuWaitSemaphore(app.queue, gpuGetSubmissionSemaphoreEXT(app.queue), submission);
	}

	gpuFreePipeline(app.queue, generate);
	gpuFree(app.queue, generate_data);

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
		.albedo = ALBEDO_SAMPLED,
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
		data->fit[0] = window_aspect > 1.0f ? 1.0f / window_aspect : 1.0f;
		data->fit[1] = window_aspect > 1.0f ? 1.0f : window_aspect;

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
