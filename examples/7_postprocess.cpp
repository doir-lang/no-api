#include "common.hpp"

using namespace test;

// Each pass is a fullscreen triangle whose fragment shader reads the textures its PassData names
static const std::string slang_common = R"slang(
import noapi;

struct PassData : IGpuLoadable {
	uint source;
	uint bloom;
	float time;
	float threshold;
	float2 inverseTarget;
	float2 step;

	static const uint gpuStride = 32;

	static PassData gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		PassData out;
		out.source = from.next<uint>();
		out.bloom = from.next<uint>();
		out.time = from.next<float>();
		out.threshold = from.next<float>();
		out.inverseTarget = from.next<float2>();
		out.step = from.next<float2>();
		return out;
	}
}

GpuSamplerDesc linearClamp() {
	return GpuSamplerDesc(ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, FILTER_LINEAR, FILTER_LINEAR, FILTER_LINEAR);
}

float3 sampleRGB(uint texture, float2 uv) {
	return gpuSample2D(texture, gpuGetSamplerIndex(linearClamp()), uv).rgb;
}

PassData passData() { return *gpuFragmentData<PassData>(); }

float2 targetUV(float4 position) { return position.xy * passData().inverseTarget; }

[shader("vertex")]
float4 vertexMain(uint index : SV_VertexID) : SV_Position {
	return float4(float2((index << 1) & 2, index & 2) * 2.0 - 1.0, 0.0, 1.0);
}
)slang";

// An HDR scene: orbs brighter than 1 over a dim gradient, so only they bloom
static const std::string slang_scene = slang_common + R"slang(
[shader("fragment")]
float4 fragmentMain(float4 position : SV_Position) : SV_Target {
	let data = passData();
	let uv = targetUV(position);
	let aspect = data.inverseTarget.y / data.inverseTarget.x;
	let p = float2((uv.x - 0.5) * aspect, uv.y - 0.5);

	float3 color = lerp(float3(0.02, 0.02, 0.05), float3(0.08, 0.04, 0.10), uv.y);
	for(uint i = 0; i < 6; ++i) {
		let phase = data.time * (0.3 + 0.07 * float(i)) + float(i) * 1.047;
		let center = float2(cos(phase), sin(phase * 1.3)) * float2(0.45, 0.28);
		let hue = 0.5 + 0.5 * cos(6.28318 * (float(i) / 6.0 + float3(0.0, 0.33, 0.67)));
		let d = length(p - center);
		color += hue * (smoothstep(0.045, 0.035, d) * 6.0 + 0.002 / (d * d + 0.002) * 0.3);
	}
	return float4(color, 1.0);
}
)slang";

// Keeps what is brighter than the threshold, at half resolution
static const std::string slang_bright = slang_common + R"slang(
[shader("fragment")]
float4 fragmentMain(float4 position : SV_Position) : SV_Target {
	let data = passData();
	let color = sampleRGB(data.source, targetUV(position));
	let brightness = max(color.r, max(color.g, color.b));
	return float4(color * max(brightness - data.threshold, 0.0) / max(brightness, 1e-4), 1.0);
}
)slang";

// One direction of a separable 9 tap Gaussian, `step` texels of the source apart
static const std::string slang_blur = slang_common + R"slang(
[shader("fragment")]
float4 fragmentMain(float4 position : SV_Position) : SV_Target {
	let data = passData();
	let uv = targetUV(position);
	static const float weights[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };
	float3 sum = sampleRGB(data.source, uv) * weights[0];
	for(uint i = 1; i < 5; ++i) {
		sum += sampleRGB(data.source, uv + data.step * float(i)) * weights[i];
		sum += sampleRGB(data.source, uv - data.step * float(i)) * weights[i];
	}
	return float4(sum, 1.0);
}
)slang";

// Scene plus bloom, tonemapped, with chromatic aberration and a vignette toward the edges
static const std::string slang_composite = slang_common + R"slang(
[shader("fragment")]
float4 fragmentMain(float4 position : SV_Position) : SV_Target {
	let data = passData();
	let uv = targetUV(position);
	let offset = (uv - 0.5) * 0.004;
	let scene = float3(sampleRGB(data.source, uv + offset).r, sampleRGB(data.source, uv).g, sampleRGB(data.source, uv - offset).b);
	var color = scene + sampleRGB(data.bloom, uv) * 1.2;
	color = saturate((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14));
	color *= 1.0 - 0.6 * dot(uv - 0.5, uv - 0.5);
	return float4(pow(color, float3(1.0 / 2.2)), 1.0);
}
)slang";


struct PassData {
	uint32_t source;
	uint32_t bloom;
	float time;
	float threshold;
	std::array<float, 2> inverse_target;
	std::array<float, 2> step;
};
static_assert(sizeof(PassData) == 32, "PassData::gpuStride in the shader has to match");

static constexpr GpuSamplerDesc LINEAR_CLAMP {
	.address_mode_u = ADDRESS_MODE_CLAMP,
	.address_mode_v = ADDRESS_MODE_CLAMP,
	.address_mode_w = ADDRESS_MODE_CLAMP,
};

// Heap slots: the scene, then the two half resolution targets the blur ping-pongs between
enum : uint32_t { SCENE, BLOOM_A, BLOOM_B, TARGET_COUNT };

static constexpr uint32_t BLUR_ROUNDS = 3;
// Scene, bright pass, two blur directions per round, composite
static constexpr uint32_t PASS_COUNT = 3 + 2 * BLUR_ROUNDS;

struct Targets {
	std::array<Texture, TARGET_COUNT> textures;
	uint32_t width = 0, height = 0;
};

// Sampleable and attachable both, which is what lets one pass render into a target the next one samples
static void create_targets(App& app, Targets& targets, GpuTextureDescriptor* heap) {
	if(targets.width == app.width && targets.height == app.height) return;
	if(targets.width) {
		gpuWaitIdleEXT(app.queue);
		for(auto& target: targets.textures) gpuFree(app.queue, target.memory);
	}

	targets.width = app.width;
	targets.height = app.height;
	for(uint32_t i = 0; i < TARGET_COUNT; ++i) {
		auto full = i == SCENE;
		targets.textures[i] = create_texture(app.queue, GpuTextureDesc{
			.dimensions = {full ? app.width : std::max(app.width / 2, 1u), full ? app.height : std::max(app.height / 2, 1u), 1},
			.format = FORMAT_RGBA16_FLOAT,
			.usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_COLOR_ATTACHMENT),
		});
		heap[i] = gpuTextureViewDescriptor(app.queue, targets.textures[i].texture, GpuViewDesc{});
	}
	std::println("targets: {}x{} scene, {}x{} bloom", app.width, app.height, targets.textures[BLOOM_A].desc.dimensions.x, targets.textures[BLOOM_A].desc.dimensions.y);
}

TEST_MAIN({
	App app;
	open_window(app, "NoAPI Post-processing");
	create_device(app);

	GpuColorTarget hdr { .format = FORMAT_RGBA16_FLOAT };
	GpuColorTarget surface { .format = app.surface_format };
	auto pipeline = [&](const std::string& source, GpuColorTarget& target) {
		auto bytes = string_to_bytes(source);
		auto out = gpuCreateGraphicsPipeline(app.queue, bytes, bytes, GpuRasterDesc{ .colorTargets = {&target, 1} });
		assert(out && "Pipeline creation failed");
		return out;
	};
	auto scene_pipeline = pipeline(slang_scene, hdr);
	auto bright_pipeline = pipeline(slang_bright, hdr);
	auto blur_pipeline = pipeline(slang_blur, hdr);
	auto composite_pipeline = pipeline(slang_composite, surface);

	auto heap = gpuMalloc<GpuTextureDescriptor>(app.queue, TARGET_COUNT, MEMORY_DESCRIPTOR_HEAP);
	auto heap_gpu = gpuHostToDevicePointer(app.queue, heap);
	Targets targets;

	// One slot per pass, since they are all recorded before any runs
	auto passes = gpuMalloc<PassData>(app.queue, PASS_COUNT);
	auto passes_gpu = gpuHostToDevicePointer(app.queue, passes);
	auto indices = gpuMalloc<uint32_t>(app.queue, 3);
	for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
	auto indices_gpu = gpuHostToDevicePointer(app.queue, indices);
	gpuSyncMemoryEXT(app.queue, indices_gpu);

	auto started = glfwGetTime();
	run(app, [&](App& app, GpuCommandBuffer* cmd, const GpuTexture* backbuffer) {
		create_targets(app, targets, heap);
		gpuSyncMemoryEXT(cmd, heap_gpu);
		gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
		gpuSetEnabledSamplersEXT(cmd, {&LINEAR_CLAMP, 1});

		struct Pass {
			const GpuTexture* target;
			std::array<uint32_t, 2> extent;
			GpuPipeline* pipeline;
			PassData data;
		};
		auto& textures = targets.textures;
		auto half = std::array{textures[BLOOM_A].desc.dimensions.x, textures[BLOOM_A].desc.dimensions.y};
		std::vector<Pass> list = {
			{textures[SCENE].texture, {app.width, app.height}, scene_pipeline, {}},
			{textures[BLOOM_A].texture, half, bright_pipeline, {.source = SCENE, .threshold = 1.0f}},
		};
		// Wider each round, so three rounds of nine taps reach far
		for(uint32_t round = 0; round < BLUR_ROUNDS; ++round) {
			auto spread = float(1u << round);
			list.push_back({textures[BLOOM_B].texture, half, blur_pipeline, {.source = BLOOM_A, .step = {spread / float(half[0]), 0.0f}}});
			list.push_back({textures[BLOOM_A].texture, half, blur_pipeline, {.source = BLOOM_B, .step = {0.0f, spread / float(half[1])}}});
		}
		list.push_back({backbuffer, {app.width, app.height}, composite_pipeline, {.source = SCENE, .bloom = BLOOM_A}});
		assert(list.size() == PASS_COUNT);

		// Written and synced before the passes are recorded, since on WebGPU the sync is a copy at this point in the command buffer
		auto time = float(glfwGetTime() - started);
		for(uint32_t i = 0; i < PASS_COUNT; ++i) {
			passes[i] = list[i].data;
			passes[i].time = time;
			passes[i].inverse_target = {1.0f / float(list[i].extent[0]), 1.0f / float(list[i].extent[1])};
		}
		gpuSyncMemoryEXT(cmd, passes_gpu);

		for(uint32_t i = 0; i < PASS_COUNT; ++i) {
			// Each pass reads what the last one wrote, and may overwrite what the one before it read
			gpuBarrier(cmd, STAGE(STAGE_RASTER_COLOR_OUT | STAGE_PIXEL_SHADER), STAGE(STAGE_PIXEL_SHADER | STAGE_RASTER_COLOR_OUT));
			GpuColorAttachment color { .texture = list[i].target, .loadOp = LOAD_OP_DONT_CARE };
			GpuRenderPassDesc desc { .colorAttachments = {&color, 1} };
			gpuBeginRenderPass(cmd, desc);
			gpuSetPipeline(cmd, list[i].pipeline);
			auto root = passes_gpu + i * sizeof(PassData);
			gpuDrawIndexedInstanced(cmd, root, root, indices_gpu, 3, 1);
			gpuEndRenderPass(cmd, desc);
		}
	}, [&](App& app) {
		for(auto pipeline: {scene_pipeline, bright_pipeline, blur_pipeline, composite_pipeline})
			gpuFreePipeline(app.queue, pipeline);
		for(auto& target: targets.textures)
			if(target.texture) gpuFree(app.queue, target.memory);
		gpuFree(app.queue, passes);
		gpuFree(app.queue, indices);
		gpuFree(app.queue, heap);
	});

	return 0;
})
