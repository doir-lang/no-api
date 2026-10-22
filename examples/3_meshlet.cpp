#include "common.hpp"

#include <cmath>
#include <numbers>

using namespace test;

static constexpr uint32_t MESHLET_COUNT = 16;
static constexpr uint32_t MESHLET_SEGMENTS = 6;
static constexpr uint32_t DIRECT_MESHLETS = MESHLET_COUNT / 2;
static constexpr uint32_t INDIRECT_MESHLETS = MESHLET_COUNT - DIRECT_MESHLETS;


static const std::string slang_meshlet = R"slang(
import noapi;

static const uint MAX_VERTICES = 64;
static const uint MAX_TRIANGLES = 32;

struct Varyings {
	float4 position : SV_Position;
	float3 color : COLOR;
}

struct MeshletVertex : IGpuLoadable {
	float2 position;
	float3 color;

	static const uint gpuStride = 20;

	static MeshletVertex gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		MeshletVertex out;
		out.position = from.next<float2>();
		out.color = from.next<float3>();
		return out;
	}
}

struct Meshlet : IGpuLoadable {
	GpuPtr<MeshletVertex> vertices;
	GpuPtr<uint> indices;
	uint vertexCount;
	uint triangleCount;

	static const uint gpuStride = 24;

	static Meshlet gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		Meshlet out;
		out.vertices = from.next<GpuPtr<MeshletVertex> >();
		out.indices = from.next<GpuPtr<uint> >();
		out.vertexCount = from.next<uint>();
		out.triangleCount = from.next<uint>();
		return out;
	}
}

struct MeshletDraw : IGpuLoadable {
	GpuPtr<Meshlet> meshlets;
	float angle;
	float aspect;

	static const uint gpuStride = 16;

	static MeshletDraw gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		MeshletDraw out;
		out.meshlets = from.next<GpuPtr<Meshlet> >();
		out.angle = from.next<float>();
		out.aspect = from.next<float>();
		return out;
	}
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(MAX_VERTICES, 1, 1)]
void meshletMain(
	uint thread : SV_GroupThreadID,
	uint group : SV_GroupID,
	out vertices Varyings outVertices[MAX_VERTICES],
	out indices uint3 outTriangles[MAX_TRIANGLES]
) {
	let draw = *gpuMeshletData<MeshletDraw>();
	let meshlet = draw.meshlets[group];

	SetMeshOutputCounts(meshlet.vertexCount, meshlet.triangleCount);

	if(thread < meshlet.vertexCount) {
		let vertex = meshlet.vertices[thread];

		let s = sin(draw.angle);
		let c = cos(draw.angle);
		let rotated = float2(
			vertex.position.x * c - vertex.position.y * s,
			vertex.position.x * s + vertex.position.y * c);

		Varyings varyings;
		varyings.position = float4(rotated.x / draw.aspect, rotated.y, 0.0, 1.0);
		varyings.color = vertex.color;
		outVertices[thread] = varyings;
	}

	if(thread < meshlet.triangleCount)
		outTriangles[thread] = uint3(
			meshlet.indices[thread * 3 + 0],
			meshlet.indices[thread * 3 + 1],
			meshlet.indices[thread * 3 + 2]);
}

[shader("fragment")]
float4 fragmentMain(Varyings varyings) : SV_Target {
	return float4(varyings.color, 1.0);
}
)slang";



struct MeshletVertex {
	float x, y;
	float r, g, b;
};
static_assert(sizeof(MeshletVertex) == 20, "MeshletVertex::gpuStride in the shader has to match");

struct Meshlet {
	gpu* vertices;
	gpu* indices;
	uint32_t vertex_count;
	uint32_t triangle_count;
};
static_assert(sizeof(Meshlet) == 24, "Meshlet::gpuStride in the shader has to match");

struct MeshletDraw {
	gpu* meshlets;
	float angle;
	float aspect;
};
static_assert(sizeof(MeshletDraw) == 16, "MeshletDraw::gpuStride in the shader has to match");

struct Disc {
	std::vector<MeshletVertex> vertices;
	std::vector<uint32_t> indices;
	uint32_t vertices_per_meshlet = 0;
	uint32_t triangles_per_meshlet = 0;
};

static Disc build_disc(uint32_t count, uint32_t segments) {
	constexpr float RADIUS = 0.85f;
	constexpr auto TAU = 2.0f * std::numbers::pi_v<float>;

	Disc out;
	out.vertices_per_meshlet = segments + 2;
	out.triangles_per_meshlet = segments;

	for(uint32_t wedge = 0; wedge < count; ++wedge) {
		auto hue = float(wedge) / float(count);
		vec3 color {
			0.5f + 0.5f * std::cos(TAU * hue),
			0.5f + 0.5f * std::cos(TAU * (hue + 1.0f / 3.0f)),
			0.5f + 0.5f * std::cos(TAU * (hue + 2.0f / 3.0f)),
		};

		out.vertices.push_back({0.0f, 0.0f, color.x * 0.35f, color.y * 0.35f, color.z * 0.35f});

		auto start = TAU * float(wedge) / float(count);
		auto step = TAU / float(count) / float(segments);
		for(uint32_t segment = 0; segment <= segments; ++segment) {
			auto angle = start + step * float(segment);
			out.vertices.push_back({
				RADIUS * std::cos(angle), RADIUS * std::sin(angle),
				color.x, color.y, color.z,
			});
		}

		for(uint32_t segment = 0; segment < segments; ++segment) {
			out.indices.push_back(0);
			out.indices.push_back(segment + 1);
			out.indices.push_back(segment + 2);
		}
	}

	return out;
}

TEST_MAIN({
	App app;
	open_window(app, "NoAPI Meshlets");
	create_device(app);

	gpuSetDiagnosticCallbackEXT(app.queue, [](GpuQueue*, GPU_DIAGNOSTIC kind, GpuStringView message, void*) {
		std::println(std::cerr, "[noapi {}] {}",
			kind == GPU_DIAGNOSTIC_UNSUPPORTED ? "unsupported" : "emulated",
			std::string_view(message.ptr, message.count));
	}, nullptr);

	const auto capabilities = gpuGetCapabilitiesEXT(app.queue);
	std::println("mesh shaders: {}", capabilities.mesh_shaders);

	auto disc = build_disc(MESHLET_COUNT, MESHLET_SEGMENTS);
	std::println("disc: {} meshlets ({} direct, {} indirect), {} vertices and {} triangles each",
		MESHLET_COUNT, DIRECT_MESHLETS, INDIRECT_MESHLETS,
		disc.vertices_per_meshlet, disc.triangles_per_meshlet);

	auto vertices = gpuMalloc<MeshletVertex>(app.queue, disc.vertices.size());
	memcpy(vertices, disc.vertices.data(), disc.vertices.size() * sizeof(MeshletVertex));
	auto vertices_gpu = gpuHostToDevicePointer(app.queue, vertices);

	auto indices = gpuMalloc<uint32_t>(app.queue, disc.indices.size());
	memcpy(indices, disc.indices.data(), disc.indices.size() * sizeof(uint32_t));
	auto indices_gpu = gpuHostToDevicePointer(app.queue, indices);

	auto meshlets = gpuMalloc<Meshlet>(app.queue, MESHLET_COUNT);
	for(uint32_t i = 0; i < MESHLET_COUNT; ++i)
		meshlets[i] = Meshlet{
			.vertices = vertices_gpu + size_t(i) * disc.vertices_per_meshlet * sizeof(MeshletVertex),
			.indices = indices_gpu + size_t(i) * disc.triangles_per_meshlet * 3 * sizeof(uint32_t),
			.vertex_count = disc.vertices_per_meshlet,
			.triangle_count = disc.triangles_per_meshlet,
		};
	auto meshlets_gpu = gpuHostToDevicePointer(app.queue, meshlets);

	auto draws = gpuMalloc<MeshletDraw>(app.queue, 2);
	draws[0] = MeshletDraw{.meshlets = meshlets_gpu, .angle = 0.0f, .aspect = 1.0f};
	draws[1] = MeshletDraw{
		.meshlets = meshlets_gpu + size_t(DIRECT_MESHLETS) * sizeof(Meshlet),
		.angle = 0.0f,
		.aspect = 1.0f,
	};
	auto draws_gpu = gpuHostToDevicePointer(app.queue, draws);
	auto indirect_draw_gpu = draws_gpu + sizeof(MeshletDraw);

	auto dispatch = gpuMalloc<uvec3>(app.queue);
	*dispatch = uvec3{INDIRECT_MESHLETS, 1, 1};
	auto dispatch_gpu = gpuHostToDevicePointer(app.queue, dispatch);

	gpuSyncMemoryEXT(app.queue, vertices_gpu);
	gpuSyncMemoryEXT(app.queue, indices_gpu);
	gpuSyncMemoryEXT(app.queue, meshlets_gpu);
	gpuSyncMemoryEXT(app.queue, dispatch_gpu);

	GpuColorTarget target { .format = app.surface_format };
	auto pipeline = gpuCreateGraphicsMeshletPipeline(app.queue, string_to_bytes(slang_meshlet), string_to_bytes(slang_meshlet), GpuRasterDesc{
		.cull = CULL_NONE,
		.colorTargets = {&target, 1},
	});

	assert(bool(pipeline) == capabilities.mesh_shaders
		&& "gpuCreateGraphicsMeshletPipeline disagreed with the queue's mesh_shaders capability");
	if(!pipeline)
		std::println(std::cerr,
			"this backend has no mesh shaders, so the disc cannot be drawn -- expect a clear window");

	auto started = glfwGetTime();

	run(app, [&](App& app, GpuCommandBuffer* cmd, const GpuTexture* backbuffer) {
		auto elapsed = float(glfwGetTime() - started);
		for(uint32_t i = 0; i < 2; ++i) {
			draws[i].angle = elapsed * 0.4f;
			draws[i].aspect = app.aspect();
		}
		gpuSyncMemoryEXT(cmd, draws_gpu);

		GpuColorAttachment color {
			.texture = backbuffer,
			.loadOp = LOAD_OP_CLEAR,
			.clearValue = {0.03f, 0.03f, 0.05f, 1.0f},
		};
		GpuRenderPassDesc pass { .colorAttachments = {&color, 1} };

		gpuBeginRenderPass(cmd, pass);
		if(pipeline) {
			gpuSetPipeline(cmd, pipeline);
			gpuDrawMeshlets(cmd, draws_gpu, draws_gpu, {DIRECT_MESHLETS, 1, 1});
			gpuDrawMeshletsIndirect(cmd, indirect_draw_gpu, indirect_draw_gpu, dispatch_gpu);
		} else {
			gpuDrawMeshlets(cmd, draws_gpu, draws_gpu, {DIRECT_MESHLETS, 1, 1});
			gpuDrawMeshletsIndirect(cmd, indirect_draw_gpu, indirect_draw_gpu, dispatch_gpu);
		}
		gpuEndRenderPass(cmd, pass);
	}, [&](App& app) {
		if(pipeline) gpuFreePipeline(app.queue, pipeline);
		gpuFree(app.queue, dispatch);
		gpuFree(app.queue, draws);
		gpuFree(app.queue, meshlets);
		gpuFree(app.queue, indices);
		gpuFree(app.queue, vertices);
	});

	return 0;
})
