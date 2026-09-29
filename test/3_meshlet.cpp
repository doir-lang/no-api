// A disc built out of meshlets: what a mesh shader pipeline looks like when there is no vertex
// stage, no vertex input and no index buffer anywhere in the frame.
//
// Everything new here is on the mesh shading side:
//
// - gpuCreateGraphicsMeshletPipeline, which is an optional feature. It is the first call in this
//   API that a whole backend can decline, so this test is as much about the declining as about the
//   drawing: gpuGetCapabilitiesEXT().mesh_shaders says up front whether there is a mesh stage, the
//   pipeline creation returns null where there is not, and the two draws report and do nothing. On
//   WebGPU that is always the path taken -- WebGPU has no mesh shaders in any form -- so this test
//   is expected to render nothing but its clear color there, and to say so rather than crash.
// - a [shader("mesh")] entry point, which writes its own vertices and its own triangle indices
//   into `out vertices` / `out indices` arrays after declaring how many of each it will fill with
//   SetMeshOutputCounts. One thread group per meshlet; the group is the meshlet index.
// - gpuMeshletData<T>(), the root data accessor a mesh shader reads instead of gpuVertexData<T>().
//   It is the same pointer -- a mesh shader replaces the vertex stage rather than joining it -- and
//   exists so the shader does not have to read as though it had a vertex stage.
// - both meshlet draws: gpuDrawMeshlets for the first half of the disc and
//   gpuDrawMeshletsIndirect, reading its group count out of gpu memory, for the second. Each half
//   is a distinct set of wedges, so a path that silently draws nothing leaves a visible hole
//   rather than being covered for by the other.
//
// The disc is 16 wedges, each one a meshlet: a center vertex, a rim, and a fan of triangles
// between them. The indices inside a meshlet are local to that meshlet's own vertex array, which
// is the whole point of the format -- deduplication happens offline, per meshlet, and the hardware
// index deduplication a gpuDrawIndexed* pass leans on is not involved at all.

#include "common.hpp"

#include <cmath>
#include <numbers>

using namespace test;

// One wedge of the disc per meshlet, split in two halves: the first drawn directly, the second
// indirectly. Both counts stay well inside the 64 vertices / 32 triangles the shader declares.
static constexpr uint32_t MESHLET_COUNT = 16;
static constexpr uint32_t MESHLET_SEGMENTS = 6; // Triangles in one wedge's fan
static constexpr uint32_t DIRECT_MESHLETS = MESHLET_COUNT / 2;
static constexpr uint32_t INDIRECT_MESHLETS = MESHLET_COUNT - DIRECT_MESHLETS;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

static const std::string slang_meshlet = R"slang(
import noapi;

// What one thread group may emit. A mesh shader has to promise these statically -- they size the
// output arrays -- and then say how much of the promise it is using, per group, with
// SetMeshOutputCounts. The real hardware ceilings are higher (256 vertices / 128 triangles on AMD,
// 126 / 64 on Nvidia); a wedge of this disc needs far less.
static const uint MAX_VERTICES = 64;
static const uint MAX_TRIANGLES = 32;

struct Varyings {
	float4 position : SV_Position;
	float3 color : COLOR;
}

// One entry of a meshlet's vertex array, matching MeshletVertex on the C++ side: five floats,
// {x, y, r, g, b}. The same layout 0_basic's Vertex has -- nothing about a vertex changes because
// a mesh shader is the one reading it.
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

// One meshlet, matching Meshlet on the C++ side. Self contained by construction: `indices` are
// local to `vertices`, so nothing outside the group this describes is ever read.
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

// The root data struct the draw was handed, matching MeshletDraw on the C++ side. `meshlets` is
// where this draw's run of wedges starts, which is how the direct and indirect halves are told
// apart without the shader knowing anything about either.
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

/**
 * meshletMain -- One thread group per meshlet, one thread per output vertex.
 *
 * SetMeshOutputCounts has to run before anything is written and has to agree across the group,
 * which is why it is outside both of the bounds checks below: the counts come from the meshlet,
 * not from what an individual thread turns out to have to do.
 */
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(MAX_VERTICES, 1, 1)]
void meshletMain(
	uint thread : SV_GroupThreadID,
	uint group : SV_GroupID,
	out vertices Varyings outVertices[MAX_VERTICES],
	out indices uint3 outTriangles[MAX_TRIANGLES]
) {
	// The root data is one pointer plus two floats, so the pointer to it is a pointer to a struct
	let draw = *gpuMeshletData<MeshletDraw>();
	let meshlet = draw.meshlets[group];

	SetMeshOutputCounts(meshlet.vertexCount, meshlet.triangleCount);

	// Threads past this meshlet's vertex count have nothing to emit. They still had to reach
	// SetMeshOutputCounts above, so this is a bounds check rather than an early return.
	if(thread < meshlet.vertexCount) {
		let vertex = meshlet.vertices[thread];

		// Spun and squared up here rather than on the host, so the mesh stage is doing real work
		// with the root data instead of copying a position through
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

	// One triangle per thread as well, from the meshlet's own local indices
	if(thread < meshlet.triangleCount)
		outTriangles[thread] = uint3(
			meshlet.indices[thread * 3 + 0],
			meshlet.indices[thread * 3 + 1],
			meshlet.indices[thread * 3 + 2]);
}

// The pixel stage never touches the root data it was handed; it just interpolates, exactly as it
// would behind a vertex stage. Nothing in a fragment shader knows which of the two produced it.
[shader("fragment")]
float4 fragmentMain(Varyings varyings) : SV_Target {
	return float4(varyings.color, 1.0);
}
)slang";

// ---------------------------------------------------------------------------
// Host side layouts
// ---------------------------------------------------------------------------

// Each of these has to be packed exactly the way its shader counterpart's gpuStride and gpuLoad
// say, since GpuReader walks the fields with no padding anywhere.

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

/**
 * Disc -- The wedges of the disc, flattened into the one vertex array and one index array the
 * meshlets point into.
 *
 * Every meshlet holds the same number of each, so a meshlet's slice is its index times that count;
 * the counts are still stored per meshlet, because a real meshlet stream's are not uniform and the
 * shader has no business assuming they are.
 */
struct Disc {
	std::vector<MeshletVertex> vertices;
	std::vector<uint32_t> indices; // Local to each meshlet's own slice of `vertices`
	uint32_t vertices_per_meshlet = 0;
	uint32_t triangles_per_meshlet = 0;
};

/**
 * build_disc -- Cut a disc into \p count wedges and make each one a meshlet.
 *
 * A wedge is a triangle fan: the center, then \p segments + 1 points around the rim, joined into
 * \p segments triangles wound counter-clockwise. The color is the wedge's own, so a meshlet that
 * fails to draw is a missing colored slice rather than a subtle seam.
 *
 * @param count How many wedges to cut the disc into, one meshlet each.
 * @param segments Triangles in one wedge's fan.
 */
static Disc build_disc(uint32_t count, uint32_t segments) {
	constexpr float RADIUS = 0.85f;
	constexpr auto TAU = 2.0f * std::numbers::pi_v<float>;

	Disc out;
	out.vertices_per_meshlet = segments + 2; // The center, plus one more rim point than triangles
	out.triangles_per_meshlet = segments;

	for(uint32_t wedge = 0; wedge < count; ++wedge) {
		// Each wedge gets a hue of its own, stepped around the color wheel the same way it is
		// stepped around the disc, so the two halves of the draw are told apart by eye
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

		// Local indices: 0 is this meshlet's center, 1..segments+1 its rim
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

	// Where a call declines, this is what says so. The two draws and the pipeline creation all
	// report through here, once per distinct message, so an unsupported backend explains itself
	// instead of quietly drawing nothing.
	gpuSetDiagnosticCallbackEXT(app.queue, [](GpuQueue*, GPU_DIAGNOSTIC kind, GpuStringView message, void*) {
		std::println(std::cerr, "[noapi {}] {}",
			kind == GPU_DIAGNOSTIC_UNSUPPORTED ? "unsupported" : "emulated",
			std::string_view(message.ptr, message.count));
	}, nullptr);

	// The one thing to read before building a meshlet path at all
	const auto capabilities = gpuGetCapabilitiesEXT(app.queue);
	std::println("mesh shaders: {}", capabilities.mesh_shaders);

	//
	// Geometry. One vertex array and one index array for the whole disc, plus the meshlet records
	// that carve them up -- which is all a mesh shader gets, since nothing binds a vertex buffer
	// or an index buffer to a mesh pipeline.
	//
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

	// The meshlet array itself. Each record points at its own slice of the two arrays above; the
	// byte offsets are the host side stride, because a gpu pointer steps in bytes.
	auto meshlets = gpuMalloc<Meshlet>(app.queue, MESHLET_COUNT);
	for(uint32_t i = 0; i < MESHLET_COUNT; ++i)
		meshlets[i] = Meshlet{
			.vertices = vertices_gpu + size_t(i) * disc.vertices_per_meshlet * sizeof(MeshletVertex),
			.indices = indices_gpu + size_t(i) * disc.triangles_per_meshlet * 3 * sizeof(uint32_t),
			.vertex_count = disc.vertices_per_meshlet,
			.triangle_count = disc.triangles_per_meshlet,
		};
	auto meshlets_gpu = gpuHostToDevicePointer(app.queue, meshlets);

	//
	// Root data, one struct per draw. The only difference between them is where in the meshlet
	// array their run of wedges starts, so the same shader covers both halves without knowing that
	// one of them came from an indirect dispatch.
	//
	auto draws = gpuMalloc<MeshletDraw>(app.queue, 2);
	draws[0] = MeshletDraw{.meshlets = meshlets_gpu, .angle = 0.0f, .aspect = 1.0f};
	draws[1] = MeshletDraw{
		.meshlets = meshlets_gpu + size_t(DIRECT_MESHLETS) * sizeof(Meshlet),
		.angle = 0.0f,
		.aspect = 1.0f,
	};
	auto draws_gpu = gpuHostToDevicePointer(app.queue, draws);
	auto indirect_draw_gpu = draws_gpu + sizeof(MeshletDraw);

	// The group count gpuDrawMeshletsIndirect reads, in gpu memory rather than in the call. Written
	// once from the host here; the point of the indirect form is that a compute pass could write it
	// instead, and nothing about the draw would change.
	auto dispatch = gpuMalloc<uvec3>(app.queue);
	*dispatch = uvec3{INDIRECT_MESHLETS, 1, 1};
	auto dispatch_gpu = gpuHostToDevicePointer(app.queue, dispatch);

	gpuSyncMemoryEXT(app.queue, vertices_gpu);
	gpuSyncMemoryEXT(app.queue, indices_gpu);
	gpuSyncMemoryEXT(app.queue, meshlets_gpu);
	gpuSyncMemoryEXT(app.queue, dispatch_gpu);

	//
	// Pipeline. One module supplies the mesh and the pixel stage both, found by stage rather than
	// by name, exactly as a vertex + pixel module is. There is no topology to pick -- the shader's
	// [outputtopology] is what decides that -- and no vertex layout to describe.
	//
	GpuColorTarget target { .format = app.surface_format };
	auto pipeline = gpuCreateGraphicsMeshletPipeline(app.queue, string_to_bytes(slang_meshlet), string_to_bytes(slang_meshlet), GpuRasterDesc{
		.cull = CULL_NONE, // The wedges spin, so neither winding stays front facing
		.colorTargets = {&target, 1},
	});

	// The capability has to be the truth about the pipeline, in both directions: a backend that
	// promises a mesh stage has to produce one, and a backend that does not has to decline rather
	// than hand back something unusable.
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
		// Outside the render pass, since it can record a copy: the root data the frame just rewrote
		// has to be on its way before anything reads it
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
			// Half the disc each, from the same pipeline and the same shader: the direct draw is
			// handed its group count, the indirect one reads it out of gpu memory
			gpuDrawMeshlets(cmd, draws_gpu, draws_gpu, {DIRECT_MESHLETS, 1, 1});
			gpuDrawMeshletsIndirect(cmd, indirect_draw_gpu, indirect_draw_gpu, dispatch_gpu);
		} else {
			// The unsupported path, recorded on purpose rather than skipped. With no mesh pipeline
			// to bind, both of these have to notice that for themselves and decline -- reporting
			// through the callback above and leaving the pass empty -- instead of recording a draw
			// that would be invalid.
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
