#include "common.hpp"
#include "teapot.hpp"

using namespace test;


static const std::string slang_scene_module = R"slang(
module outline_scene;
import noapi;

public struct Vertex : IGpuLoadable {
	public float3 position;
	public float3 normal;

	public static const uint gpuStride = 24;

	public static Vertex gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		Vertex out;
		out.position = from.next<float3>();
		out.normal = from.next<float3>();
		return out;
	}
}

public struct SceneData : IGpuLoadable {
	public GpuPtr<Vertex> vertices;
	public float4x4 viewProjection;
	public float3 lightDirection;
	public float outlineWidth;

	public static const uint gpuStride = 88;

	public static SceneData gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		SceneData out;
		out.vertices = from.next<GpuPtr<Vertex> >();
		out.viewProjection = from.next<float4x4>();
		out.lightDirection = from.next<float3>();
		out.outlineWidth = from.next<float>();
		return out;
	}
}

public struct Varyings {
	public float4 position : SV_Position;
	public float3 normal : NORMAL;
}

public Varyings transform(uint index, float offset) {
	let scene = *gpuVertexData<SceneData>();
	let vertex = scene.vertices[index];

	Varyings output;
	output.position = mul(scene.viewProjection, float4(vertex.position + offset * vertex.normal, 1.0));
	output.normal = vertex.normal;
	return output;
}

public float4 shade(Varyings varyings, float3 albedo) {
	let scene = *gpuFragmentData<SceneData>();
	let lit = 0.25 + 0.75 * saturate(dot(normalize(varyings.normal), -scene.lightDirection));
	return float4(albedo * lit, 1.0);
}
)slang";

// Every pixel the teapot covers is marked in the stencil buffer as it is drawn
static const std::string slang_teapot = R"slang(
import noapi;
import outline_scene;

[shader("vertex")] Varyings vertexMain(uint index : SV_VertexID) { return transform(index, 0.0); }
[shader("fragment")] float4 fragmentMain(Varyings varyings) : SV_Target { return shade(varyings, float3(0.85, 0.35, 0.25)); }
)slang";

// The teapot again, pushed out along its normals, drawn only where the stencil says the teapot itself isn't
static const std::string slang_outline = R"slang(
import noapi;
import outline_scene;

[shader("vertex")] Varyings vertexMain(uint index : SV_VertexID) { return transform(index, (*gpuVertexData<SceneData>()).outlineWidth); }
[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(0.05, 0.05, 0.08, 1.0); }
)slang";

// Four vertices as a strip, clockwise seen from above; culling counter-clockwise faces hides it when the camera dips below
static const std::string slang_floor = R"slang(
import noapi;
import outline_scene;

[shader("vertex")] Varyings vertexMain(uint index : SV_VertexID) {
	let scene = *gpuVertexData<SceneData>();
	let corner = float2(float(index & 1u), float(index >> 1u)) * 6.0 - 3.0;

	Varyings output;
	output.position = mul(scene.viewProjection, float4(corner.x, -0.8, corner.y, 1.0));
	output.normal = float3(0.0, 1.0, 0.0);
	return output;
}
[shader("fragment")] float4 fragmentMain(Varyings varyings) : SV_Target { return shade(varyings, float3(0.6, 0.65, 0.7)); }
)slang";


struct SceneData {
	gpu* vertices;
	std::array<float, 16> view_projection;
	std::array<float, 3> light_direction;
	float outline_width;
};
static_assert(sizeof(SceneData) == 88, "SceneData::gpuStride in outline_scene has to match");


TEST_MAIN({
	App app;
	open_window(app, "NoAPI Outline");

	gpuAddShaderModuleEXT("outline_scene", GpuStringView{slang_scene_module.data(), slang_scene_module.size()});

	app.sample_count = 4;
	create_device(app);

	// The outline needs a stencil; which combined depth/stencil formats exist is up to the device
	for(auto format: {FORMAT_D24_PLUS_S8_UINT, FORMAT_D32_FLOAT_S8_UINT}) {
		GpuTextureDesc desc {.dimensions = {app.width, app.height, 1}, .sampleCount = app.sample_count, .format = format, .usage = USAGE_DEPTH_STENCIL_ATTACHMENT};
		if(gpuTextureSizeAlign(app.queue, desc).size) {
			app.depth_format = format;
			break;
		}
	}
	assert(app.depth_format != FORMAT_NONE && "No combined depth/stencil format with 4 samples");

	std::vector<uint32_t> teapot_indices;
	auto teapot_vertices = load_teapot(teapot_indices);

	auto vertices = gpuMalloc<MeshVertex>(app.queue, teapot_vertices.size());
	memcpy(vertices, teapot_vertices.data(), teapot_vertices.size() * sizeof(MeshVertex));
	auto indices = gpuMalloc<uint32_t>(app.queue, teapot_indices.size());
	memcpy(indices, teapot_indices.data(), teapot_indices.size() * sizeof(uint32_t));
	auto floor_indices = gpuMalloc<uint16_t>(app.queue, 4);
	for(uint16_t i = 0; i < 4; ++i) floor_indices[i] = i;

	auto scene = gpuMalloc<SceneData>(app.queue);
	*scene = SceneData{
		.vertices = gpuHostToDevicePointer(app.queue, vertices),
		.light_direction = {-0.4f, -0.8f, -0.45f},
		.outline_width = 0.03f,
	};
	auto scene_gpu = gpuHostToDevicePointer(app.queue, scene);
	auto indices_gpu = gpuHostToDevicePointer(app.queue, indices);
	auto floor_indices_gpu = gpuHostToDevicePointer(app.queue, floor_indices);
	gpuSyncMemoryEXT(app.queue, scene->vertices);
	gpuSyncMemoryEXT(app.queue, indices_gpu);
	gpuSyncMemoryEXT(app.queue, floor_indices_gpu);

	GpuColorTarget target { .format = app.surface_format };
	GpuRasterDesc raster {
		.sampleCount = uint8_t(app.sample_count),
		.depthFormat = app.depth_format,
		.stencilFormat = app.depth_format,
		.colorTargets = {&target, 1},
	};
	auto teapot_pipeline = gpuCreateGraphicsPipeline(app.queue, string_to_bytes(slang_teapot), string_to_bytes(slang_teapot), raster);
	auto outline_pipeline = gpuCreateGraphicsPipeline(app.queue, string_to_bytes(slang_outline), string_to_bytes(slang_outline), raster);
	GpuRasterDesc floor_raster = raster;
	floor_raster.topology = TOPOLOGY_TRIANGLE_STRIP;
	floor_raster.cull = CULL_CCW;
	auto floor_pipeline = gpuCreateGraphicsPipeline(app.queue, string_to_bytes(slang_floor), string_to_bytes(slang_floor), floor_raster);
	assert(teapot_pipeline && outline_pipeline && floor_pipeline && "Pipeline creation failed");

	// The teapot marks every pixel it covers with 1, and the outline only lands where nothing was marked
	constexpr GpuStencil mark {.test = OP_ALWAYS, .passOp = STENCIL_OP_REPLACE, .reference = 1};
	constexpr GpuStencil unmarked {.test = OP_NOT_EQUAL, .reference = 1};
	auto floor_depth = gpuCreateDepthStencilState(app.queue, GpuDepthStencilDesc{
		.depthMode = DEPTH_FLAGS(DEPTH_READ | DEPTH_WRITE),
		.depthTest = OP_LESS,
	});
	auto teapot_depth = gpuCreateDepthStencilState(app.queue, GpuDepthStencilDesc{
		.depthMode = DEPTH_FLAGS(DEPTH_READ | DEPTH_WRITE),
		.depthTest = OP_LESS,
		.stencilFront = mark,
		.stencilBack = mark,
	});
	auto outline_depth = gpuCreateDepthStencilState(app.queue, GpuDepthStencilDesc{
		.depthMode = DEPTH_READ,
		.depthTest = OP_LESS_EQUAL,
		.stencilWriteMask = 0,
		.stencilFront = unmarked,
		.stencilBack = unmarked,
	});

	auto index_count = uint32_t(teapot_indices.size());
	bool outline = true, o_held = false;
	std::println("controls: O toggles the outline");

	run(app, [&](App& app, GpuCommandBuffer* cmd, const GpuTexture* backbuffer) {
		bool o_down = glfwGetKey(app.window, GLFW_KEY_O) == GLFW_PRESS;
		if(o_down && !o_held) outline = !outline;
		o_held = o_down;

		auto angle = float(glfwGetTime()) * 0.4f;
		vec3 eye {std::cos(angle) * 3.4f, 1.2f + std::sin(angle * 0.7f) * 2.4f, std::sin(angle) * 3.4f};
		auto view_projection = perspective(0.9f, app.aspect(), 0.1f, 100.0f) * look_at(eye, {0, 0, 0}, {0, 1, 0});
		memcpy(scene->view_projection.data(), view_projection.m.data(), sizeof(scene->view_projection));
		gpuSyncMemoryEXT(cmd, scene_gpu);

		auto color = color_attachment(app, backbuffer, LOAD_OP_CLEAR, {0.55f, 0.7f, 0.85f, 1.0f});
		GpuDepthStencilAttachment depth {.texture = app.depth.texture, .loadOp = LOAD_OP_CLEAR, .storeOp = STORE_OP_DONT_CARE, .clearValue = 1.0};
		GpuDepthStencilAttachment stencil {.texture = app.depth.texture, .loadOp = LOAD_OP_CLEAR, .storeOp = STORE_OP_DONT_CARE, .clearValue = 0};
		GpuRenderPassDesc pass {
			.depthAttachment = depth,
			.stencilAttachment = stencil,
			.colorAttachments = {&color, 1},
		};

		gpuBeginRenderPass(cmd, pass);

		gpuSetDepthStencilState(cmd, floor_depth);
		gpuSetPipeline(cmd, floor_pipeline);
		gpuDrawIndexedInstanced(cmd, scene_gpu, scene_gpu, floor_indices_gpu, 4, 1, INDEX_TYPE_UINT16);

		gpuSetDepthStencilState(cmd, teapot_depth);
		gpuSetPipeline(cmd, teapot_pipeline);
		gpuDrawIndexedInstanced(cmd, scene_gpu, scene_gpu, indices_gpu, index_count, 1);

		if(outline) {
			gpuSetDepthStencilState(cmd, outline_depth);
			gpuSetPipeline(cmd, outline_pipeline);
			gpuDrawIndexedInstanced(cmd, scene_gpu, scene_gpu, indices_gpu, index_count, 1);
		}

		gpuEndRenderPass(cmd, pass);
	}, [&](App& app) {
		gpuFreeDepthStencilState(app.queue, floor_depth);
		gpuFreeDepthStencilState(app.queue, teapot_depth);
		gpuFreeDepthStencilState(app.queue, outline_depth);
		gpuFreePipeline(app.queue, floor_pipeline);
		gpuFreePipeline(app.queue, teapot_pipeline);
		gpuFreePipeline(app.queue, outline_pipeline);
		gpuFree(app.queue, vertices);
		gpuFree(app.queue, indices);
		gpuFree(app.queue, floor_indices);
		gpuFree(app.queue, scene);
	});

	return 0;
})
