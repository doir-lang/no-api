#include "common.hpp"

#include "stb_image.h"
#include "tiny_obj_loader.h"

using namespace test;

static constexpr uint32_t ENVIRONMENT_WIDTH = 1024;
static constexpr uint32_t ENVIRONMENT_HEIGHT = 512;


static const std::string slang_scene_module = R"slang(
module pbr_scene;
import noapi;

public static const float PI = 3.14159265359;

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
	public float3 cameraPosition;
	public float exposure;
	public float3 cameraRight;
	public float roughness;
	public float3 cameraUp;
	public float metallic;
	public float3 cameraForward;
	public uint environment;
	public float3 baseColor;
	public float environmentMips;
	public float2 rayScale;

	public static const uint gpuStride = 160;

	public static SceneData gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		SceneData out;
		out.vertices = from.next<GpuPtr<Vertex> >();
		out.viewProjection = from.next<float4x4>();
		out.cameraPosition = from.next<float3>();
		out.exposure = from.next<float>();
		out.cameraRight = from.next<float3>();
		out.roughness = from.next<float>();
		out.cameraUp = from.next<float3>();
		out.metallic = from.next<float>();
		out.cameraForward = from.next<float3>();
		out.environment = from.next<uint>();
		out.baseColor = from.next<float3>();
		out.environmentMips = from.next<float>();
		out.rayScale = from.next<float2>();
		return out;
	}
}

public GpuSamplerDesc environmentSampler() {
	return GpuSamplerDesc(ADDRESS_MODE_REPEAT, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, FILTER_LINEAR, FILTER_LINEAR, FILTER_LINEAR);
}

public float2 directionToEquirect(float3 direction) {
	return float2(
		atan2(direction.z, direction.x) / (2.0 * PI) + 0.5,
		acos(clamp(direction.y, -1.0, 1.0)) / PI
	);
}

public float3 sampleEnvironment(SceneData scene, float3 direction, float mip) {
	let uv = directionToEquirect(normalize(direction));
	let slot = gpuGetSamplerIndex(environmentSampler());
	return gpuSample(scene.environment, slot, float3(uv, 0.0), 0u, mip).rgb;
}

public float3 tonemap(float3 color) {
	return saturate((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14));
}

public float3 encodeGamma(float3 color) {
	return pow(saturate(color), float3(1.0 / 2.2));
}
)slang";

static const std::string slang_sky = R"slang(
import noapi;
import pbr_scene;

struct Varyings {
	float4 position : SV_Position;
	float2 ndc : TEXCOORD0;
}

[shader("vertex")]
Varyings vertexMain(uint index : SV_VertexID) {
	let corner = float2(float((index << 1u) & 2u), float(index & 2u)) * 2.0 - 1.0;

	Varyings output;
	output.position = float4(corner, 1.0, 1.0);
	output.ndc = corner;
	return output;
}

[shader("fragment")]
float4 fragmentMain(Varyings varyings) : SV_Target {
	let scene = *gpuFragmentData<SceneData>();

	let direction = scene.cameraForward
		+ scene.cameraRight * (varyings.ndc.x * scene.rayScale.x)
		+ scene.cameraUp * (varyings.ndc.y * scene.rayScale.y);

	return float4(encodeGamma(tonemap(sampleEnvironment(scene, direction, 0.0) * scene.exposure)), 1.0);
}
)slang";

static const std::string slang_teapot = R"slang(
import noapi;
import pbr_scene;

struct Varyings {
	float4 position : SV_Position;
	float3 worldPosition : TEXCOORD0;
	float3 normal : NORMAL;
}

float3 fresnelSchlickRoughness(float cosTheta, float3 f0, float roughness) {
	let ceiling = max(float3(1.0 - roughness), f0);
	return f0 + (ceiling - f0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float2 environmentBRDF(float roughness, float NoV) {
	const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
	const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
	let r = roughness * c0 + c1;
	let a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
	return float2(-1.04, 1.04) * a004 + float2(r.z, r.w);
}

[shader("vertex")]
Varyings vertexMain(uint index : SV_VertexID) {
	let scene = *gpuVertexData<SceneData>();
	let vertex = scene.vertices[index];

	Varyings output;
	output.position = mul(scene.viewProjection, float4(vertex.position, 1.0));
	output.worldPosition = vertex.position;
	output.normal = vertex.normal;
	return output;
}

[shader("fragment")]
float4 fragmentMain(Varyings varyings) : SV_Target {
	let scene = *gpuFragmentData<SceneData>();

	var N = normalize(varyings.normal);
	let V = normalize(scene.cameraPosition - varyings.worldPosition);
	if(dot(N, V) < 0.0) N = -N;

	let NoV = saturate(dot(N, V));
	let R = reflect(-V, N);

	let roughness = clamp(scene.roughness, 0.04, 1.0);
	let f0 = lerp(float3(0.04), scene.baseColor, scene.metallic);
	let diffuseColor = scene.baseColor * (1.0 - scene.metallic);

	let irradiance = sampleEnvironment(scene, N, scene.environmentMips - 3.0);

	let prefiltered = sampleEnvironment(scene, R, roughness * (scene.environmentMips - 1.0));
	let ab = environmentBRDF(roughness, NoV);

	let specularWeight = fresnelSchlickRoughness(NoV, f0, roughness);
	let diffuseWeight = float3(1.0) - specularWeight;

	let color = diffuseWeight * diffuseColor * irradiance + prefiltered * (f0 * ab.x + float3(ab.y));

	return float4(encodeGamma(tonemap(color * scene.exposure)), 1.0);
}
)slang";


struct MeshVertex {
	std::array<float, 3> position;
	std::array<float, 3> normal;
};
static_assert(sizeof(MeshVertex) == 24, "Vertex::gpuStride in pbr_scene has to match");

struct SceneData {
	gpu* vertices;
	std::array<float, 16> view_projection;
	std::array<float, 3> camera_position;
	float exposure;
	std::array<float, 3> camera_right;
	float roughness;
	std::array<float, 3> camera_up;
	float metallic;
	std::array<float, 3> camera_forward;
	uint32_t environment;
	std::array<float, 3> base_color;
	float environment_mips;
	std::array<float, 2> ray_scale;
};
static_assert(sizeof(SceneData) == 160, "SceneData::gpuStride in pbr_scene has to match");

static constexpr GpuSamplerDesc ENVIRONMENT_SAMPLER {
	.address_mode_u = ADDRESS_MODE_REPEAT,
	.address_mode_v = ADDRESS_MODE_CLAMP,
	.address_mode_w = ADDRESS_MODE_CLAMP,
};


static std::vector<uint16_t> load_environment() {
	constexpr auto path = TEST_ASSET_DIR "/sunflowers_puresky_2k.hdr";

	int width, height, channels;
	auto pixels = stbi_loadf(path, &width, &height, &channels, 3);
	if(!pixels) throw std::runtime_error(std::string("Failed to load ") + path + ": " + stbi_failure_reason());

	auto resampled = resample<3>(std::span<const float>(pixels, size_t(width) * height * 3),
		uint32_t(width), uint32_t(height), ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT);

	auto peak = *std::max_element(resampled.begin(), resampled.end());
	std::println("environment: {}x{} ({} channels), resampled to {}x{}, peak radiance {:.1f}",
		width, height, channels, ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, peak);
	stbi_image_free(pixels);

	std::vector<uint16_t> texels(size_t(ENVIRONMENT_WIDTH) * ENVIRONMENT_HEIGHT * 4);
	for(size_t i = 0; i < size_t(ENVIRONMENT_WIDTH) * ENVIRONMENT_HEIGHT; ++i) {
		texels[i * 4 + 0] = float_to_half(resampled[i * 3 + 0]);
		texels[i * 4 + 1] = float_to_half(resampled[i * 3 + 1]);
		texels[i * 4 + 2] = float_to_half(resampled[i * 3 + 2]);
		texels[i * 4 + 3] = float_to_half(1.0f);
	}
	return texels;
}

static std::vector<MeshVertex> load_teapot(std::vector<uint32_t>& out_indices) {
	constexpr auto path = TEST_ASSET_DIR "/teapot.obj";

	tinyobj::ObjReaderConfig config;
	config.triangulate = true;
	config.mtl_search_path = TEST_ASSET_DIR;

	tinyobj::ObjReader reader;
	if(!reader.ParseFromFile(path, config))
		throw std::runtime_error(std::string("Failed to load ") + path + ": " + reader.Error());
	if(!reader.Warning().empty()) std::println("[obj] {}", reader.Warning());

	auto& attrib = reader.GetAttrib();
	auto count = attrib.vertices.size() / 3;

	std::vector<MeshVertex> vertices(count);
	for(size_t i = 0; i < count; ++i)
		vertices[i] = {{attrib.vertices[i * 3], attrib.vertices[i * 3 + 1], attrib.vertices[i * 3 + 2]}, {0, 0, 0}};

	out_indices.clear();
	for(auto& shape: reader.GetShapes())
		for(size_t face = 0; face + 2 < shape.mesh.indices.size(); face += 3) {
			std::array<uint32_t, 3> corner;
			for(int i = 0; i < 3; ++i)
				corner[i] = uint32_t(shape.mesh.indices[face + i].vertex_index);

			auto at = [&](uint32_t index) {
				auto& v = vertices[index].position;
				return vec3{v[0], v[1], v[2]};
			};
			auto weighted = cross(at(corner[1]) - at(corner[0]), at(corner[2]) - at(corner[0]));

			for(int i = 0; i < 3; ++i) {
				auto& normal = vertices[corner[i]].normal;
				normal[0] += weighted.x;
				normal[1] += weighted.y;
				normal[2] += weighted.z;
				out_indices.push_back(corner[i]);
			}
		}

	auto& first = vertices.front().position;
	vec3 low{first[0], first[1], first[2]}, high = low;
	for(auto& vertex: vertices) {
		low = {std::min(low.x, vertex.position[0]), std::min(low.y, vertex.position[1]), std::min(low.z, vertex.position[2])};
		high = {std::max(high.x, vertex.position[0]), std::max(high.y, vertex.position[1]), std::max(high.z, vertex.position[2])};
	}
	auto centre = (low + high) * 0.5f;
	auto extent = high - low;
	auto scale = 2.0f / std::max({extent.x, extent.y, extent.z});

	for(auto& vertex: vertices) {
		vertex.position[0] = (vertex.position[0] - centre.x) * scale;
		vertex.position[1] = (vertex.position[1] - centre.y) * scale;
		vertex.position[2] = (vertex.position[2] - centre.z) * scale;

		auto normal = normalize(vec3{vertex.normal[0], vertex.normal[1], vertex.normal[2]});
		vertex.normal[0] = normal.x;
		vertex.normal[1] = normal.y;
		vertex.normal[2] = normal.z;
	}

	std::println("teapot: {} vertices, {} triangles", vertices.size(), out_indices.size() / 3);
	return vertices;
}


struct Material {
	float roughness = 0.2f;
	float metallic = 1.0f;
	float exposure = 1.0f;
	bool metal_held = false;
};

static void poll_material(GLFWwindow* window, Material& material, float dt) {
	if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) material.roughness += dt * 0.5f;
	if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) material.roughness -= dt * 0.5f;
	material.roughness = std::clamp(material.roughness, 0.0f, 1.0f);

	if(glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) material.exposure *= 1.0f + dt;
	if(glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) material.exposure /= 1.0f + dt;
	material.exposure = std::clamp(material.exposure, 0.05f, 20.0f);

	bool metal_down = glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS;
	if(metal_down && !material.metal_held) material.metallic = material.metallic > 0.5f ? 0.0f : 1.0f;
	material.metal_held = metal_down;
}

TEST_MAIN({
	App app;
	open_window(app, "NoAPI PBR");

	gpuAddShaderModuleEXT("pbr_scene", GpuStringView{slang_scene_module.data(), slang_scene_module.size()});

	app.depth_format = FORMAT_D32_FLOAT;
	create_device(app);

	auto panorama = load_environment();
	auto environment = create_texture(app.queue, GpuTextureDesc{
		.dimensions = {ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT, 1},
		.mipCount = mip_count_for(ENVIRONMENT_WIDTH, ENVIRONMENT_HEIGHT),
		.format = FORMAT_RGBA16_FLOAT,
		.usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_DST),
	});
	upload_texture(app.queue, environment, std::as_bytes(std::span(panorama)));
	std::println("environment: {} mip levels, roughness 1 reads level {}",
		environment.desc.mipCount, environment.desc.mipCount - 1);

	auto heap = gpuMalloc<GpuTextureDescriptor>(app.queue, 1, MEMORY_DESCRIPTOR_HEAP);
	heap[0] = gpuTextureViewDescriptor(app.queue, environment.texture, GpuViewDesc{});
	auto heap_gpu = gpuHostToDevicePointer(app.queue, heap);
	gpuSyncMemoryEXT(app.queue, heap_gpu);

	std::vector<uint32_t> teapot_indices;
	auto teapot_vertices = load_teapot(teapot_indices);

	auto vertices = gpuMalloc<MeshVertex>(app.queue, teapot_vertices.size());
	memcpy(vertices, teapot_vertices.data(), teapot_vertices.size() * sizeof(MeshVertex));

	auto indices = gpuMalloc<uint32_t>(app.queue, teapot_indices.size());
	memcpy(indices, teapot_indices.data(), teapot_indices.size() * sizeof(uint32_t));

	auto sky_indices = gpuMalloc<uint32_t>(app.queue, 3);
	sky_indices[0] = 0; sky_indices[1] = 1; sky_indices[2] = 2;

	auto scene = gpuMalloc<SceneData>(app.queue);
	*scene = SceneData{
		.vertices = gpuHostToDevicePointer(app.queue, vertices),
		.environment = 0,
		.base_color = {0.95f, 0.80f, 0.55f},
		.environment_mips = float(environment.desc.mipCount),
	};
	auto scene_gpu = gpuHostToDevicePointer(app.queue, scene);
	auto indices_gpu = gpuHostToDevicePointer(app.queue, indices);
	auto sky_indices_gpu = gpuHostToDevicePointer(app.queue, sky_indices);

	gpuSyncMemoryEXT(app.queue, scene->vertices);
	gpuSyncMemoryEXT(app.queue, indices_gpu);
	gpuSyncMemoryEXT(app.queue, sky_indices_gpu);

	GpuColorTarget target { .format = app.surface_format };
	GpuRasterDesc raster {
		.cull = CULL_NONE,
		.depthFormat = FORMAT_D32_FLOAT,
		.colorTargets = {&target, 1},
	};

	auto sky_pipeline = gpuCreateGraphicsPipeline(app.queue, string_to_bytes(slang_sky), string_to_bytes(slang_sky), raster);
	assert(sky_pipeline && "Sky pipeline creation failed");
	auto teapot_pipeline = gpuCreateGraphicsPipeline(app.queue, string_to_bytes(slang_teapot), string_to_bytes(slang_teapot), raster);
	assert(teapot_pipeline && "Teapot pipeline creation failed");

	auto sky_depth = gpuCreateDepthStencilState(app.queue, GpuDepthStencilDesc{});
	auto teapot_depth = gpuCreateDepthStencilState(app.queue, GpuDepthStencilDesc{
		.depthMode = DEPTH_FLAGS(DEPTH_READ | DEPTH_WRITE),
		.depthTest = OP_LESS,
	});

	Material material;
	auto index_count = uint32_t(teapot_indices.size());
	double last_time = glfwGetTime();

	std::println("controls: up/down roughness, M metal/dielectric, [ ] exposure");

	run(app, [&](App& app, GpuCommandBuffer* cmd, const GpuTexture* backbuffer) {
		auto now = glfwGetTime();
		auto dt = float(now - last_time);
		last_time = now;
		poll_material(app.window, material, dt);

		constexpr float fov = 0.9f;
		auto angle = float(now) * 0.3f;
		vec3 eye {std::cos(angle) * 3.2f, 1.1f, std::sin(angle) * 3.2f};
		vec3 target {0.0f, 0.0f, 0.0f};

		auto forward = normalize(target - eye);
		auto right = normalize(cross(forward, vec3{0, 1, 0}));
		auto up = cross(right, forward);

		auto view_projection = perspective(fov, app.aspect(), 0.1f, 100.0f) * look_at(eye, target, {0, 1, 0});
		memcpy(scene->view_projection.data(), view_projection.m.data(), sizeof(scene->view_projection));

		scene->camera_position[0] = eye.x; scene->camera_position[1] = eye.y; scene->camera_position[2] = eye.z;
		scene->camera_right[0] = right.x; scene->camera_right[1] = right.y; scene->camera_right[2] = right.z;
		scene->camera_up[0] = up.x; scene->camera_up[1] = up.y; scene->camera_up[2] = up.z;
		scene->camera_forward[0] = forward.x; scene->camera_forward[1] = forward.y; scene->camera_forward[2] = forward.z;

		auto tan_half = std::tan(fov * 0.5f);
		scene->ray_scale[0] = tan_half * app.aspect();
		scene->ray_scale[1] = tan_half;

		scene->roughness = material.roughness;
		scene->metallic = material.metallic;
		scene->exposure = material.exposure;

		gpuSyncMemoryEXT(cmd, scene_gpu);
		gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
		gpuSetEnabledSamplersEXT(cmd, {&ENVIRONMENT_SAMPLER, 1});

		GpuColorAttachment color {
			.texture = backbuffer,
			.loadOp = LOAD_OP_DONT_CARE,
		};
		GpuDepthStencilAttachment depth {
			.texture = app.depth.texture,
			.loadOp = LOAD_OP_CLEAR,
			.clearValue = 1.0,
		};
		GpuRenderPassDesc pass {
			.depthAttachment = depth,
			.colorAttachments = {&color, 1},
		};

		gpuBeginRenderPass(cmd, pass);

		gpuSetDepthStencilState(cmd, sky_depth);
		gpuSetPipeline(cmd, sky_pipeline);
		gpuDrawIndexedInstanced(cmd, scene_gpu, scene_gpu, sky_indices_gpu, 3, 1);

		gpuSetDepthStencilState(cmd, teapot_depth);
		gpuSetPipeline(cmd, teapot_pipeline);
		gpuDrawIndexedInstanced(cmd, scene_gpu, scene_gpu, indices_gpu, index_count, 1);

		gpuEndRenderPass(cmd, pass);
	}, [&](App& app) {
		gpuFreeDepthStencilState(app.queue, sky_depth);
		gpuFreeDepthStencilState(app.queue, teapot_depth);
		gpuFreePipeline(app.queue, sky_pipeline);
		gpuFreePipeline(app.queue, teapot_pipeline);
		gpuFree(app.queue, environment.memory);
		gpuFree(app.queue, heap);
		gpuFree(app.queue, vertices);
		gpuFree(app.queue, indices);
		gpuFree(app.queue, sky_indices);
		gpuFree(app.queue, scene);
	});

	return 0;
})
