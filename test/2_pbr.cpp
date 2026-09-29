// A physically based teapot lit entirely by a skybox.
//
// There is no light in this scene. Everything the teapot is lit by comes out of one
// equirectangular HDR panorama, sampled twice per pixel through the split sum approximation:
//
// - the diffuse term reads the environment around the surface normal, from far enough down the mip
//   chain that the blur stands in for a cosine convolution
// - the specular term reads it around the reflection vector, at the mip whose blur matches the
//   material's roughness, and multiplies it by an analytic fit to the environment BRDF that a
//   conventional renderer would instead look up in a precomputed table
//
// Which makes the mip chain the point of the test. gpuCopyToTexture can only write level 0, so the
// chain is rasterized by gpuBlitTextureEXT halving the level above it (see upload_texture in
// common.hpp), and because nothing in this API samples with implicit derivatives, roughness picks
// its level explicitly through gpuSample's mip argument. Turning roughness up with the arrow keys
// is watching that chain get walked.
//
// What else is new since 1_texture: a depth buffer and gpuSetDepthStencilState, two pipelines in
// one render pass, a shared Slang module registered with gpuAddShaderModuleEXT so both of them can
// import the scene layout rather than repeat it, and a mesh loaded off disk with normals computed
// for it.
//
// Controls: up/down for roughness, M for metal/dielectric, [ and ] for exposure.

#include "common.hpp"

#include "stb_image.h"
#include "tiny_obj_loader.h"

using namespace test;

// The panorama is resampled to this before upload; see resample in common.hpp for why it is a
// power of two. Halving all the way down gives the 11 levels roughness interpolates over.
static constexpr uint32_t ENVIRONMENT_WIDTH = 1024;
static constexpr uint32_t ENVIRONMENT_HEIGHT = 512;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

/**
 * The scene layout and the environment lookups, as a module both pipelines import.
 *
 * A pipeline's source has to hold exactly one entry point per stage it is compiled for, so the sky
 * and the teapot cannot live in one module. Registering the part they share with
 * gpuAddShaderModuleEXT is how they stop being two copies of it -- and it means the root data
 * layout, which has to agree with the C++ struct below field for field, is written down once.
 */
static const std::string slang_scene_module = R"slang(
module pbr_scene;
import noapi;

public static const float PI = 3.14159265359;

// One entry of the vertex array, matching MeshVertex on the C++ side: six floats
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

/**
 * The root data both draws are handed, matching SceneData on the C++ side.
 *
 * The fields are ordered so that every float3 is followed by a float, which is what makes the
 * host struct pack with no padding anywhere -- nothing in this API inserts any, so a gap here
 * would have to be spelled out on both sides with a GpuReader.skip.
 */
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
	public uint environment; // Index into the texture heap, and the whole of how it is bound
	public float3 baseColor;
	public float environmentMips;
	public float2 rayScale; // Half the film's extent at unit distance

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

// The sampler the host enabled with gpuSetEnabledSamplersEXT. A panorama wraps in longitude and
// stops at the poles, so u repeats and v clamps; repeating v would fold the sky back over itself
// one texel past the top.
public GpuSamplerDesc environmentSampler() {
	return GpuSamplerDesc(ADDRESS_MODE_REPEAT, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP,
		FILTER_LINEAR, FILTER_LINEAR, FILTER_LINEAR);
}

/// Where a direction lands on an equirectangular panorama: longitude across, latitude down, with
/// v = 0 at +Y because that is the row the .hdr starts with.
public float2 directionToEquirect(float3 direction) {
	return float2(
		atan2(direction.z, direction.x) / (2.0 * PI) + 0.5,
		acos(clamp(direction.y, -1.0, 1.0)) / PI);
}

/**
 * The environment's radiance along `direction`, read at an explicit mip level.
 *
 * `mip` is the whole lighting model: 0 is the sky itself, the middle of the chain is a rough
 * reflection, and the far end is blurred enough to pass for irradiance.
 */
public float3 sampleEnvironment(SceneData scene, float3 direction, float mip) {
	let uv = directionToEquirect(normalize(direction));
	let slot = gpuGetSamplerIndex(environmentSampler());
	return gpuSample(scene.environment, slot, float3(uv, 0.0), 0u, mip).rgb;
}

/// Narkowicz's curve fit to the ACES filmic tone map, which is what turns the panorama's very
/// large radiances -- the sun in this one runs into the thousands -- into something displayable.
public float3 tonemap(float3 color) {
	return saturate((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14));
}

/// The surface was configured to a non-sRGB format so that nothing encodes behind our back, which
/// leaves the encoding to do here.
public float3 encodeGamma(float3 color) {
	return pow(saturate(color), float3(1.0 / 2.2));
}
)slang";

/// The sky: a full screen triangle with the view ray rebuilt from the camera basis, which is
/// cheaper and shorter than inverting the view projection to do the same job.
static const std::string slang_sky = R"slang(
import noapi;
import pbr_scene;

struct Varyings {
	float4 position : SV_Position;
	float2 ndc : TEXCOORD0; // -1..1 across the viewport, +Y up, whichever backend this is
}

[shader("vertex")]
Varyings vertexMain(uint index : SV_VertexID) {
	// An oversized triangle covering the whole viewport from three vertices
	let corner = float2(float((index << 1u) & 2u), float(index & 2u)) * 2.0 - 1.0;

	Varyings output;
	// The far plane. The sky neither tests nor writes depth, so this only matters to anything
	// downstream that starts caring.
	output.position = float4(corner, 1.0, 1.0);
	output.ndc = corner;
	return output;
}

[shader("fragment")]
float4 fragmentMain(Varyings varyings) : SV_Target {
	let scene = *gpuFragmentData<SceneData>();

	// rayScale carries the field of view and the aspect ratio
	let direction = scene.cameraForward
		+ scene.cameraRight * (varyings.ndc.x * scene.rayScale.x)
		+ scene.cameraUp * (varyings.ndc.y * scene.rayScale.y);

	return float4(encodeGamma(tonemap(sampleEnvironment(scene, direction, 0.0) * scene.exposure)), 1.0);
}
)slang";

/// The teapot: image based lighting through the split sum approximation, with no light source in
/// the scene at all.
static const std::string slang_teapot = R"slang(
import noapi;
import pbr_scene;

struct Varyings {
	float4 position : SV_Position;
	float3 worldPosition : TEXCOORD0;
	float3 normal : NORMAL;
}

/**
 * Schlick's Fresnel with a roughness aware ceiling.
 *
 * The plain form goes to white at grazing angles, which on a rough surface is brighter than the
 * environment it is supposed to be reflecting; clamping the ceiling to 1 - roughness is the usual
 * fix and is what keeps a rough metal from developing a bright rim.
 */
float3 fresnelSchlickRoughness(float cosTheta, float3 f0, float roughness) {
	let ceiling = max(float3(1.0 - roughness), f0);
	return f0 + (ceiling - f0) * pow(saturate(1.0 - cosTheta), 5.0);
}

/**
 * Karis's analytic fit to the split sum's environment BRDF: the scale and bias to apply to the
 * specular colour for a given roughness and view angle.
 *
 * A production renderer bakes this into a two channel lookup texture once. The fit is used here
 * instead so that the scene owns exactly one texture and the lighting is all in the shader.
 */
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
	// The model is already in world space -- the C++ side centred and scaled it once, on load --
	// so there is no model matrix to apply to either the position or the normal
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
	// The teapot is drawn with culling off, because the model's lid and spout are open surfaces
	// whose winding can't be relied on. Facing the normal towards the viewer is what keeps the
	// inside of it from shading as though it were lit from behind.
	if(dot(N, V) < 0.0) N = -N;

	let NoV = saturate(dot(N, V));
	let R = reflect(-V, N);

	// A perfectly smooth surface has no mip to read for its diffuse term and a Fresnel that never
	// settles, so roughness gets a floor rather than reaching zero
	let roughness = clamp(scene.roughness, 0.04, 1.0);
	// Metals have no diffuse and take their specular colour from the albedo; dielectrics reflect
	// a flat 4% and keep their albedo for the diffuse
	let f0 = lerp(float3(0.04), scene.baseColor, scene.metallic);
	let diffuseColor = scene.baseColor * (1.0 - scene.metallic);

	// Diffuse: the irradiance arriving around N. Properly this is the environment convolved with a
	// cosine lobe, computed once offline; three levels from the end of the mip chain the panorama
	// is a handful of texels across and blurry enough to stand in for it.
	let irradiance = sampleEnvironment(scene, N, scene.environmentMips - 3.0);

	// Specular: the split sum approximation. One sample of prefiltered radiance around R, times
	// the environment BRDF. The prefilter is a box filtered mip chain rather than a GGX filtered
	// one, so the lobe's shape is only approximately right -- but it widens with roughness at
	// roughly the right rate, and it costs one texture and one sample.
	let prefiltered = sampleEnvironment(scene, R, roughness * (scene.environmentMips - 1.0));
	let ab = environmentBRDF(roughness, NoV);

	let specularWeight = fresnelSchlickRoughness(NoV, f0, roughness);
	let diffuseWeight = float3(1.0) - specularWeight;

	let color = diffuseWeight * diffuseColor * irradiance
		+ prefiltered * (f0 * ab.x + float3(ab.y));

	return float4(encodeGamma(tonemap(color * scene.exposure)), 1.0);
}
)slang";

// ---------------------------------------------------------------------------
// Host side layouts
// ---------------------------------------------------------------------------

struct MeshVertex {
	float position[3];
	float normal[3];
};
static_assert(sizeof(MeshVertex) == 24, "Vertex::gpuStride in pbr_scene has to match");

// Keep in sync with SceneData in pbr_scene above, field for field. Every float3 is followed by a
// float so that the whole thing packs without padding, which is what lets the shader read it with
// a plain run of GpuReader::next calls.
struct SceneData {
	gpu* vertices;
	float view_projection[16];
	float camera_position[3];
	float exposure;
	float camera_right[3];
	float roughness;
	float camera_up[3];
	float metallic;
	float camera_forward[3];
	uint32_t environment;
	float base_color[3];
	float environment_mips;
	float ray_scale[2];
};
static_assert(sizeof(SceneData) == 160, "SceneData::gpuStride in pbr_scene has to match");

// The sampler pbr_scene's environmentSampler() describes; gpuGetSamplerIndex matches them by the
// packed value, so only the fields have to agree
static constexpr GpuSamplerDesc ENVIRONMENT_SAMPLER {
	.address_mode_u = ADDRESS_MODE_REPEAT,
	.address_mode_v = ADDRESS_MODE_CLAMP,
	.address_mode_w = ADDRESS_MODE_CLAMP,
};

// ---------------------------------------------------------------------------
// Assets
// ---------------------------------------------------------------------------

/**
 * load_environment -- Read the .hdr panorama and pack it as the half float texels a
 * FORMAT_RGBA16_FLOAT texture holds.
 *
 * Half rather than full float because the 32 bit float formats are not filterable
 * (gpuFormatIsFilterableEXT), which would rule out both the bilinear sampling the lookups want and
 * the blits that build the mip chain.
 */
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

/**
 * load_teapot -- Read the mesh, give it the smooth normals the file does not carry, and place it
 * at the origin at a size the camera below is framed for.
 *
 * The .obj is positions and triangles and nothing else, so the normals are accumulated per vertex
 * from the faces that share it. The cross product of two edges is already scaled by twice the
 * triangle's area, so summing them unnormalized weights each face by its size, which is what keeps
 * the teapot's dense rim from dragging the normals around on its sparse body.
 *
 * @param out_indices Filled in with the triangle list.
 */
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

	// The file has no normals and no per-face vertex splitting, so a position index identifies a
	// vertex outright and the accumulated normal is shared by every face touching it
	out_indices.clear();
	for(auto& shape: reader.GetShapes())
		for(size_t face = 0; face + 2 < shape.mesh.indices.size(); face += 3) {
			uint32_t corner[3];
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

	// Centre it on the origin and scale its longest axis to 2 units, so that the camera and the
	// near/far planes below don't have to know anything about this particular file
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

// ---------------------------------------------------------------------------

/// What the keyboard drives, so that the mip chain the lighting reads can actually be seen moving.
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

	// Edge triggered, so that holding the key doesn't flicker the material every frame
	bool metal_down = glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS;
	if(metal_down && !material.metal_held) material.metallic = material.metallic > 0.5f ? 0.0f : 1.0f;
	material.metal_held = metal_down;
}

TEST_MAIN({
	App app;
	open_window(app, "NoAPI PBR");

	// Registered before the queue exists, let alone any pipeline: shaders are compiled inside the
	// pipeline creation calls, so anything they import has to be resolvable by then
	gpuAddShaderModuleEXT("pbr_scene", GpuStringView{slang_scene_module.data(), slang_scene_module.size()});

	// The teapot needs somewhere to be occluded by itself, so the harness carries a depth buffer
	// and keeps it the size of the window
	app.depth_format = FORMAT_D32_FLOAT;
	create_device(app);

	//
	// The environment. One texture, a full mip chain, and the whole of the scene's lighting.
	// USAGE_COLOR_ATTACHMENT is there on top of USAGE_SAMPLED because those mips are rendered
	// rather than copied; see upload_texture.
	//
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

	auto heap = gpuMalloc<GpuTextureDescriptor>(app.queue, 1);
	heap[0] = gpuTextureViewDescriptor(app.queue, environment.texture, GpuViewDesc{});
	auto heap_gpu = gpuHostToDevicePointer(app.queue, heap);
	gpuSyncMemoryEXT(app.queue, heap_gpu);

	//
	// Geometry: the teapot, plus three indices for the sky's full screen triangle. The sky's
	// vertex shader builds its own corners from SV_VertexID, but every draw entry point in this
	// API is indexed, so it still needs an index buffer to be handed.
	//
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
		// A warm metal, so that what the teapot shows is mostly the sky rather than its own colour
		.base_color = {0.95f, 0.80f, 0.55f},
		.environment_mips = float(environment.desc.mipCount),
	};
	auto scene_gpu = gpuHostToDevicePointer(app.queue, scene);
	auto indices_gpu = gpuHostToDevicePointer(app.queue, indices);
	auto sky_indices_gpu = gpuHostToDevicePointer(app.queue, sky_indices);

	gpuSyncMemoryEXT(app.queue, scene->vertices);
	gpuSyncMemoryEXT(app.queue, indices_gpu);
	gpuSyncMemoryEXT(app.queue, sky_indices_gpu);

	//
	// Two pipelines over one render pass. Both have to name the depth format the pass will bind,
	// even the sky, which neither tests nor writes it.
	//
	GpuColorTarget target { .format = app.surface_format };
	GpuRasterDesc raster {
		.cull = CULL_NONE, // The teapot's lid and spout are open surfaces; see the fragment shader
		.depthFormat = FORMAT_D32_FLOAT,
		.colorTargets = {&target, 1},
	};

	auto sky_pipeline = gpuCreateGraphicsPipeline(app.queue, string_to_bytes(slang_sky), string_to_bytes(slang_sky), raster);
	assert(sky_pipeline && "Sky pipeline creation failed");
	auto teapot_pipeline = gpuCreateGraphicsPipeline(app.queue, string_to_bytes(slang_teapot), string_to_bytes(slang_teapot), raster);
	assert(teapot_pipeline && "Teapot pipeline creation failed");

	// Depth/stencil state is separate from the pipeline in this API, so one pipeline pair covers
	// both of these and switching between them costs no recompile on a backend with the hardware
	// for it
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

		//
		// The camera orbits so that the reflections move, which is the only way to see that they
		// are the environment and not a texture painted on the teapot.
		//
		constexpr float fov = 0.9f;
		auto angle = float(now) * 0.3f;
		vec3 eye {std::cos(angle) * 3.2f, 1.1f, std::sin(angle) * 3.2f};
		vec3 target {0.0f, 0.0f, 0.0f};

		auto forward = normalize(target - eye);
		auto right = normalize(cross(forward, vec3{0, 1, 0}));
		auto up = cross(right, forward);

		auto view_projection = perspective(fov, app.aspect(), 0.1f, 100.0f) * look_at(eye, target, {0, 1, 0});
		memcpy(scene->view_projection, view_projection.m, sizeof(scene->view_projection));

		scene->camera_position[0] = eye.x; scene->camera_position[1] = eye.y; scene->camera_position[2] = eye.z;
		scene->camera_right[0] = right.x; scene->camera_right[1] = right.y; scene->camera_right[2] = right.z;
		scene->camera_up[0] = up.x; scene->camera_up[1] = up.y; scene->camera_up[2] = up.z;
		scene->camera_forward[0] = forward.x; scene->camera_forward[1] = forward.y; scene->camera_forward[2] = forward.z;

		// Half the film's extent at unit distance, which is what turns a clip space position back
		// into a view ray in the sky shader
		auto tan_half = std::tan(fov * 0.5f);
		scene->ray_scale[0] = tan_half * app.aspect();
		scene->ray_scale[1] = tan_half;

		scene->roughness = material.roughness;
		scene->metallic = material.metallic;
		scene->exposure = material.exposure;

		// All three record copies, so they come before the pass opens
		gpuSyncMemoryEXT(cmd, scene_gpu);
		gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
		gpuSetEnabledSamplersEXT(cmd, {&ENVIRONMENT_SAMPLER, 1});

		// The sky covers every pixel, so the colour attachment never needs clearing; the depth
		// buffer does, because the sky doesn't write it
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
