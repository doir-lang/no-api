#include "common.hpp"

using namespace test;

static constexpr uint32_t TRACE_GROUP = 8;

static constexpr float GROUND_HEIGHT = -1.0f;

static const vec3 SUN_DIRECTION = normalize(vec3{0.6f, 0.8f, 0.35f});


static const std::string slang_trace = R"slang(
import noapi;

static const float T_MIN = 1e-3;
static const float T_MAX = 1e4;

static const float3 LIGHT_COLOR = float3(1.0, 0.96, 0.88) * 2.6;
static const float3 AMBIENT = float3(0.16, 0.20, 0.28);

static const float3 GROUND_LIGHT = float3(0.72, 0.72, 0.74);
static const float3 GROUND_DARK = float3(0.14, 0.15, 0.17);
static const float GROUND_REFLECTIVITY = 0.06;

struct Sphere : IGpuLoadable {
	float3 center;
	float radius;
	float3 albedo;
	float reflectivity;

	static const uint gpuStride = 32;

	static Sphere gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		Sphere out;
		out.center = from.next<float3>();
		out.radius = from.next<float>();
		out.albedo = from.next<float3>();
		out.reflectivity = from.next<float>();
		return out;
	}
}

struct TraceData : IGpuLoadable {
	GpuPtr<Sphere> spheres;
	uint sphereCount;
	uint target;
	uint2 size;
	float2 rayScale;
	float3 cameraPosition;
	float exposure;
	float3 cameraForward;
	uint bounces;
	float3 lightDirection;
	float groundHeight;

	static const uint gpuStride = 80;

	static TraceData gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		TraceData out;
		out.spheres = from.next<GpuPtr<Sphere> >();
		out.sphereCount = from.next<uint>();
		out.target = from.next<uint>();
		out.size = from.next<uint2>();
		out.rayScale = from.next<float2>();
		out.cameraPosition = from.next<float3>();
		out.exposure = from.next<float>();
		out.cameraForward = from.next<float3>();
		out.bounces = from.next<uint>();
		out.lightDirection = from.next<float3>();
		out.groundHeight = from.next<float>();
		return out;
	}
}

struct Hit {
	float distance;
	float3 position;
	float3 normal;
	float3 albedo;
	float reflectivity;
	float curvature;
}

float intersectSphere(float3 origin, float3 direction, Sphere sphere) {
	let oc = origin - sphere.center;
	let b = dot(oc, direction);
	let c = dot(oc, oc) - sphere.radius * sphere.radius;

	let discriminant = b * b - c;
	if(discriminant < 0.0) return -1.0;

	let root = sqrt(discriminant);
	let near = -b - root;
	if(near > T_MIN) return near;

	let far = -b + root;
	return far > T_MIN ? far : -1.0;
}

float intersectGround(float3 origin, float3 direction, float height) {
	if(abs(direction.y) < 1e-6) return -1.0;

	let t = (height - origin.y) / direction.y;
	return t > T_MIN ? t : -1.0;
}

float2 checkerIntegral(float2 x) { return abs(frac(x * 0.5) - 0.5); }

float3 groundAlbedo(float2 p, float2 w) {
	let i = 2.0 * (checkerIntegral(p - 0.5 * w) - checkerIntegral(p + 0.5 * w)) / max(w, 1e-6);
	return lerp(GROUND_DARK, GROUND_LIGHT, 0.5 - 0.5 * i.x * i.y);
}

float2 planeFootprint(float3 direction, float width) {
	let flat = direction.xz;
	let len = length(flat);
	if(len < 1e-6) return float2(width);

	let axis = flat / len;
	let major = width / max(abs(direction.y), 1e-4);
	return sqrt(float2(
		major * major * axis.x * axis.x + width * width * axis.y * axis.y,
		major * major * axis.y * axis.y + width * width * axis.x * axis.x));
}

bool closestHit(TraceData data, float3 origin, float3 direction, float coneWidth, float coneSpread, out Hit hit) {
	var nearest = T_MAX;
	var found = false;

	hit.distance = T_MAX;
	hit.position = origin;
	hit.normal = float3(0.0, 1.0, 0.0);
	hit.albedo = float3(0.0);
	hit.reflectivity = 0.0;
	hit.curvature = 0.0;

	for(uint i = 0u; i < data.sphereCount; ++i) {
		let sphere = data.spheres[i];
		let t = intersectSphere(origin, direction, sphere);
		if(t < 0.0 || t >= nearest) continue;

		nearest = t;
		found = true;
		hit.distance = t;
		hit.position = origin + direction * t;
		hit.normal = normalize(hit.position - sphere.center);
		hit.albedo = sphere.albedo;
		hit.reflectivity = sphere.reflectivity;
		hit.curvature = 1.0 / sphere.radius;
	}

	let ground = intersectGround(origin, direction, data.groundHeight);
	if(ground >= 0.0 && ground < nearest) {
		nearest = ground;
		found = true;
		let position = origin + direction * ground;

		hit.distance = ground;
		hit.position = position;
		hit.normal = float3(0.0, 1.0, 0.0);
		hit.albedo = groundAlbedo(position.xz, planeFootprint(direction, coneWidth + coneSpread * ground));
		hit.reflectivity = GROUND_REFLECTIVITY;
		hit.curvature = 0.0;
	}

	return found;
}

bool occluded(TraceData data, float3 origin) {
	for(uint i = 0u; i < data.sphereCount; ++i)
		if(intersectSphere(origin, data.lightDirection, data.spheres[i]) > 0.0) return true;
	return false;
}

float3 sky(TraceData data, float3 direction) {
	let gradient = lerp(float3(0.58, 0.68, 0.82), float3(0.16, 0.32, 0.62), saturate(direction.y * 0.5 + 0.5));
	let sun = pow(saturate(dot(direction, data.lightDirection)), 256.0) * 12.0;
	return gradient + LIGHT_COLOR * sun;
}

float3 shade(TraceData data, Hit hit) {
	let facing = saturate(dot(hit.normal, data.lightDirection));
	let lit = facing > 0.0 && !occluded(data, hit.position + hit.normal * 1e-3);
	return hit.albedo * (AMBIENT + (lit ? LIGHT_COLOR * facing : float3(0.0)));
}

float3 trace(TraceData data, float3 origin, float3 direction, float spread) {
	var radiance = float3(0.0);
	var throughput = float3(1.0);
	var from = origin;
	var along = direction;
	var width = 0.0;
	var cone = spread;

	for(uint bounce = 0u; bounce < data.bounces; ++bounce) {
		Hit hit;
		if(!closestHit(data, from, along, width, cone, hit)) {
			radiance += throughput * sky(data, along);
			break;
		}

		width += cone * hit.distance;
		cone += 2.0 * width * hit.curvature;

		radiance += throughput * shade(data, hit) * (1.0 - hit.reflectivity);

		throughput *= hit.albedo * hit.reflectivity;
		if(max(throughput.x, max(throughput.y, throughput.z)) < 1e-3) break;

		from = hit.position + hit.normal * 1e-3;
		along = reflect(along, hit.normal);
	}

	return radiance;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 thread : SV_DispatchThreadID) {
	let data = *gpuComputeData<TraceData>();
	if(any(thread.xy >= data.size)) return;

	let ndc = ((float2(thread.xy) + 0.5) / float2(data.size)) * 2.0 - 1.0;
	let right = normalize(cross(data.cameraForward, float3(0.0, 1.0, 0.0)));
	let up = cross(right, data.cameraForward);
	let raw = data.cameraForward
		+ right * (ndc.x * data.rayScale.x)
		- up * (ndc.y * data.rayScale.y);

	let invLength = 1.0 / length(raw);
	let direction = raw * invLength;

	let alongX = (right * (2.0 * data.rayScale.x / float(data.size.x))) * invLength;
	let alongY = (up * (2.0 * data.rayScale.y / float(data.size.y))) * invLength;
	let spread = max(
		length(alongX - direction * dot(direction, alongX)),
		length(alongY - direction * dot(direction, alongY))
	);

	let radiance = trace(data, data.cameraPosition, direction, spread) * data.exposure;

	let mapped = radiance / (radiance + 1.0);
	let encoded = pow(saturate(mapped), float3(1.0 / 2.2));

	gpuStoreTexture2D(data.target, thread.xy, float4(encoded, 1.0));
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
	uint image;

	static const uint gpuStride = 16;

	static RectData gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		RectData out;
		out.vertices = from.next<GpuPtr<Vertex> >();
		out.image = from.next<uint>();
		return out;
	}
}

struct Varyings {
	float4 position : SV_Position;
	float2 uv : TEXCOORD0;
}

GpuSamplerDesc imageSampler() {
	return GpuSamplerDesc(ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP,
		FILTER_LINEAR, FILTER_LINEAR, FILTER_LINEAR);
}

[shader("vertex")]
Varyings vertexMain(uint index : SV_VertexID) {
	let data = *gpuVertexData<RectData>();
	let vertex = data.vertices[index];

	Varyings output;
	output.position = float4(vertex.position, 0.0, 1.0);
	output.uv = vertex.uv;
	return output;
}

[shader("fragment")]
float4 fragmentMain(Varyings varyings) : SV_Target {
	let data = *gpuFragmentData<RectData>();
	return gpuSample2D(data.image, gpuGetSamplerIndex(imageSampler()), varyings.uv);
}
)slang";


struct Sphere {
	std::array<float, 3> center;
	float radius;
	std::array<float, 3> albedo;
	float reflectivity;
};
static_assert(sizeof(Sphere) == 32, "Sphere::gpuStride in the shader has to match");

struct TraceData {
	gpu* spheres;
	uint32_t sphere_count;
	uint32_t target;
	std::array<uint32_t, 2> size;
	std::array<float, 2> ray_scale;
	std::array<float, 3> camera_position;
	float exposure;
	std::array<float, 3> camera_forward;
	uint32_t bounces;
	std::array<float, 3> light_direction;
	float ground_height;
};
static_assert(sizeof(TraceData) == 80, "TraceData::gpuStride in the shader has to match");

struct Vertex {
	float x, y;
	float u, v;
};

struct RectData {
	gpu* vertices;
	uint32_t image;
	uint32_t _padding;
};
static_assert(sizeof(RectData) == 16, "RectData::gpuStride in the shader has to match");

static constexpr GpuSamplerDesc IMAGE_SAMPLER {
	.address_mode_u = ADDRESS_MODE_CLAMP,
	.address_mode_v = ADDRESS_MODE_CLAMP,
	.address_mode_w = ADDRESS_MODE_CLAMP,
};

static constexpr uint32_t IMAGE_SAMPLED = 0;
static constexpr uint32_t IMAGE_STORAGE = 1;

static constexpr std::array<Sphere, 4> SCENE {{
	{.center = {-1.3f, -0.35f, -0.4f}, .radius = 0.65f, .albedo = {0.85f, 0.32f, 0.28f}, .reflectivity = 0.08f},
	{.center = { 0.0f,  0.00f,  0.0f}, .radius = 1.00f, .albedo = {0.95f, 0.95f, 0.97f}, .reflectivity = 0.92f},
	{.center = { 1.5f, -0.45f, -0.7f}, .radius = 0.55f, .albedo = {0.30f, 0.70f, 0.45f}, .reflectivity = 0.10f},
	{.center = { 0.5f, -0.55f,  1.4f}, .radius = 0.45f, .albedo = {0.90f, 0.75f, 0.35f}, .reflectivity = 0.45f},
}};


static Texture create_trace_image(App& app, GpuTextureDescriptor* heap, gpu* heap_gpu) {
	auto image = create_texture(app.queue, GpuTextureDesc{
		.dimensions = {app.width, app.height, 1},
		.format = FORMAT_RGBA8_UNORM,
		.usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_STORAGE),
	});

	heap[IMAGE_SAMPLED] = gpuTextureViewDescriptor(app.queue, image.texture, GpuViewDesc{});
	heap[IMAGE_STORAGE] = gpuRWTextureViewDescriptor(app.queue, image.texture, GpuViewDesc{.mipCount = 1});
	gpuSyncMemoryEXT(app.queue, heap_gpu);

	return image;
}

struct Controls {
	uint32_t bounces = 3;
	float exposure = 1.0f;
	bool up_held = false, down_held = false;
};

static void poll_controls(GLFWwindow* window, Controls& controls, float dt) {
	bool up_down = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
	if(up_down && !controls.up_held && controls.bounces < 8) ++controls.bounces;
	controls.up_held = up_down;

	bool down_down = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS;
	if(down_down && !controls.down_held && controls.bounces > 1) --controls.bounces;
	controls.down_held = down_down;

	if(glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) controls.exposure *= 1.0f + dt;
	if(glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) controls.exposure /= 1.0f + dt;
	controls.exposure = std::clamp(controls.exposure, 0.05f, 20.0f);
}

TEST_MAIN({
	App app;
	open_window(app, "NoAPI Ray Tracing");
	create_device(app);

	auto heap = gpuMalloc<GpuTextureDescriptor>(app.queue, 2, MEMORY_DESCRIPTOR_HEAP);
	auto heap_gpu = gpuHostToDevicePointer(app.queue, heap);

	auto image = create_trace_image(app, heap, heap_gpu);
	std::println("image: {}x{}, {} mip level, retraced every frame", app.width, app.height, image.desc.mipCount);

	auto spheres = gpuMalloc<Sphere>(app.queue, SCENE.size());
	memcpy(spheres, SCENE.data(), SCENE.size() * sizeof(Sphere));
	auto spheres_gpu = gpuHostToDevicePointer(app.queue, spheres);

	auto trace = gpuMalloc<TraceData>(app.queue);
	*trace = TraceData{
		.spheres = spheres_gpu,
		.sphere_count = uint32_t(SCENE.size()),
		.target = IMAGE_STORAGE,
		.light_direction = {SUN_DIRECTION.x, SUN_DIRECTION.y, SUN_DIRECTION.z},
		.ground_height = GROUND_HEIGHT,
	};
	auto trace_gpu = gpuHostToDevicePointer(app.queue, trace);

	gpuSyncMemoryEXT(app.queue, spheres_gpu);
	gpuSyncMemoryEXT(app.queue, trace_gpu);

	auto vertices = gpuMalloc<Vertex>(app.queue, 4);
	vertices[0] = {-1.0f,  1.0f, 0.0f, 0.0f};
	vertices[1] = { 1.0f,  1.0f, 1.0f, 0.0f};
	vertices[2] = { 1.0f, -1.0f, 1.0f, 1.0f};
	vertices[3] = {-1.0f, -1.0f, 0.0f, 1.0f};

	auto indices = gpuMalloc<uint32_t>(app.queue, 6);
	constexpr std::array<uint32_t, 6> quad = {0, 1, 2, 0, 2, 3};
	memcpy(indices, quad.data(), sizeof(quad));

	auto rect = gpuMalloc<RectData>(app.queue);
	*rect = RectData{
		.vertices = gpuHostToDevicePointer(app.queue, vertices),
		.image = IMAGE_SAMPLED,
	};
	auto rect_gpu = gpuHostToDevicePointer(app.queue, rect);
	auto indices_gpu = gpuHostToDevicePointer(app.queue, indices);

	gpuSyncMemoryEXT(app.queue, rect->vertices);
	gpuSyncMemoryEXT(app.queue, indices_gpu);
	gpuSyncMemoryEXT(app.queue, rect_gpu);

	auto tracer = gpuCreateComputePipeline(app.queue, string_to_bytes(slang_trace));
	assert(tracer && "Tracing pipeline creation failed");

	GpuColorTarget target { .format = app.surface_format };
	auto pipeline = gpuCreateGraphicsPipeline(app.queue, string_to_bytes(slang_rectangle), string_to_bytes(slang_rectangle), GpuRasterDesc{
		.colorTargets = {&target, 1},
	});
	assert(pipeline && "Pipeline creation failed");

	constexpr float fov = 0.9f;

	Controls controls;
	double last_time = glfwGetTime();

	std::println("scene: {} spheres over a plane at y = {}", SCENE.size(), GROUND_HEIGHT);
	std::println("controls: up/down bounces, [ ] exposure");

	run(app, [&](App& app, GpuCommandBuffer* cmd, const GpuTexture* backbuffer) {
		auto now = glfwGetTime();
		auto dt = float(now - last_time);
		last_time = now;
		poll_controls(app.window, controls, dt);

		bool resized = image.desc.dimensions.x != app.width || image.desc.dimensions.y != app.height;
		if(resized) {
			gpuWaitIdleEXT(app.queue);
			gpuFree(app.queue, image.memory);
			image = create_trace_image(app, heap, heap_gpu);
		}

		auto angle = float(now) * 0.25f;
		vec3 eye {std::cos(angle) * 4.6f, 1.4f, std::sin(angle) * 4.6f};
		auto forward = normalize(vec3{0.0f, 0.0f, 0.0f} - eye);

		trace->camera_position = {eye.x, eye.y, eye.z};
		trace->camera_forward = {forward.x, forward.y, forward.z};

		trace->size = {app.width, app.height};
		auto tan_half = std::tan(fov * 0.5f);
		trace->ray_scale = {tan_half * app.aspect(), tan_half};

		trace->bounces = controls.bounces;
		trace->exposure = controls.exposure;

		gpuSyncMemoryEXT(cmd, trace_gpu);
		gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
		gpuSetEnabledSamplersEXT(cmd, {&IMAGE_SAMPLER, 1});

		if(resized) gpuBarrier(cmd, STAGE_ALL, STAGE_COMPUTE, HAZARD_DESCRIPTORS);

		gpuSetPipeline(cmd, tracer);
		gpuDispatch(cmd, trace_gpu, {
			(app.width + TRACE_GROUP - 1) / TRACE_GROUP,
			(app.height + TRACE_GROUP - 1) / TRACE_GROUP,
			1,
		});
		gpuBarrier(cmd, STAGE_COMPUTE, STAGE_PIXEL_SHADER);

		GpuColorAttachment color {
			.texture = backbuffer,
			.loadOp = LOAD_OP_DONT_CARE,
		};
		GpuRenderPassDesc pass { .colorAttachments = {&color, 1} };

		gpuBeginRenderPass(cmd, pass);
		gpuSetPipeline(cmd, pipeline);
		gpuDrawIndexedInstanced(cmd, rect_gpu, rect_gpu, indices_gpu, 6, 1);
		gpuEndRenderPass(cmd, pass);

		gpuBarrier(cmd, STAGE_PIXEL_SHADER, STAGE_COMPUTE);
	}, [&](App& app) {
		gpuFreePipeline(app.queue, pipeline);
		gpuFreePipeline(app.queue, tracer);
		gpuFree(app.queue, image.memory);
		gpuFree(app.queue, heap);
		gpuFree(app.queue, spheres);
		gpuFree(app.queue, trace);
		gpuFree(app.queue, vertices);
		gpuFree(app.queue, indices);
		gpuFree(app.queue, rect);
	});

	return 0;
})
