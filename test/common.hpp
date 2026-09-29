#pragma once

/**
 * Shared harness for the tests past 0_basic.
 *
 * 0_basic deliberately stands alone: it is the file to read to see the whole flow from an empty
 * process to a presented frame, with nothing hidden behind a helper. Everything here is the
 * second copy of that same flow, factored out so that the tests which follow can be about the
 * thing they are testing -- sampling a texture, shading a mesh -- rather than about glfw and
 * surface configuration.
 *
 * What it provides:
 * - the backend setup/shutdown pair, per backend, exactly as 0_basic spells it out
 * - App: window, queue, surface, an optional depth buffer, and a frame loop that keeps all
 *   three sized to the window
 * - a little row major vector/matrix math, laid out the way the shader side loads it
 * - create_texture / upload_texture, including the mip chain, which is not a copy (see there)
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef NOAPI_BACKEND_VULKAN
	#include <volk.h>
	#define GLFW_INCLUDE_VULKAN
#endif

#include <GLFW/glfw3.h>

#ifdef NOAPI_BACKEND_VULKAN
	#include <vulkan/noapi.hpp>
#else
	#include <webgpu/noapi.hpp>
	#include "glfw3webgpu.h"
#endif

#ifdef __EMSCRIPTEN__
	#include <emscripten/emscripten.h>
#endif

#ifndef TEST_ASSET_DIR
	#define TEST_ASSET_DIR "."
#endif

namespace test {

// ---------------------------------------------------------------------------
// Backend setup
// ---------------------------------------------------------------------------

#ifdef NOAPI_BACKEND_VULKAN

	using BackendDefault = GpuVulkanDefault;

	inline BackendDefault setup_backend(GLFWwindow* window) {
		auto vk = gpuSetupDefaultVulkanEXT([window](VkInstance instance) -> VkSurfaceKHR {
			VkSurfaceKHR surface;
			if(glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
				throw std::runtime_error("Failed to setup Vulkan surface");
			return surface;
		});
		if(!vk) throw std::runtime_error(vk.error());
		return *vk;
	}

	inline void shutdown_backend(BackendDefault& vulkan) {
		vkDestroyDevice(vulkan.device, nullptr);
		vkDestroySurfaceKHR(vulkan.instance, vulkan.surface, nullptr);
		if(vulkan.messenger) vkDestroyDebugUtilsMessengerEXT(vulkan.instance, vulkan.messenger, nullptr);
		vkDestroyInstance(vulkan.instance, nullptr);
		volkFinalize();
	}

	inline constexpr const char* backend_name = "vulkan";

#else

	using BackendDefault = GpuWebGPUDefault;

	inline BackendDefault setup_backend(GLFWwindow* window) {
		auto wg = gpuSetupDefaultWebGPUEXT([window](WGPUInstance instance) -> WGPUSurface {
			auto surface = glfwCreateWindowWGPUSurface(instance, window);
			if(!surface) throw std::runtime_error("Failed to setup WebGPU surface");
			return surface;
		});
		if(!wg) throw std::runtime_error(wg.error());
		return *wg;
	}

	inline void shutdown_backend(BackendDefault& wgpu) {
		wgpuSurfaceRelease(wgpu.surface);
		wgpuDeviceRelease(wgpu.device);
		wgpuAdapterRelease(wgpu.adapter);
		wgpuInstanceRelease(wgpu.instance);
	}

	inline constexpr const char* backend_name = "webgpu";

#endif

// ---------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------

struct vec3 {
	float x = 0, y = 0, z = 0;

	constexpr vec3() = default;
	constexpr vec3(float x, float y, float z): x(x), y(y), z(z) {}
	constexpr explicit vec3(float all): x(all), y(all), z(all) {}
};

constexpr vec3 operator+(vec3 a, vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr vec3 operator-(vec3 a, vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
constexpr vec3 operator*(vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
constexpr vec3 operator-(vec3 a) { return {-a.x, -a.y, -a.z}; }
constexpr vec3& operator+=(vec3& a, vec3 b) { a = a + b; return a; }

constexpr float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

constexpr vec3 cross(vec3 a, vec3 b) {
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline float length(vec3 a) { return std::sqrt(dot(a, a)); }

inline vec3 normalize(vec3 a) {
	auto l = length(a);
	return l > 0 ? a * (1.0f / l) : a;
}

/**
 * mat4 -- A 4x4 matrix stored row major, which is the layout the shader side reads.
 *
 * The portable Slang module loads a matrix as its rows back to back (see IGpuLoadable), and
 * Slang's mul(M, v) dots each row of M with v, so a matrix written out row by row here arrives
 * as the same logical matrix and transforms in the obvious direction. A column major host
 * matrix would arrive transposed.
 */
struct mat4 {
	float m[4][4] = {};

	static constexpr mat4 identity() {
		mat4 out;
		for(int i = 0; i < 4; ++i) out.m[i][i] = 1.0f;
		return out;
	}
};

constexpr mat4 operator*(const mat4& a, const mat4& b) {
	mat4 out;
	for(int r = 0; r < 4; ++r)
		for(int c = 0; c < 4; ++c)
			for(int k = 0; k < 4; ++k)
				out.m[r][c] += a.m[r][k] * b.m[k][c];
	return out;
}

/**
 * look_at -- A right handed view matrix looking from \p eye towards \p target.
 *
 * @param eye Camera position in world space.
 * @param target Point the camera is aimed at.
 * @param up Rough world up; the basis is re-orthogonalized against the view direction.
 */
inline mat4 look_at(vec3 eye, vec3 target, vec3 up) {
	auto forward = normalize(target - eye);
	auto right = normalize(cross(forward, up));
	auto true_up = cross(right, forward);

	mat4 out = mat4::identity();
	out.m[0][0] = right.x;   out.m[0][1] = right.y;   out.m[0][2] = right.z;   out.m[0][3] = -dot(right, eye);
	out.m[1][0] = true_up.x; out.m[1][1] = true_up.y; out.m[1][2] = true_up.z; out.m[1][3] = -dot(true_up, eye);
	// Right handed: the camera looks down -Z, so the view space Z axis is the negated forward
	out.m[2][0] = -forward.x; out.m[2][1] = -forward.y; out.m[2][2] = -forward.z; out.m[2][3] = dot(forward, eye);
	return out;
}

/**
 * perspective -- A projection matrix mapping view space onto a [0, 1] depth range.
 *
 * No backend dependent Y flip: Vulkan's clip space does point the opposite way down Y to
 * WebGPU's, but the shader compiler already squares that away by giving the SPIR-V target
 * VulkanInvertY (see slang_compiler.hpp), so the same clip position lands the same way up on
 * both. Adding a sign here would be a second flip undoing the first.
 *
 * @param fov_y_radians Vertical field of view.
 * @param aspect Viewport width divided by its height.
 * @param near Near plane distance (positive).
 * @param far Far plane distance (positive, greater than \p near).
 */
inline mat4 perspective(float fov_y_radians, float aspect, float near, float far) {
	auto focal = 1.0f / std::tan(fov_y_radians * 0.5f);

	mat4 out;
	out.m[0][0] = focal / aspect;
	out.m[1][1] = focal;
	// Zero to one depth, which is what both backends' clip space uses
	out.m[2][2] = far / (near - far);
	out.m[2][3] = (near * far) / (near - far);
	out.m[3][2] = -1.0f;
	return out;
}

// ---------------------------------------------------------------------------
// Memory and textures
// ---------------------------------------------------------------------------

/**
 * device_pointer -- The gpu address of a gpuMalloc'd allocation, whichever kind of pointer the
 * backend handed back for it.
 *
 * MEMORY_DEFAULT is CPU mapped on both backends and so always needs translating. MEMORY_TEXTURE
 * is where the two disagree: Vulkan has no host mapping to translate and returns the device
 * address itself, while WebGPU keeps a CPU shadow copy of every allocation and returns that. A
 * failed translation therefore means the pointer already was a device address.
 *
 * @param queue The queue the memory was allocated on.
 * @param ptr Whatever gpuMalloc returned.
 */
inline gpu* device_pointer(GpuQueue* queue, void* ptr) {
	if(auto device = gpuHostToDevicePointer(queue, ptr)) return device;
	return (gpu*)ptr;
}

/**
 * Texture -- A texture and the allocation backing it, which is what the copy and descriptor
 * calls need to be handed together.
 */
struct Texture {
	GpuTexture* texture = nullptr;
	gpu* memory = nullptr; // What gpuCreateTexture was given, and what gpuCopyToTexture writes
	GpuTextureDesc desc = {};
};

/**
 * mip_count_for -- How many mip levels a texture of this size has, down to 1x1.
 */
inline uint32_t mip_count_for(uint32_t width, uint32_t height) {
	uint32_t count = 1;
	while(width > 1 || height > 1) {
		width = width > 1 ? width / 2 : 1;
		height = height > 1 ? height / 2 : 1;
		++count;
	}
	return count;
}

/**
 * create_texture -- Allocate memory sized for \p desc and create the texture on it.
 *
 * The two halves are separate in the API on purpose (see gpuTextureSizeAlign), and nothing here
 * ever wants one without the other.
 *
 * @param queue The queue to create on.
 * @param desc What to create. Its usage decides which monotexture a WebGPU backend files it under.
 */
inline Texture create_texture(GpuQueue* queue, GpuTextureDesc desc) {
	auto size = gpuTextureSizeAlign(queue, desc);
	auto memory = device_pointer(queue, gpuMalloc(queue, size, MEMORY_TEXTURE));
	auto texture = gpuCreateTexture(queue, desc, memory);
	assert(texture && "Texture creation failed");
	return {texture, memory, desc};
}

/**
 * upload_texture -- Fill mip 0 of \p texture from \p pixels, then fill the rest of its mip chain
 * by halving it, and wait for all of it to land.
 *
 * The mip chain is rasterized rather than copied: gpuCopyToTexture takes no mip level and only
 * ever writes level 0, so each smaller level is a gpuBlitTextureEXT sampling the level above it.
 * That is why a texture created for this has USAGE_COLOR_ATTACHMENT on top of USAGE_SAMPLED --
 * the blit renders into the destination -- and why each blit is separated from the next by a
 * barrier, since the level it writes is the level the next one samples.
 *
 * @param queue The queue the texture was created on.
 * @param texture Destination, from create_texture.
 * @param pixels Tightly packed level 0 texels, one image row per row, in the texture's format.
 */
inline void upload_texture(GpuQueue* queue, const Texture& texture, std::span<const std::byte> pixels) {
	auto staging = gpuMalloc(queue, pixels.size(), 256, MEMORY_DEFAULT);
	assert(staging && "Staging allocation for a texture upload failed");
	memcpy(staging, pixels.data(), pixels.size());
	auto staging_gpu = gpuHostToDevicePointer(queue, staging);

	auto cmd = gpuStartCommandRecording(queue);
	gpuSyncMemoryEXT(cmd, staging_gpu);
	gpuCopyToTexture(cmd, texture.memory, staging_gpu, texture.texture);
	gpuBarrier(cmd, STAGE_TRANSFER, STAGE_PIXEL_SHADER);

	for(uint32_t mip = 1; mip < texture.desc.mipCount; ++mip) {
		gpuBlitTextureEXT(cmd, texture.texture, texture.texture, true, mip, 0, mip - 1, 0);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_PIXEL_SHADER);
	}

	auto submission = gpuSubmit(queue, {&cmd, 1});
	gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), submission);
	gpuFree(queue, staging);
}

/**
 * resample -- Box filter an image onto a different grid, averaging over the source texels each
 * destination texel covers.
 *
 * The assets are far larger than anything worth sampling here, and on the WebGPU backend the size
 * of a texture decides which monotexture atlas it is filed under, rounded up to a power of two:
 * the 6160x4640 photo would ask for an 8192x8192 atlas with a full mip chain, hundreds of
 * megabytes for an image that is never shown at more than window resolution. Resampling to a
 * power of two also makes a texture exactly fill its slot in that atlas, so nothing here depends
 * on how the backend pads the leftovers.
 *
 * @tparam CHANNELS Components per texel, interleaved, in both images.
 * @param src Tightly packed source texels.
 * @param src_width Source width in texels.
 * @param src_height Source height in texels.
 * @param dst_width Destination width in texels.
 * @param dst_height Destination height in texels.
 */
template<int CHANNELS, typename T>
std::vector<T> resample(std::span<const T> src, uint32_t src_width, uint32_t src_height,
	uint32_t dst_width, uint32_t dst_height
) {
	assert(src.size() >= size_t(src_width) * src_height * CHANNELS);
	std::vector<T> out(size_t(dst_width) * dst_height * CHANNELS);

	for(uint32_t y = 0; y < dst_height; ++y) {
		auto y0 = y * src_height / dst_height;
		// At least one row, so that upsampling picks the nearest texel rather than averaging none
		auto y1 = std::max(y0 + 1, (y + 1) * src_height / dst_height);

		for(uint32_t x = 0; x < dst_width; ++x) {
			auto x0 = x * src_width / dst_width;
			auto x1 = std::max(x0 + 1, (x + 1) * src_width / dst_width);

			float sum[CHANNELS] = {};
			for(auto sy = y0; sy < y1; ++sy)
				for(auto sx = x0; sx < x1; ++sx)
					for(int c = 0; c < CHANNELS; ++c)
						sum[c] += float(src[(size_t(sy) * src_width + sx) * CHANNELS + c]);

			auto scale = 1.0f / float((y1 - y0) * (x1 - x0));
			for(int c = 0; c < CHANNELS; ++c) {
				auto averaged = sum[c] * scale;
				if constexpr(std::is_integral_v<T>) averaged += 0.5f; // Round rather than truncate
				out[(size_t(y) * dst_width + x) * CHANNELS + c] = T(averaged);
			}
		}
	}
	return out;
}

/**
 * float_to_half -- Pack a float into the IEEE binary16 a FORMAT_RGBA16_FLOAT texel holds.
 *
 * Half floats are the widest thing a sampler will filter for us: the 32 bit float formats are
 * not filterable (gpuFormatIsFilterableEXT), which rules them out for anything sampled with
 * FILTER_LINEAR or blitted down a mip chain. Subnormal results are flushed to zero, which
 * nothing here can tell apart from the real thing.
 *
 * @param value The float to pack.
 */
inline uint16_t float_to_half(float value) {
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));

	auto sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
	auto exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127;
	auto mantissa = bits & 0x7fffffu;

	if(exponent > 15) return sign | 0x7bffu; // Saturate rather than turn into an infinity
	if(exponent < -14) return sign; // Below the smallest normal half
	return static_cast<uint16_t>(sign | ((exponent + 15) << 10) | (mantissa >> 13));
}

// ---------------------------------------------------------------------------
// Window, surface and frame loop
// ---------------------------------------------------------------------------

/**
 * App -- The window, the queue, the surface and (when asked for) a depth buffer, kept sized to
 * the window by run().
 *
 * The surface is configured to the first non-sRGB format it offers, so a shader's output reaches
 * the screen as written and each test can decide for itself what space it is working in.
 */
struct App {
	GLFWwindow* window = nullptr;
	BackendDefault backend = {};
	GpuQueue* queue = nullptr;
	GpuSurface* surface = nullptr;

	uint32_t width = 0, height = 0;
	FORMAT surface_format = FORMAT_NONE;

	/**
	 * Depth buffer. Left absent unless depth_format is set before run(), in which case it is
	 * created and recreated to match the window.
	 */
	FORMAT depth_format = FORMAT_NONE;
	Texture depth = {};

	float aspect() const { return height ? float(width) / float(height) : 1.0f; }
};

/**
 * open_window -- Create the window and hook up the resize callback, before any gpu object exists.
 *
 * Shader diagnostics are routed here too, since a module registered or a pipeline compiled later
 * has no other way to report what went wrong.
 *
 * @param app Filled in with the window and its size.
 * @param title Window title; the backend name is appended.
 * @param width Initial window width.
 * @param height Initial window height.
 */
inline void open_window(App& app, const char* title, int width = 1280, int height = 720) {
	if(!glfwInit()) throw std::runtime_error("Glfw initialization failed");

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	auto full_title = std::string(title) + " (" + backend_name + ")";
	app.window = glfwCreateWindow(width, height, full_title.c_str(), nullptr, nullptr);
	if(!app.window) throw std::runtime_error("Failed to create glfw window");

	glfwSetWindowUserPointer(app.window, &app);
	glfwSetFramebufferSizeCallback(app.window, [](GLFWwindow* window, int width, int height) {
		auto& app = *(App*)glfwGetWindowUserPointer(window);
		app.width = uint32_t(width);
		app.height = uint32_t(height);
	});

	int fb_width, fb_height;
	glfwGetFramebufferSize(app.window, &fb_width, &fb_height);
	app.width = uint32_t(fb_width);
	app.height = uint32_t(fb_height);

	gpuSetShaderDiagnosticCallbackEXT([](GpuStringView message, void*) {
		std::println(std::cerr, "[shader] {}", std::string_view(message.ptr, message.count));
	}, nullptr);
}

/**
 * create_device -- Bring up the backend, the queue and the surface for an already opened window.
 *
 * Split from open_window because the shader modules a test registers have to be in place before
 * any pipeline is created, and pipeline creation needs the queue this makes.
 *
 * @param app Must already carry a window; filled in with the backend, queue and surface.
 */
inline void create_device(App& app) {
	app.backend = setup_backend(app.window);
	app.queue = gpuCreateQueue(app.backend);

	app.surface = gpuCreateSurfaceEXT(app.queue, app.backend, GpuSurfaceDescriptor{
		.texture = {
			.dimensions = {app.width, app.height, 1},
			.format = FORMAT_NONE, // Whatever the surface prefers, until the capabilities are known
			.usage = USAGE_COLOR_ATTACHMENT,
		},
		.presentMode = PRESENT_MODE_FIFO,
	});
	assert(app.surface && "Surface creation failed");

	// A linear format, so that what the shader writes is what gets shown and each test can do its
	// own tone mapping and encoding where it can be seen
	auto caps = gpuGetSurfaceCapabilitiesEXT(app.queue, app.surface);
	app.surface_format = gpuSurfaceGetConfigurationEXT(app.surface).texture.format;
	for(auto format: caps.formatList())
		if(!gpuFormatIsSrgbEXT(format)) {
			auto config = gpuSurfaceGetConfigurationEXT(app.surface);
			config.texture.format = format;
			gpuSurfaceReconfigureEXT(app.queue, app.surface, config);
			app.surface_format = format;
			break;
		}

	std::println("{}: {}x{}, surface format {}", backend_name, app.width, app.height, (int)app.surface_format);
}

namespace detail {
	/// Resize the surface, and the depth buffer with it, to the window's current size.
	inline void resize(App& app) {
		auto desc = gpuSurfaceGetConfigurationEXT(app.surface);
		desc.texture.dimensions = {app.width, app.height, 1};
		gpuSurfaceReconfigureEXT(app.queue, app.surface, desc);

		if(app.depth_format == FORMAT_NONE) return;
		if(app.depth.texture && app.depth.desc.dimensions.x == app.width && app.depth.desc.dimensions.y == app.height)
			return;

		// The old depth buffer may still be in flight, and freeing its memory is what destroys the
		// texture on it, so nothing may be reading it any more
		if(app.depth.texture) {
			gpuWaitIdleEXT(app.queue);
			gpuFree(app.queue, app.depth.memory);
		}

		app.depth = create_texture(app.queue, GpuTextureDesc{
			.dimensions = {app.width, app.height, 1},
			.format = app.depth_format,
			.usage = USAGE_DEPTH_STENCIL_ATTACHMENT,
		});
	}

	/// What run() hands each frame: the acquired backbuffer and a command buffer to record into.
	using FrameFn = std::function<void(App&, GpuCommandBuffer*, const GpuTexture*)>;

	inline void frame(App& app, const FrameFn& draw) {
		// A minimized window has nothing to present, and a surface can't be configured for a zero extent
		if(app.width == 0 || app.height == 0) return;

		auto configured = gpuSurfaceGetConfigurationEXT(app.surface).texture.dimensions;
		if(configured.x != app.width || configured.y != app.height || (app.depth_format != FORMAT_NONE && !app.depth.texture))
			resize(app);

		auto target = gpuSurfaceNextTextureEXT(app.queue, app.surface);
		if(!target) { // Out of date, lost or timed out: reconfigure and try again next frame
			resize(app);
			return;
		}

		auto cmd = gpuStartCommandRecording(app.queue);
		draw(app, cmd, target);
		auto submission = gpuSubmit(app.queue, {&cmd, 1});
		gpuSurfacePresentEXT(app.queue, app.surface, submission);
	}

	// Emscripten drives the loop by calling back into us, so the state it needs has to outlive the
	// call that starts it
	inline App* loop_app = nullptr;
	inline FrameFn loop_draw;
}

/**
 * run -- Drive frames until the window closes, then tear everything down.
 *
 * \p draw is handed the acquired backbuffer and an open command buffer, and is responsible for
 * the render passes it wants in it; the surface, the depth buffer, submission and presentation
 * are handled around it. Under Emscripten this never returns, and the teardown never runs,
 * because the browser owns the loop.
 *
 * @param app The app to drive.
 * @param draw Called once per frame with (app, command buffer, backbuffer).
 * @param teardown Called after the loop and before the queue is freed, to release what the test
 * itself created.
 */
inline void run(App& app, detail::FrameFn draw, const std::function<void(App&)>& teardown = {}) {
#ifdef __EMSCRIPTEN__
	detail::loop_app = &app;
	detail::loop_draw = std::move(draw);
	emscripten_set_main_loop([] {
		glfwPollEvents();
		detail::frame(*detail::loop_app, detail::loop_draw);
	}, 0, true);
#else
	while(!glfwWindowShouldClose(app.window)) {
		glfwPollEvents();
		detail::frame(app, draw);
	}

	gpuWaitIdleEXT(app.queue);
	if(teardown) teardown(app);

	if(app.depth.texture) gpuFree(app.queue, app.depth.memory);
	gpuFreeSurfaceEXT(app.queue, app.surface);
	gpuFreeQueue(app.queue);
	shutdown_backend(app.backend);

	glfwDestroyWindow(app.window);
	glfwTerminate();
#endif
}

} // namespace test

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/**
 * TEST_MAIN -- The platform entry point, so that each test only writes its body.
 *
 * A Windows GUI subsystem binary starts at WinMain and has no console attached, so one is made
 * before anything prints. Everywhere else this is just main.
 */
#ifdef _WIN32
	#include <windows.h>

	// Variadic because the body is full of commas at the top level, which a single parameter
	// would be split on
	#define TEST_MAIN(...) \
		static int test_main(); \
		int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) { \
			AllocConsole(); \
			freopen("CONOUT$", "w", stdout); \
			freopen("CONOUT$", "w", stderr); \
			freopen("CONIN$", "r", stdin); \
			return test_main(); \
		} \
		static int test_main() __VA_ARGS__
#else
	/// See the Windows flavour above for why this is variadic.
	#define TEST_MAIN(...) int main() __VA_ARGS__
#endif
