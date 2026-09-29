#pragma once


#include <algorithm>
#include <array>
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

struct mat4 {
	std::array<std::array<float, 4>, 4> m = {};

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

inline mat4 look_at(vec3 eye, vec3 target, vec3 up) {
	auto forward = normalize(target - eye);
	auto right = normalize(cross(forward, up));
	auto true_up = cross(right, forward);

	mat4 out = mat4::identity();
	out.m[0][0] = right.x;   out.m[0][1] = right.y;   out.m[0][2] = right.z;   out.m[0][3] = -dot(right, eye);
	out.m[1][0] = true_up.x; out.m[1][1] = true_up.y; out.m[1][2] = true_up.z; out.m[1][3] = -dot(true_up, eye);
	out.m[2][0] = -forward.x; out.m[2][1] = -forward.y; out.m[2][2] = -forward.z; out.m[2][3] = dot(forward, eye);
	return out;
}

inline mat4 perspective(float fov_y_radians, float aspect, float near, float far) {
	auto focal = 1.0f / std::tan(fov_y_radians * 0.5f);

	mat4 out;
	out.m[0][0] = focal / aspect;
	out.m[1][1] = focal;
	out.m[2][2] = far / (near - far);
	out.m[2][3] = (near * far) / (near - far);
	out.m[3][2] = -1.0f;
	return out;
}


inline gpu* device_pointer(GpuQueue* queue, void* ptr) {
	if(auto device = gpuHostToDevicePointer(queue, ptr)) return device;
	return (gpu*)ptr;
}

struct Texture {
	GpuTexture* texture = nullptr;
	gpu* memory = nullptr;
	GpuTextureDesc desc = {};
};

inline uint32_t mip_count_for(uint32_t width, uint32_t height) {
	uint32_t count = 1;
	while(width > 1 || height > 1) {
		width = width > 1 ? width / 2 : 1;
		height = height > 1 ? height / 2 : 1;
		++count;
	}
	return count;
}

inline Texture create_texture(GpuQueue* queue, GpuTextureDesc desc) {
	auto size = gpuTextureSizeAlign(queue, desc);
	auto memory = device_pointer(queue, gpuMalloc(queue, size, MEMORY_TEXTURE));
	auto texture = gpuCreateTexture(queue, desc, memory);
	assert(texture && "Texture creation failed");
	return {texture, memory, desc};
}

inline void blit_mip_chain(GpuCommandBuffer* cmd, const Texture& texture) {
	for(uint32_t mip = 1; mip < texture.desc.mipCount; ++mip) {
		gpuBlitTextureEXT(cmd, texture.texture, texture.texture, true, mip, 0, mip - 1, 0);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_PIXEL_SHADER);
	}
}

inline void upload_texture(GpuQueue* queue, const Texture& texture, std::span<const std::byte> pixels) {
	auto staging = gpuMalloc(queue, pixels.size(), 256, MEMORY_DEFAULT);
	assert(staging && "Staging allocation for a texture upload failed");
	memcpy(staging, pixels.data(), pixels.size());
	auto staging_gpu = gpuHostToDevicePointer(queue, staging);

	auto cmd = gpuStartCommandRecording(queue);
	gpuSyncMemoryEXT(cmd, staging_gpu);
	gpuCopyToTexture(cmd, texture.memory, staging_gpu, texture.texture);
	gpuBarrier(cmd, STAGE_TRANSFER, STAGE_PIXEL_SHADER);
	blit_mip_chain(cmd, texture);

	auto submission = gpuSubmit(queue, {&cmd, 1});
	gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), submission);
	gpuFree(queue, staging);
}

template<int CHANNELS, typename T>
std::vector<T> resample(std::span<const T> src, uint32_t src_width, uint32_t src_height, uint32_t dst_width, uint32_t dst_height ) {
	assert(src.size() >= size_t(src_width) * src_height * CHANNELS);
	std::vector<T> out(size_t(dst_width) * dst_height * CHANNELS);

	for(uint32_t y = 0; y < dst_height; ++y) {
		auto y0 = y * src_height / dst_height;
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
				if constexpr(std::is_integral_v<T>) averaged += 0.5f;
				out[(size_t(y) * dst_width + x) * CHANNELS + c] = T(averaged);
			}
		}
	}
	return out;
}

inline uint16_t float_to_half(float value) {
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));

	auto sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
	auto exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127;
	auto mantissa = bits & 0x7fffffu;

	if(exponent > 15) return sign | 0x7bffu;
	if(exponent < -14) return sign;
	return static_cast<uint16_t>(sign | ((exponent + 15) << 10) | (mantissa >> 13));
}


struct App {
	GLFWwindow* window = nullptr;
	BackendDefault backend = {};
	GpuQueue* queue = nullptr;
	GpuSurface* surface = nullptr;

	uint32_t width = 0, height = 0;
	FORMAT surface_format = FORMAT_NONE;

	FORMAT depth_format = FORMAT_NONE;
	Texture depth = {};

	float aspect() const { return height ? float(width) / float(height) : 1.0f; }
};

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

inline void create_device(App& app) {
	app.backend = setup_backend(app.window);
	app.queue = gpuCreateQueue(app.backend);

	app.surface = gpuCreateSurfaceEXT(app.queue, app.backend, GpuSurfaceDescriptor{
		.texture = {
			.dimensions = {app.width, app.height, 1},
			.format = FORMAT_NONE,
			.usage = USAGE_COLOR_ATTACHMENT,
		},
		.presentMode = PRESENT_MODE_BEST_AVAILABLE,
	});
	assert(app.surface && "Surface creation failed");

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
	inline void resize(App& app) {
		auto desc = gpuSurfaceGetConfigurationEXT(app.surface);
		desc.texture.dimensions = {app.width, app.height, 1};
		gpuSurfaceReconfigureEXT(app.queue, app.surface, desc);

		if(app.depth_format == FORMAT_NONE) return;
		if(app.depth.texture && app.depth.desc.dimensions.x == app.width && app.depth.desc.dimensions.y == app.height)
			return;

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

	using FrameFn = std::function_ref<void(App&, GpuCommandBuffer*, const GpuTexture*)>;

	inline void frame(App& app, const FrameFn& draw) {
		if(app.width == 0 || app.height == 0) return;

		auto configured = gpuSurfaceGetConfigurationEXT(app.surface).texture.dimensions;
		if(configured.x != app.width || configured.y != app.height || (app.depth_format != FORMAT_NONE && !app.depth.texture))
			resize(app);

		auto target = gpuSurfaceNextTextureEXT(app.queue, app.surface);
		if(!target) {
			resize(app);
			return;
		}

		auto cmd = gpuStartCommandRecording(app.queue);
		draw(app, cmd, target);
		auto submission = gpuSubmit(app.queue, {&cmd, 1});
		gpuSurfacePresentEXT(app.queue, app.surface, submission);
	}

	inline App* loop_app = nullptr;
	inline FrameFn loop_draw = [](App&, GpuCommandBuffer*, const GpuTexture*) {};
}

inline void run(App& app, detail::FrameFn draw, const std::function_ref<void(App&)>& teardown = [](App&) {}) {
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
	teardown(app);

	if(app.depth.texture) gpuFree(app.queue, app.depth.memory);
	gpuFreeSurfaceEXT(app.queue, app.surface);
	gpuFreeQueue(app.queue);
	shutdown_backend(app.backend);

	glfwDestroyWindow(app.window);
	glfwTerminate();
#endif
}

}


#ifdef _WIN32
	#include <windows.h>

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
	#define TEST_MAIN(...) int main() __VA_ARGS__
#endif
