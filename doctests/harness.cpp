#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "harness.h"

#ifdef NOAPI_DOCTEST_WINDOW
	#include <GLFW/glfw3.h>
	#include "glfw3webgpu.h"
#endif

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <string>

static std::atomic<int> device_errors = 0;

static GpuDoctestBackend backend;
static GpuQueue* queue;

// Errors can arrive on a driver thread, where doctest cannot record them, so they are counted and checked once the example has finished.
static void count_error(void*, int type, GpuStringView message) {
	std::fprintf(stderr, "device error %d: %.*s\n", type, (int)message.size(), message.data());
	++device_errors;
}

GpuQueue* noapi_doctest_queue(void) {
	return queue;
}

GpuDoctestBackend noapi_doctest_backend(void) {
	return backend;
}

void noapi_doctest_begin(void) {
	device_errors = 0;
}

void noapi_doctest_end(void) {
	gpuWaitIdleEXT(queue);
	CHECK_MESSAGE(device_errors == 0, "the example raised device or validation errors (printed above)");
}

void noapi_doctest_fail(const char* file, int line, const char* expression) {
	ADD_FAIL_AT(file, line, (doctest::String("assert(") + expression + ") failed"));
}

#ifdef NOAPI_DOCTEST_WINDOW
static GLFWwindow* window = nullptr;
#endif

#ifdef NOAPI_BACKEND_WEBGPU
// WebGPU has no headless surface, so a hidden window, where there is a display, gives the surface examples something to present to.
static WGPUSurface create_window_surface([[maybe_unused]] WGPUInstance instance) {
#ifdef NOAPI_DOCTEST_WINDOW
	if(!(std::getenv("DISPLAY") || std::getenv("WAYLAND_DISPLAY")) || !glfwInit()) return nullptr;
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	window = glfwCreateWindow(640, 480, "noapi doctests", nullptr, nullptr);
	return window ? glfwCreateWindowWGPUSurface(instance, window) : nullptr;
#else
	return nullptr;
#endif
}
#endif

static bool setup() {
#ifdef NOAPI_BACKEND_VULKAN
	// A headless surface, where the driver offers one, gives the surface examples a swapchain to work with; without one they take their windowless branch.
	const char* headless[] = {"VK_KHR_surface", "VK_EXT_headless_surface"};
	auto create_headless = [](VkInstance instance) -> VkSurfaceKHR {
		auto create = (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT");
		VkHeadlessSurfaceCreateInfoEXT info {.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT};
		VkSurfaceKHR surface = VK_NULL_HANDLE;
		if(create && create(instance, &info, nullptr, &surface) != VK_SUCCESS) surface = VK_NULL_HANDLE;
		return surface;
	};
	if(volkInitialize() != VK_SUCCESS) return false;
	auto created = gpuSetupDefaultVulkanEXT(create_headless, count_error, VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, headless);
	if(!created) created = gpuSetupDefaultVulkanEXT([](VkInstance) -> VkSurfaceKHR { return VK_NULL_HANDLE; }, count_error);
#else
	auto created = gpuSetupDefaultWebGPUEXT(create_window_surface, count_error);
#endif
	if(!created) {
		std::fprintf(stderr, "doctest: could not create a headless device: %s\n", created.error().c_str());
		return false;
	}
	backend = *created;
	queue = gpuCreateQueue(backend);
	if(!queue) std::fprintf(stderr, "doctest: could not create a queue\n");
	return queue;
}

static void shutdown() {
	gpuFreeQueue(queue);
#ifdef NOAPI_BACKEND_VULKAN
	vkDestroyDevice(backend.device, nullptr);
	if(backend.surface) vkDestroySurfaceKHR(backend.instance, backend.surface, nullptr);
	if(backend.messenger) vkDestroyDebugUtilsMessengerEXT(backend.instance, backend.messenger, nullptr);
	vkDestroyInstance(backend.instance, nullptr);
	volkFinalize();
#else
	if(backend.surface) wgpuSurfaceRelease(backend.surface);
	wgpuDeviceRelease(backend.device);
	wgpuAdapterRelease(backend.adapter);
	wgpuInstanceRelease(backend.instance);
#endif
#ifdef NOAPI_DOCTEST_WINDOW
	if(window) glfwDestroyWindow(window);
	glfwTerminate();
#endif
}

int main(int argc, char** argv) {
	// tests.cpp runs this binary again with this argument, to watch the default error callback end a process that has no device yet
	if(argc > 1 && std::string(argv[1]) == "--call-default-error-callback") {
		const char* text = "an example of a device error";
		gpuDefaultErrorCallbackEXT(nullptr, 0, {text, strlen(text)});
		return 0;
	}

	if(!setup()) return 1;
	doctest::Context context(argc, argv);
	int result = context.run();
	shutdown();
	return result;
}
