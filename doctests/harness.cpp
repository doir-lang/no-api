#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "harness.h"

#include <atomic>
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

static bool setup() {
#ifdef NOAPI_BACKEND_VULKAN
	auto created = gpuSetupDefaultVulkanEXT([](VkInstance) -> VkSurfaceKHR { return VK_NULL_HANDLE; }, count_error);
#else
	auto created = gpuSetupDefaultWebGPUEXT([](WGPUInstance) -> WGPUSurface { return nullptr; }, count_error);
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
	if(backend.messenger) vkDestroyDebugUtilsMessengerEXT(backend.instance, backend.messenger, nullptr);
	vkDestroyInstance(backend.instance, nullptr);
	volkFinalize();
#else
	wgpuDeviceRelease(backend.device);
	wgpuAdapterRelease(backend.adapter);
	wgpuInstanceRelease(backend.instance);
#endif
}

int main(int argc, char** argv) {
	if(!setup()) return 1;
	doctest::Context context(argc, argv);
	int result = context.run();
	shutdown();
	return result;
}
