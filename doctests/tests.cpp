// Unit tests: the edge cases, failure modes and exhaustive checks that would crowd a header example. The header examples (see tools/doctest.py) are the rest of the suite.
#include <doctest/doctest.h>
#include "harness.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#ifdef __linux__
	#include <malloc.h>
	#include <sys/wait.h>
#endif

#ifdef __linux__
// In a second run of this binary with every Vulkan driver hidden, which WebGPU (through Dawn) depends on as well
TEST_CASE("the backend's setup says why it failed when there is no device") {
	auto self = std::filesystem::read_symlink("/proc/self/exe").string();
	auto log = std::filesystem::temp_directory_path() / "noapi-doctest-no-device.log";
	int status = std::system(("VK_DRIVER_FILES=/nonexistent.json '" + self + "' > '" + log.string() + "' 2>&1").c_str());
	REQUIRE(WIFEXITED(status));
	CHECK(WEXITSTATUS(status) == 1);

	std::ifstream output(log);
	std::string text((std::istreambuf_iterator<char>(output)), std::istreambuf_iterator<char>());
	CHECK(text.find("could not create a headless device") != std::string::npos);
	std::filesystem::remove(log);
}
#endif

#if defined(NOAPI_BACKEND_WEBGPU) && defined(__linux__)
// In a second run of this binary, since the callback ends whatever process calls it (see main in harness.cpp)
TEST_CASE("gpuDefaultErrorCallbackEXT terminates the process") {
	// Resolved here: inside std::system, /proc/self would be the shell
	auto self = std::filesystem::read_symlink("/proc/self/exe").string();
	REQUIRE(std::system(("'" + self + "' --help > /dev/null").c_str()) == 0); // The binary runs at all

	int status = std::system(("'" + self + "' --call-default-error-callback 2> /dev/null").c_str());
	REQUIRE(WIFEXITED(status));
	CHECK(WEXITSTATUS(status) != 0);
}
#endif

#ifdef NOAPI_BACKEND_WEBGPU
// Copies `bytes` from `from` into `readback` and returns them, so what the GPU holds can be checked
static std::vector<uint8_t> read_back(GpuQueue* queue, uint8_t* readback, void* from, size_t bytes) {
	auto cmd = gpuStartCommandRecording(queue);
	gpuMemCpy(cmd, gpuHostToDevicePointer(queue, readback), gpuHostToDevicePointer(queue, from), bytes);
	gpuSyncMemoryEXT(cmd, gpuHostToDevicePointer(queue, readback));
	gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
	return std::vector<uint8_t>(readback, readback + bytes);
}

// A queue told its device binds at most 64 KiB of storage runs through its five monobuffers quickly, so growing, moving on and running out can all be watched
TEST_CASE("gpuMalloc grows a monobuffer, moves on to the next when one is full, reuses freed holes, and fails once all are taken") {
	auto backend = noapi_doctest_backend();
	WGPULimits limits = backend.limits;
	limits.maxStorageBufferBindingSize = 64 << 10;
	auto queue = gpuCreateQueue(backend.adapter, backend.device, limits, gpuDefaultCpuAllocator);
	REQUIRE(queue);
	noapi_doctest_begin();

	auto readback = (uint8_t*)gpuMalloc(queue, 40 << 10, 16, MEMORY_READBACK); // Grows the first monobuffer from nothing
	REQUIRE(readback);

	// Each block holds its own byte, written and synced as it is made, so an overlap or a lost move shows up when read back
	std::vector<std::pair<uint8_t*, size_t>> blocks;
	auto allocate = [&](size_t bytes) {
		auto block = (uint8_t*)gpuMalloc(queue, bytes);
		if(!block) return false;
		memset(block, int(blocks.size() + 1), bytes);
		gpuSyncMemoryImmediateEXT(queue, gpuHostToDevicePointer(queue, block));
		blocks.emplace_back(block, bytes);
		return true;
	};
	auto check = [&] {
		for(size_t i = 0; i < blocks.size(); ++i)
			if(blocks[i].first) {
				auto contents = read_back(queue, readback, blocks[i].first, blocks[i].second);
				CHECK(contents.front() == i + 1);
				CHECK(contents.back() == i + 1);
			}
	};

	REQUIRE(allocate(256)); // Fills the first monobuffer: it still fits at the end, and the 23 KiB after it goes on the freelist
	for(int i = 0; i < 4; ++i) REQUIRE(allocate(40 << 10)); // One for each of the other monobuffers
	check();

	CHECK(!gpuMalloc(queue, 40 << 10)); // Every monobuffer is taken
	CHECK(errno == WGPUErrorType_OutOfMemory);
	CHECK(!gpuMalloc(queue, 128 << 10)); // Larger than any monobuffer can be
	CHECK(errno == WGPUErrorType_OutOfMemory);

	// Two blocks from the first monobuffer's tail, then freed: alone neither hole fits 20 KiB, joined with the rest they do
	REQUIRE(allocate(8 << 10));
	REQUIRE(allocate(8 << 10));
	for(int i: {5, 6}) {
		gpuFree(queue, blocks[i].first);
		blocks[i].first = nullptr;
	}
	REQUIRE(allocate(20 << 10));
	check();

	for(auto [block, _bytes]: blocks)
		if(block) gpuFree(queue, block);
	gpuFree(queue, readback);
	gpuFreeQueue(queue);
	noapi_doctest_end();
}

// Each shape of texture costs a binding, so a queue told its device binds one of each kind runs out at the second shape
TEST_CASE("gpuCreateTexture fails once every binding a texture shape needs is taken") {
	auto backend = noapi_doctest_backend();
	WGPULimits limits = backend.limits;
	limits.maxSampledTexturesPerShaderStage = 1;
	limits.maxStorageTexturesPerShaderStage = 1;
	auto queue = gpuCreateQueue(backend.adapter, backend.device, limits, gpuDefaultCpuAllocator);
	REQUIRE(queue);
	noapi_doctest_begin();

	std::vector<gpu*> memory;
	auto create = [&](uint32_t size, TEXTURE_USAGE_FLAGS usage) {
		GpuTextureDesc desc = {.dimensions = {size, size, 1}, .format = FORMAT_RGBA8_UNORM, .usage = usage};
		memory.push_back((gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc)));
		return gpuCreateTexture(queue, desc, memory.back());
	};
	CHECK(create(16, USAGE_SAMPLED));
	CHECK(create(16, USAGE_SAMPLED)); // The same shape shares the binding
	CHECK(!create(32, USAGE_SAMPLED));
	CHECK(errno == WGPUErrorType_OutOfMemory);
	CHECK(create(16, USAGE_STORAGE));
	CHECK(!create(32, USAGE_STORAGE));
	CHECK(errno == WGPUErrorType_OutOfMemory);

	for(auto block: memory) gpuFree(queue, block);
	gpuFreeQueue(queue);
	noapi_doctest_end();
}
#endif

// On WebGPU textures of one shape share an atlas, which grows as more arrive and hands freed layers to the next; on Vulkan each is an image of its own. Either way every texture keeps its own texels.
TEST_CASE("textures keep their own texels as the storage they share grows and is reused") {
	auto queue = noapi_doctest_queue();
	noapi_doctest_begin();

	constexpr uint32_t count = 5;
	auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
		import noapi;
		[shader("compute")] [numthreads(5, 1, 1)]
		void main(uint3 thread : SV_DispatchThreadID) { gpuComputeData<float4>()[thread.x] = gpuSample2D(thread.x, gpuGetSamplerIndex(gpuDefaultSampler()), float2(0.25, 0.25)); }
	)slang"));
	REQUIRE(pipeline);

	auto texels = gpuMalloc<uint32_t>(queue, 64 * 64);
	auto texels_gpu = gpuHostToDevicePointer(queue, texels);
	gpu* memory[count] = {};
	uint8_t red[count] = {};
	auto create = [&](uint32_t i, uint32_t size, uint8_t value) {
		GpuTextureDesc desc = {.dimensions = {size, size, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_TRANSFER_DST)};
		memory[i] = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory[i]);
		REQUIRE(texture);
		for(uint32_t t = 0; t < 64 * 64; ++t) texels[t] = 0xff000000u | value;
		red[i] = value;
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, texels_gpu);
		gpuCopyToTexture(cmd, memory[i], texels_gpu, texture);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		return texture;
	};

	GpuTexture* textures[count];
	for(uint32_t i = 0; i < count; ++i) textures[i] = create(i, i == 1 ? 48 : 64, uint8_t(40 * (i + 1))); // Two growths, one smaller than its slot

	// A storage texture freed alongside, so freed layers of two atlases sit together
	GpuTextureDesc storage_desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_STORAGE};
	auto storage = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, storage_desc));
	REQUIRE(gpuCreateTexture(queue, storage_desc, storage));
	gpuFree(queue, memory[2]);
	gpuFree(queue, storage);
	textures[2] = create(2, 64, 250); // Into the layers just freed

	auto heap = gpuMalloc<GpuTextureDescriptor>(queue, count, MEMORY_DESCRIPTOR_HEAP);
	for(uint32_t i = 0; i < count; ++i) heap[i] = gpuTextureViewDescriptor(queue, textures[i], {});
	auto heap_gpu = gpuHostToDevicePointer(queue, heap);
	auto colors = gpuMalloc<float>(queue, 4 * count, MEMORY_READBACK);
	auto colors_gpu = gpuHostToDevicePointer(queue, colors);

	auto cmd = gpuStartCommandRecording(queue);
	gpuSyncMemoryEXT(cmd, heap_gpu);
	gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
	gpuSetPipeline(cmd, pipeline);
	gpuDispatch(cmd, colors_gpu, {1, 1, 1});
	gpuSyncMemoryEXT(cmd, colors_gpu);
	gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
	for(uint32_t i = 0; i < count; ++i)
		CHECK(colors[i * 4] == doctest::Approx(red[i] / 255.0f).epsilon(0.01));

	gpuFreePipeline(queue, pipeline);
	gpuFree(queue, colors);
	gpuFree(queue, heap);
	gpuFree(queue, texels);
	for(auto block: memory) gpuFree(queue, block);
	noapi_doctest_end();
}

#ifdef NOAPI_BACKEND_VULKAN
// Host allocations made through these fail while `failing` is set, which is how a device out of memory looks to the backend
static bool failing = false;

struct Block { void* base; size_t size; };

static void* failable_allocate(void*, size_t size, size_t align, VkSystemAllocationScope) {
	if(failing) return nullptr;
	align = std::max(align, alignof(Block));
	auto base = (char*)std::malloc(size + align + sizeof(Block));
	auto user = (char*)(((uintptr_t)(base + sizeof(Block)) + align - 1) & ~(uintptr_t)(align - 1));
	((Block*)user)[-1] = {base, size};
	return user;
}

static void failable_free(void*, void* memory) {
	if(memory) std::free(((Block*)memory)[-1].base);
}

static void* failable_reallocate(void* userdata, void* memory, size_t size, size_t align, VkSystemAllocationScope scope) {
	if(!memory) return failable_allocate(userdata, size, align, scope);
	if(!size) {
		failable_free(userdata, memory);
		return nullptr;
	}
	auto out = failable_allocate(userdata, size, align, scope);
	if(!out) return nullptr;
	memcpy(out, memory, std::min(size, ((Block*)memory)[-1].size));
	failable_free(userdata, memory);
	return out;
}

static VkAllocationCallbacks failable = {
	.pfnAllocation = failable_allocate,
	.pfnReallocation = failable_reallocate,
	.pfnFree = failable_free,
};

TEST_CASE("objects the device can't allocate are reported as failures") {
	auto backend = noapi_doctest_backend();
	auto queue = gpuCreateQueue(backend.instance, backend.gpu, backend.device, backend.graphics_queue, backend.graphics_queue_family, true, gpuDefaultCpuAllocator, &failable);
	REQUIRE(queue);
	noapi_doctest_begin();

	auto shader = string_to_bytes(R"slang(
		import noapi;
		[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(0, 0, 0, 1); }
		[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
	)slang");
	GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
	GpuSurface* surface = backend.surface ? gpuCreateSurfaceEXT(queue, backend.surface, {.texture = {.dimensions = {64, 64, 1}, .usage = USAGE_COLOR_ATTACHMENT}}) : nullptr;

	failing = true;
	CHECK(!gpuCreateSemaphore(queue, 0));
	CHECK(errno == VK_ERROR_OUT_OF_HOST_MEMORY);
	CHECK(!gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}}));
	CHECK(!gpuCreateComputePipeline(queue, string_to_bytes(R"slang(import noapi; [shader("compute")] [numthreads(1, 1, 1)] void main() {})slang")));
	if(surface) CHECK(!gpuSurfaceNextTextureEXT(queue, surface)); // Its semaphores can't be made
	failing = false;
	noapi_doctest_begin(); // The validation layer reports each failed allocation, which is what this test is for

	if(surface) gpuFreeSurfaceEXT(queue, surface);
	gpuFreeQueue(queue);
	noapi_doctest_end();
}
#endif

// Freed layers lying next to each other join up, so an array larger than either one fits where both were
TEST_CASE("freed texture layers are joined for a larger array") {
	auto queue = noapi_doctest_queue();
	noapi_doctest_begin();

	auto texels = gpuMalloc<uint32_t>(queue, 4 * 16 * 16);
	auto texels_gpu = gpuHostToDevicePointer(queue, texels);
	auto readback = gpuMalloc<uint32_t>(queue, 4 * 16 * 16, MEMORY_READBACK);
	auto readback_gpu = gpuHostToDevicePointer(queue, readback);
	auto round_trip = [&](gpu* memory, GpuTexture* texture, uint32_t layers, uint32_t value) {
		for(uint32_t t = 0; t < layers * 16 * 16; ++t) texels[t] = value + t / (16 * 16);
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, texels_gpu);
		gpuCopyToTexture(cmd, memory, texels_gpu, texture);
		gpuBarrier(cmd, STAGE_TRANSFER, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, readback_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, readback_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		CHECK(readback[0] == value);
		CHECK(readback[layers * 16 * 16 - 1] == value + layers - 1);
	};
	auto create = [&](uint32_t layers, gpu*& memory) {
		GpuTextureDesc desc = {.type = TEXTURE_2D_ARRAY, .dimensions = {16, 16, 1}, .layerCount = layers, .format = FORMAT_RGBA8_UNORM,
			.usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_TRANSFER_SRC | USAGE_TRANSFER_DST)};
		memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		REQUIRE(texture);
		return texture;
	};

	gpu* memory[4];
	GpuTexture* textures[4];
	for(int i = 0; i < 3; ++i) textures[i] = create(2, memory[i]);
	round_trip(memory[2], textures[2], 2, 300);
	gpuFree(queue, memory[0]);
	gpuFree(queue, memory[1]);
	textures[3] = create(4, memory[3]); // Fits only where the first two were, joined
	round_trip(memory[3], textures[3], 4, 100);
	round_trip(memory[2], textures[2], 2, 300); // Left as it was

	gpu* extra;
	auto another = create(2, extra); // The hole the larger array used up is dropped on the way
	round_trip(extra, another, 2, 500);

	gpuFree(queue, extra);
	gpuFree(queue, memory[2]);
	gpuFree(queue, memory[3]);
	gpuFree(queue, readback);
	gpuFree(queue, texels);
	noapi_doctest_end();
}

#ifdef NOAPI_BACKEND_VULKAN
// Swaps one of volk's entry points for the length of a scope. gpuCreateQueue loads volk only once per instance, so a swap made before it holds.
template<typename F>
struct Swap {
	F& slot;
	F saved;
	Swap(F& slot, F replacement): slot(slot), saved(slot) { slot = replacement; }
	~Swap() { slot = saved; }
};

static PFN_vkGetPhysicalDeviceFeatures2 real_features;
static PFN_vkGetPhysicalDeviceQueueFamilyProperties real_families;
static PFN_vkGetDeviceProcAddr real_device_proc;
static bool bind_fails = false;

// A device like the one in use, but without 64 bit buffer atomics, dual source blending or swapchain present fences
static void reduced_features(VkPhysicalDevice gpu, VkPhysicalDeviceFeatures2* features) {
	real_features(gpu, features);
	features->features.dualSrcBlend = false;
	for(auto next = (VkBaseOutStructure*)features->pNext; next; next = next->pNext) {
		if(next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
			((VkPhysicalDeviceVulkan12Features*)next)->shaderBufferInt64Atomics = false;
		if(next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR)
			((VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR*)next)->swapchainMaintenance1 = false;
	}
}

static void families_without_graphics(VkPhysicalDevice gpu, uint32_t* count, VkQueueFamilyProperties* families) {
	real_families(gpu, count, families);
	for(uint32_t i = 0; families && i < *count; ++i) families[i].queueFlags &= ~VK_QUEUE_GRAPHICS_BIT;
}

static PFN_vkBindImageMemory real_bind;
static VkResult VKAPI_CALL maybe_failing_bind(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset) {
	return bind_fails ? VK_ERROR_OUT_OF_DEVICE_MEMORY : real_bind(device, image, memory, offset);
}

// VMA asks for its entry points once, when the queue's allocator is made, so the bind is intercepted there
static PFN_vkVoidFunction VKAPI_CALL device_proc_with_failing_bind(VkDevice device, const char* name) {
	if(strcmp(name, "vkBindImageMemory") == 0) {
		real_bind = (PFN_vkBindImageMemory)real_device_proc(device, name);
		return (PFN_vkVoidFunction)maybe_failing_bind;
	}
	return real_device_proc(device, name);
}

static void draw_and_present(GpuQueue* queue, GpuSurface* surface) {
	auto target = gpuSurfaceNextTextureEXT(queue, surface);
	REQUIRE(target);
	GpuColorAttachment color = {.texture = target, .loadOp = LOAD_OP_CLEAR};
	GpuRenderPassDesc pass = {.colorAttachments = {&color, 1}};
	auto cmd = gpuStartCommandRecording(queue);
	gpuBeginRenderPass(cmd, pass);
	gpuEndRenderPass(cmd, pass);
	gpuSurfacePresentEXT(queue, surface, gpuSubmit(queue, {&cmd, 1}));
}

TEST_CASE("a device without the optional features falls back as documented") {
	auto backend = noapi_doctest_backend();
	real_features = vkGetPhysicalDeviceFeatures2;
	GpuQueue* queue;
	{
		Swap swap(vkGetPhysicalDeviceFeatures2, (PFN_vkGetPhysicalDeviceFeatures2)reduced_features);
		queue = gpuCreateQueue(backend.instance, backend.gpu, backend.device, backend.graphics_queue, backend.graphics_queue_family, true, gpuDefaultCpuAllocator, nullptr);
	}
	REQUIRE(queue);
	noapi_doctest_begin();
	auto capabilities = gpuGetCapabilitiesEXT(queue);
	CHECK(!capabilities.split_barrier_signals);
	CHECK(!capabilities.dual_source_blending);

	int unsupported = 0;
	gpuSetDiagnosticCallbackEXT(queue, [](GpuQueue*, GPU_DIAGNOSTIC kind, GpuStringView, void* count) { if(kind == GPU_DIAGNOSTIC_UNSUPPORTED) ++*(int*)count; }, &unsupported);
	auto counter = gpuMalloc<uint64_t>(queue);
	auto cmd = gpuStartCommandRecording(queue);
	gpuSignalAfter(cmd, STAGE_COMPUTE, gpuHostToDevicePointer(queue, counter), 1, SIGNAL_ATOMIC_MAX);
	gpuFreeCommandBuffer(cmd);
	CHECK(unsupported == 1);

	GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
	auto shader = string_to_bytes(R"slang([shader("vertex")] float4 vertexMain() : SV_Position { return 0; } [shader("fragment")] float4 fragmentMain() : SV_Target { return 1; })slang");
	CHECK(!gpuCreateGraphicsPipeline(queue, shader, shader, {.supportDualSourceBlending = true, .colorTargets = {&target, 1}}));
	CHECK(unsupported == 2);

	// Without present fences a replaced swapchain is let go once the surface has presented past every image it had
	if(backend.surface) {
		auto surface = gpuCreateSurfaceEXT(queue, backend.surface, {.texture = {.dimensions = {64, 64, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
		REQUIRE(surface);
		draw_and_present(queue, surface);
		auto config = gpuSurfaceGetConfigurationEXT(surface);
		config.texture.dimensions = {48, 48, 1};
		gpuSurfaceReconfigureEXT(queue, surface, config);
		for(int frame = 0; frame < 8; ++frame) draw_and_present(queue, surface);
		gpuFreeSurfaceEXT(queue, surface);
	}

	gpuSetDiagnosticCallbackEXT(queue, nullptr, nullptr);
	gpuFree(queue, counter);
	gpuFreeQueue(queue);
	noapi_doctest_end();
}

TEST_CASE("a queue can't be looked up on a device without a family for it") {
	auto backend = noapi_doctest_backend();
	real_families = vkGetPhysicalDeviceQueueFamilyProperties;
	Swap swap(vkGetPhysicalDeviceQueueFamilyProperties, (PFN_vkGetPhysicalDeviceQueueFamilyProperties)families_without_graphics);
	CHECK(!gpuCreateQueue(backend.instance, backend.gpu, backend.device, VK_NULL_HANDLE, (uint32_t)-1, true, gpuDefaultCpuAllocator, nullptr));
	CHECK(errno == VK_ERROR_FEATURE_NOT_PRESENT);
}

TEST_CASE("Vulkan calls that fail are reported, and leave nothing behind") {
	auto backend = noapi_doctest_backend();
	real_device_proc = vkGetDeviceProcAddr;
	GpuQueue* queue;
	{
		Swap swap(vkGetDeviceProcAddr, (PFN_vkGetDeviceProcAddr)device_proc_with_failing_bind);
		queue = gpuCreateQueue(backend.instance, backend.gpu, backend.device, backend.graphics_queue, backend.graphics_queue_family, true, gpuDefaultCpuAllocator, nullptr);
	}
	REQUIRE(queue);
	noapi_doctest_begin();

	// A texture whose memory can't be bound isn't created
	GpuTextureDesc desc = {.dimensions = {16, 16, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED};
	auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
	bind_fails = true;
	CHECK(!gpuCreateTexture(queue, desc, memory));
	CHECK(errno == VK_ERROR_OUT_OF_DEVICE_MEMORY);
	bind_fails = false;
	CHECK(gpuCreateTexture(queue, desc, memory));
	gpuFree(queue, memory);

	// An event that can't be reset for reuse is destroyed and replaced
	auto counter = gpuMalloc<uint64_t>(queue);
	auto counter_gpu = gpuHostToDevicePointer(queue, counter);
	for(int round = 0; round < 2; ++round) {
		auto cmd = gpuStartCommandRecording(queue);
		{
			Swap swap(vkResetEvent, round == 1 ? (PFN_vkResetEvent)[](VkDevice, VkEvent) -> VkResult { return VK_ERROR_OUT_OF_DEVICE_MEMORY; } : vkResetEvent);
			gpuSignalAfter(cmd, STAGE_COMPUTE, counter_gpu, 1, SIGNAL_ATOMIC_SET);
		}
		gpuWaitBefore(cmd, STAGE_COMPUTE, counter_gpu, 1, OP_GREATER_EQUAL);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
	}
	gpuFree(queue, counter);

	if(backend.surface) {
		auto surface = gpuCreateSurfaceEXT(queue, backend.surface, {.texture = {.dimensions = {64, 64, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
		REQUIRE(surface);

		// An image the presentation engine won't hand out is reported as the surface's status
		{
			Swap swap(vkAcquireNextImageKHR, (PFN_vkAcquireNextImageKHR)[](VkDevice, VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t*) -> VkResult { return VK_ERROR_OUT_OF_DATE_KHR; });
			CHECK(!gpuSurfaceNextTextureEXT(queue, surface));
			CHECK(errno == SURFACE_OUT_OF_DATE);
		}

		// Without a present fence, which a failed vkCreateFence leaves it, a frame is still presented
		{
			draw_and_present(queue, surface); // Makes the swapchain's semaphores, so the fence is all the next present creates
			Swap swap(vkCreateFence, (PFN_vkCreateFence)[](VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence*) -> VkResult { return VK_ERROR_OUT_OF_HOST_MEMORY; });
			draw_and_present(queue, surface);
		}

		// Present fences that never signal, as after a failed present, are set aside rather than reused, and let go of with the surface
		static PFN_vkQueuePresentKHR real_present = vkQueuePresentKHR;
		real_present = vkQueuePresentKHR;
		{
			Swap swap(vkQueuePresentKHR, (PFN_vkQueuePresentKHR)[](VkQueue queue, const VkPresentInfoKHR* info) -> VkResult {
				auto stripped = *info;
				stripped.pNext = nullptr; // Drops the fence, so it never signals
				return real_present(queue, &stripped);
			});
			draw_and_present(queue, surface);
			auto config = gpuSurfaceGetConfigurationEXT(surface);
			config.texture.dimensions = {48, 48, 1};
			gpuSurfaceReconfigureEXT(queue, surface, config);
			for(int frame = 0; frame < 8; ++frame) draw_and_present(queue, surface);
		}
		gpuFreeSurfaceEXT(queue, surface);
	}

	gpuFreeQueue(queue);
	noapi_doctest_end();
}
#endif

#ifdef NOAPI_BACKEND_WEBGPU
// An adapter hands out one device, so each device of its own comes from an adapter of its own
static std::pair<WGPUAdapter, WGPUDevice> request_device(const GpuWebGPUDefault& backend, const WGPUDeviceDescriptor& desc) {
	struct Requested { WGPUAdapter adapter = nullptr; WGPUDevice device = nullptr; bool answered = false; } requested;
	wgpuInstanceRequestAdapter(backend.instance, nullptr, {
		.mode = WGPUCallbackMode_AllowSpontaneous,
		.callback = [](WGPURequestAdapterStatus, WGPUAdapter adapter, WGPUStringView, void* out, void*) { ((Requested*)out)->adapter = adapter; ((Requested*)out)->answered = true; },
		.userdata1 = &requested,
	});
	while(!requested.answered) wgpuInstanceProcessEvents(backend.instance);
	REQUIRE(requested.adapter);
	requested.answered = false;
	wgpuAdapterRequestDevice(requested.adapter, &desc, {
		.mode = WGPUCallbackMode_AllowSpontaneous,
		.callback = [](WGPURequestDeviceStatus, WGPUDevice device, WGPUStringView, void* out, void*) { ((Requested*)out)->device = device; ((Requested*)out)->answered = true; },
		.userdata1 = &requested,
	});
	while(!requested.answered) wgpuInstanceProcessEvents(backend.instance);
	REQUIRE(requested.device);
	return {requested.adapter, requested.device};
}

// Asked for without the optional features gpuSetupDefaultWebGPUEXT requests where it can
TEST_CASE("a device without dual source blending turns such pipelines down") {
	auto backend = noapi_doctest_backend();
	auto [adapter, device] = request_device(backend, {.requiredLimits = &backend.limits});

	auto queue = gpuCreateQueue(adapter, device, backend.limits, gpuDefaultCpuAllocator);
	REQUIRE(queue);
	CHECK(!gpuGetCapabilitiesEXT(queue).dual_source_blending);
	GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
	auto shader = string_to_bytes(R"slang([shader("vertex")] float4 vertexMain() : SV_Position { return 0; } [shader("fragment")] float4 fragmentMain() : SV_Target { return 1; })slang");
	CHECK(!gpuCreateGraphicsPipeline(queue, shader, shader, {.supportDualSourceBlending = true, .colorTargets = {&target, 1}}));
	gpuFreeQueue(queue);
	wgpuDeviceRelease(device);
	wgpuAdapterRelease(adapter);
}

#ifdef __linux__
// Dawn destroys a device once the program drops it, whatever still refers to it, so a queue's leftovers only show as memory that each queue made and freed takes for good
TEST_CASE("gpuFreeQueue releases everything it made on the device") {
	auto backend = noapi_doctest_backend();
	auto make_and_free_queue = [&] {
		WGPULimits limits = backend.limits;
		limits.maxStorageBufferBindingSize = 64 << 10; // So that three allocations fill two monobuffers and move on to a third
		auto queue = gpuCreateQueue(backend.adapter, backend.device, limits, gpuDefaultCpuAllocator);
		REQUIRE(queue);
		for(int i = 0; i < 3; ++i) REQUIRE(gpuMalloc(queue, 48 << 10));

		GpuTextureDesc texture_desc = {.dimensions = {16, 16, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED};
		auto texture_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, texture_desc));
		REQUIRE(gpuCreateTexture(queue, texture_desc, texture_memory));
		gpuFree(queue, texture_memory);
		auto kept_texture_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, texture_desc));
		REQUIRE(gpuCreateTexture(queue, texture_desc, kept_texture_memory)); // Left for gpuFreeQueue

		auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(import noapi; [shader("compute")] [numthreads(1, 1, 1)] void main() {})slang"));
		REQUIRE(pipeline);
		auto readback = gpuMalloc<uint32_t>(queue, 4, MEMORY_READBACK);
		auto cmd = gpuStartCommandRecording(queue);
		gpuSetPipeline(cmd, pipeline);
		gpuDispatch(cmd, nullptr, {1, 1, 1});
		gpuSyncMemoryEXT(cmd, gpuHostToDevicePointer(queue, readback));
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));

		// Abandoned mid pass, with a readback still waiting on them
		cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, gpuHostToDevicePointer(queue, readback));
		gpuSetPipeline(cmd, pipeline);
		gpuFreeCommandBuffer(cmd);
		GpuTextureDesc target_desc = {.dimensions = {16, 16, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_COLOR_ATTACHMENT};
		auto target_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, target_desc));
		GpuColorAttachment target = {.texture = gpuCreateTexture(queue, target_desc, target_memory)};
		REQUIRE(target.texture);
		cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, gpuHostToDevicePointer(queue, readback));
		gpuBeginRenderPass(cmd, {.colorAttachments = {&target, 1}});
		gpuFreeCommandBuffer(cmd);

		gpuFreePipeline(queue, pipeline);
		gpuFreeQueue(queue);
		// Dawn deletes what was released only after the GPU is done with it, on a later tick
		for(int i = 0; i < 4; ++i) (void)wgpuDeviceTick(backend.device);
	};

	noapi_doctest_begin();
	for(int i = 0; i < 4; ++i) make_and_free_queue(); // Until what is made once, such as Slang's session and Dawn's caches and pools, has stopped growing
	auto kept_by_eight = [&] {
		auto before = mallinfo2().uordblks;
		for(int i = 0; i < 8; ++i) make_and_free_queue();
		return (ptrdiff_t)(mallinfo2().uordblks - before); // Negative when Dawn trims a cache
	};
	// A leak grows with every queue, while Dawn's pools grow in rare steps, so the quieter of two runs is what counts
	CHECK(std::min(kept_by_eight(), kept_by_eight()) < 8 * 1024);
	noapi_doctest_end();
}
#endif
#endif

#ifdef NOAPI_DOCTEST_FAULTS
// The linker sends the backend's calls into Slang (and on WebGPU into Dawn) through these (see CMakeLists.txt), so a test can make them fail
namespace faults {
	bool defer_requests = false; // Answer adapter and device requests only once events are processed, as a browser does
	bool refuse_device = false;
	bool capabilities_fail = false;
	bool unknown_surface_format = false;
	int texture_status = 0; // A WGPUSurfaceGetCurrentTextureStatus to report instead of the real one
	bool present_fails = false;
	std::vector<std::function<void()>> deferred;
}

extern "C" {
	int32_t __real_slang_createGlobalSession2(const void* desc, void** session);
	int32_t __wrap_slang_createGlobalSession2(const void* desc, void** session) {
		// Slang starts once per process, so this is only ever asked for in a run of its own (see the test below)
		if(std::getenv("NOAPI_DOCTEST_NO_SLANG")) return -1;
		return __real_slang_createGlobalSession2(desc, session);
	}
}

#ifdef __linux__
// In a second run of this binary, since Slang starts once per process (see the wrapper above)
TEST_CASE("pipelines fail cleanly when the shader compiler can't start") {
	if(std::getenv("NOAPI_DOCTEST_NO_SLANG")) {
		std::string messages;
		gpuSetShaderDiagnosticCallbackEXT([](GpuStringView message, void* all) { *(std::string*)all += std::string_view(message); }, &messages);
		CHECK(!gpuCreateComputePipeline(noapi_doctest_queue(), string_to_bytes(R"slang([shader("compute")] [numthreads(1, 1, 1)] void main() {})slang")));
		CHECK(!gpuCreateComputePipeline(noapi_doctest_queue(), string_to_bytes(R"slang([shader("compute")] [numthreads(1, 1, 1)] void main() {})slang")));
		CHECK(messages.find("unavailable") != std::string::npos);
		gpuSetShaderDiagnosticCallbackEXT(nullptr, nullptr);
		return;
	}
	auto self = std::filesystem::read_symlink("/proc/self/exe").string();
	int status = std::system(("NOAPI_DOCTEST_NO_SLANG=1 '" + self + "' -tc='pipelines fail cleanly*' > /dev/null 2>&1").c_str());
	REQUIRE(WIFEXITED(status));
	CHECK(WEXITSTATUS(status) == 0);
}
#endif

#ifdef NOAPI_BACKEND_WEBGPU
extern "C" {
	WGPUFuture __real_wgpuInstanceRequestAdapter(WGPUInstance, const WGPURequestAdapterOptions*, WGPURequestAdapterCallbackInfo);
	WGPUFuture __wrap_wgpuInstanceRequestAdapter(WGPUInstance instance, const WGPURequestAdapterOptions* options, WGPURequestAdapterCallbackInfo info) {
		if(!faults::defer_requests) return __real_wgpuInstanceRequestAdapter(instance, options, info);
		static WGPURequestAdapterCallbackInfo original;
		original = info;
		info.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void*, void*) {
			auto text = std::string(message.data ? message.data : "", message.data ? message.length : 0);
			faults::deferred.push_back([=] { original.callback(status, adapter, {text.data(), text.size()}, original.userdata1, original.userdata2); });
		};
		return __real_wgpuInstanceRequestAdapter(instance, options, info);
	}

	WGPUFuture __real_wgpuAdapterRequestDevice(WGPUAdapter, const WGPUDeviceDescriptor*, WGPURequestDeviceCallbackInfo);
	WGPUFuture __wrap_wgpuAdapterRequestDevice(WGPUAdapter adapter, const WGPUDeviceDescriptor* descriptor, WGPURequestDeviceCallbackInfo info) {
		if(faults::refuse_device) {
			const char* text = "refused by a test";
			info.callback(WGPURequestDeviceStatus_Error, nullptr, {text, strlen(text)}, info.userdata1, info.userdata2);
			return {};
		}
		if(!faults::defer_requests) return __real_wgpuAdapterRequestDevice(adapter, descriptor, info);
		static WGPURequestDeviceCallbackInfo original;
		original = info;
		info.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView, void*, void*) {
			faults::deferred.push_back([=] { original.callback(status, device, {nullptr, 0}, original.userdata1, original.userdata2); });
		};
		return __real_wgpuAdapterRequestDevice(adapter, descriptor, info);
	}

	void __real_wgpuInstanceProcessEvents(WGPUInstance);
	void __wrap_wgpuInstanceProcessEvents(WGPUInstance instance) {
		auto ready = std::move(faults::deferred);
		faults::deferred.clear();
		for(auto& deliver: ready) deliver();
		__real_wgpuInstanceProcessEvents(instance);
	}

	WGPUStatus __real_wgpuSurfaceGetCapabilities(WGPUSurface, WGPUAdapter, WGPUSurfaceCapabilities*);
	WGPUStatus __wrap_wgpuSurfaceGetCapabilities(WGPUSurface surface, WGPUAdapter adapter, WGPUSurfaceCapabilities* capabilities) {
		if(faults::capabilities_fail) return WGPUStatus_Error;
		auto status = __real_wgpuSurfaceGetCapabilities(surface, adapter, capabilities);
		// A format the API has no name for, standing in for the platform specific ones some surfaces list
		if(faults::unknown_surface_format && capabilities->formatCount)
			const_cast<WGPUTextureFormat*>(capabilities->formats)[capabilities->formatCount - 1] = WGPUTextureFormat_BC1RGBAUnorm;
		return status;
	}

	void __real_wgpuSurfaceGetCurrentTexture(WGPUSurface, WGPUSurfaceTexture*);
	void __wrap_wgpuSurfaceGetCurrentTexture(WGPUSurface surface, WGPUSurfaceTexture* texture) {
		__real_wgpuSurfaceGetCurrentTexture(surface, texture);
		if(faults::texture_status) texture->status = (WGPUSurfaceGetCurrentTextureStatus)faults::texture_status;
	}

	WGPUStatus __real_wgpuSurfacePresent(WGPUSurface);
	WGPUStatus __wrap_wgpuSurfacePresent(WGPUSurface surface) {
		auto status = __real_wgpuSurfacePresent(surface);
		return faults::present_fails ? WGPUStatus_Error : status;
	}
}

TEST_CASE("setup waits for requests answered later, and says why a device was refused") {
	static int errors = 0;
	auto count = [](void*, int, GpuStringView) { ++errors; };
	GpuWebGPUDefault webgpu;
	char error[128];

	faults::defer_requests = true;
	CHECK(gpuSetupDefaultWebGPUEXT(nullptr, nullptr, count, true, &webgpu, error, sizeof(error)));
	faults::defer_requests = false;

	// A device lost for any reason but its own destruction is an error worth reporting
	wgpuDeviceForceLoss(webgpu.device, WGPUDeviceLostReason_Unknown, {"lost on purpose", WGPU_STRLEN});
	wgpuInstanceProcessEvents(webgpu.instance);
	CHECK(errors > 0);
	wgpuDeviceRelease(webgpu.device);
	wgpuAdapterRelease(webgpu.adapter);
	wgpuInstanceRelease(webgpu.instance);

	errors = 0;
	faults::refuse_device = true;
	CHECK(!gpuSetupDefaultWebGPUEXT(nullptr, nullptr, count, true, &webgpu, error, sizeof(error)));
	faults::refuse_device = false;
	CHECK(errors == 1);
	CHECK(std::string(error).find("device") != std::string::npos);
}

TEST_CASE("surface calls that fail are reported, and a suboptimal texture is still handed out") {
	auto backend = noapi_doctest_backend();
	if(!backend.surface) return;
	auto queue = noapi_doctest_queue();
	noapi_doctest_begin();
	auto surface = gpuCreateSurfaceEXT(queue, backend.surface, {.texture = {.dimensions = {64, 64, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
	REQUIRE(surface);
	auto configured = gpuSurfaceGetConfigurationEXT(surface);

	faults::capabilities_fail = true;
	auto capabilities = gpuGetSurfaceCapabilitiesEXT(queue, surface);
	CHECK(capabilities.formatCount == 0);
	CHECK(errno == WGPUErrorType_Unknown);
	auto resized = configured;
	resized.texture.dimensions = {32, 32, 1};
	gpuSurfaceReconfigureEXT(queue, surface, resized);
	CHECK(errno == WGPUErrorType_Unknown);
	CHECK(gpuSurfaceGetConfigurationEXT(surface).texture.dimensions.x == configured.texture.dimensions.x); // Left as it was
	faults::capabilities_fail = false;

	faults::unknown_surface_format = true;
	capabilities = gpuGetSurfaceCapabilitiesEXT(queue, surface);
	faults::unknown_surface_format = false;
	for(auto format: std::span(capabilities.formats, capabilities.formatCount)) CHECK(format != FORMAT_NONE);

	auto present = [&](const GpuTexture* target) {
		GpuColorAttachment color = {.texture = target, .loadOp = LOAD_OP_CLEAR};
		GpuRenderPassDesc pass = {.colorAttachments = {&color, 1}};
		auto cmd = gpuStartCommandRecording(queue);
		gpuBeginRenderPass(cmd, pass);
		gpuEndRenderPass(cmd, pass);
		gpuSurfacePresentEXT(queue, surface, gpuSubmit(queue, {&cmd, 1}));
	};
	faults::texture_status = WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal;
	auto target = gpuSurfaceNextTextureEXT(queue, surface);
	CHECK(target);
	CHECK(errno == SURFACE_SUBOPTIMAL);
	present(target);
	faults::texture_status = WGPUSurfaceGetCurrentTextureStatus_Outdated;
	CHECK(!gpuSurfaceNextTextureEXT(queue, surface));
	CHECK(errno == WGPUSurfaceGetCurrentTextureStatus_Outdated);
	faults::texture_status = 0;

	faults::present_fails = true;
	target = gpuSurfaceNextTextureEXT(queue, surface);
	REQUIRE(target);
	present(target);
	CHECK(errno == WGPUErrorType_Unknown);
	faults::present_fails = false;

	gpuWaitIdleEXT(queue);
	gpuFreeSurfaceEXT(queue, surface);
	noapi_doctest_end();
}
#endif
#endif

// ---------------------------------------------------------------------------
// Behaviour of the definitions in the public headers, beyond what their examples show
// ---------------------------------------------------------------------------

// compute.h

TEST_CASE("TEXTURE: every texture type is created and sampled through the heap") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc descs[] = {
			{.type = TEXTURE_1D, .dimensions = {16, 1, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED},
			{.type = TEXTURE_3D, .dimensions = {16, 16, 4}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED},
			{.type = TEXTURE_2D_ARRAY, .dimensions = {16, 16, 1}, .layerCount = 3, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED},
			{.type = TEXTURE_CUBE, .dimensions = {16, 16, 1}, .layerCount = 6, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED},
			{.type = TEXTURE_CUBE_ARRAY, .dimensions = {16, 16, 1}, .layerCount = 12, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED},
			{.type = TEXTURE_3D, .dimensions = {16, 16, 4}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_STORAGE},
		};
		auto heap = gpuMalloc<GpuTextureDescriptor>(queue, std::size(descs) + 1, MEMORY_DESCRIPTOR_HEAP);
		std::vector<gpu*> memory;
		for(size_t i = 0; i < std::size(descs); ++i) {
			auto size = gpuTextureSizeAlign(queue, descs[i]);
			if(size.size == 0) continue; // WebGPU has no 1D textures
			memory.push_back((gpu*)gpuMalloc(queue, size));
			auto texture = gpuCreateTexture(queue, descs[i], memory.back());
			assert(texture);
			heap[i] = descs[i].usage == USAGE_STORAGE ? gpuRWTextureViewDescriptor(queue, texture, {}) : gpuTextureViewDescriptor(queue, texture, {});
			if(descs[i].type == TEXTURE_2D_ARRAY) heap[std::size(descs)] = gpuTextureViewDescriptor(queue, texture, {.baseLayer = 1, .layerCount = 1}); // The middle layer alone
		}

		// Shaders are compiled against the textures there are, whatever their dimensionality
		auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
			import noapi;
			[shader("compute")] [numthreads(1, 1, 1)] void main() { gpuComputeData<float4>()[0] = gpuSample(1, 0, float3(0.5), 0, 0); }
		)slang"));
		assert(pipeline);

		gpuFreePipeline(queue, pipeline);
		for(auto block: memory) gpuFree(queue, block);
		gpuFree(queue, heap);
	}
	noapi_doctest_end();
}

TEST_CASE("FORMAT: every format the device offers is created and read back") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc none = GPU_TEXTURE_DESC_DEFAULT;
		none.usage = USAGE_SAMPLED;
		assert(gpuTextureSizeAlign(queue, NOAPI_REF(none)).size == 0); // FORMAT_NONE names no texel at all
		void* readback_cpu = gpuMalloc(queue, 64 * 4 * 16, 16, MEMORY_READBACK);
		gpu* readback = gpuHostToDevicePointer(queue, readback_cpu);
		for(int format = FORMAT_R8_UNORM; format <= FORMAT_D32_FLOAT_S8_UINT; ++format) {
			GpuTextureDesc desc = GPU_TEXTURE_DESC_DEFAULT;
			desc.dimensions.x = 64;
			desc.dimensions.y = 4;
			desc.format = (FORMAT)format;
			desc.usage = (TEXTURE_USAGE_FLAGS)(USAGE_TRANSFER_SRC | (gpuFormatIsDepthStencilEXT(desc.format) ? USAGE_DEPTH_STENCIL_ATTACHMENT : USAGE_TRANSFER_DST));
			GpuTextureSizeAlign size = gpuTextureSizeAlign(queue, NOAPI_REF(desc));
			if(size.size == 0) continue; // An optional format this device lacks
			gpu* memory = (gpu*)gpuMalloc(queue, size.size, size.align, MEMORY_TEXTURE);
			GpuTexture* texture = gpuCreateTexture(queue, NOAPI_REF(desc), memory);
			assert(texture);

			// Read back, as every format can be on Vulkan; WebGPU reports the depth and stencil copies it can't make and skips them
			GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
			gpuCopyFromTexture(cmd, readback, memory, texture);
			GpuCommandBufferSpan buffers = {&cmd, 1};
			gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, buffers, NULL, 0), UINT64_MAX);
			gpuFreeDevicePointerEXT(queue, memory);
		}
		gpuFree(queue, readback_cpu);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuFreeQueue frees the memory, textures and semaphores still left on the queue") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuQueue* second = gpuCreateQueue(backend);
		assert(second);
		void* memory = gpuMalloc(second, 64);
		gpuFree(second, memory);

		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED};
		auto texture_memory = (gpu*)gpuMalloc(second, gpuTextureSizeAlign(second, desc));
		assert(gpuCreateTexture(second, desc, texture_memory));
		auto counter = gpuMalloc<uint64_t>(second);
		auto counter_gpu = gpuHostToDevicePointer(second, counter);
		for(int pairs: {2, 1}) { // The second round reuses what the first one finished with
			auto cmd = gpuStartCommandRecording(second);
			for(int i = 0; i < pairs; ++i) {
				gpuSignalAfter(cmd, STAGE_COMPUTE, counter_gpu, i + 1, SIGNAL_ATOMIC_SET);
				gpuWaitBefore(cmd, STAGE_COMPUTE, counter_gpu, i + 1, OP_GREATER_EQUAL);
			}
			auto submission = gpuSubmit(second, {&cmd, 1});
			if(pairs == 2) gpuWaitSemaphore(second, gpuGetSubmissionSemaphoreEXT(second), submission);
		}
		gpuFreeQueue(second); // Takes the texture, its memory and the counter with it, once the last submission is done
	}
	noapi_doctest_end();
}

TEST_CASE("gpuFree through a GPU address forgets the allocation") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto values = gpuMalloc<float>(queue, 4);
		gpu* device = gpuHostToDevicePointer(queue, values);
		assert(device);
		gpuFree(queue, device);
		assert(!gpuHostToDevicePointer(queue, values));
	}
	noapi_doctest_end();
}

TEST_CASE("gpuTextureSizeAlign: every device offers single sampled textures") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc desc = GPU_TEXTURE_DESC_DEFAULT;
		desc.dimensions.x = desc.dimensions.y = 16;
		desc.format = FORMAT_RGBA8_UNORM;
		desc.usage = USAGE_COLOR_ATTACHMENT;
		for(uint32_t samples = 1; samples <= 64; samples *= 2) {
			desc.sampleCount = samples;
			GpuTextureSizeAlign size = gpuTextureSizeAlign(queue, NOAPI_REF(desc));
			assert(size.size > 0 || samples > 1); // Every device renders without multisampling
		}
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCreateTexture turns down descriptions the device can't make") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto memory = (gpu*)gpuMalloc(queue, 64 * 64 * 4, 256, MEMORY_TEXTURE);
		assert(!gpuCreateTexture(queue, {.dimensions = {64, 64, 1}, .sampleCount = 64, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_COLOR_ATTACHMENT}, memory));
		assert(!gpuCreateTexture(queue, {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_COLOR_ATTACHMENT}, (gpu*)((char*)memory + 256))); // Not the start of an allocation
		// Created on Vulkan; WebGPU only samples formats a sampler can filter, so it reports this one unsupported and returns NULL
		(void)gpuCreateTexture(queue, {.dimensions = {64, 64, 1}, .format = FORMAT_R32_UINT, .usage = USAGE_SAMPLED}, memory);
		gpuFree(queue, memory);

		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_SAMPLED};
		auto shared = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		assert(gpuCreateTexture(queue, desc, shared));
		assert(!gpuCreateTexture(queue, desc, shared)); // One texture per allocation
		gpuFree(queue, shared);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuRWTextureViewDescriptor: a dispatch stores into the mip a view names") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .mipCount = 2, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_STORAGE | USAGE_TRANSFER_SRC)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		auto heap = gpuMalloc<GpuTextureDescriptor>(queue, 2, MEMORY_DESCRIPTOR_HEAP);
		heap[0] = gpuRWTextureViewDescriptor(queue, texture, {.mipCount = 1});
		heap[1] = gpuRWTextureViewDescriptor(queue, texture, {.baseMip = 1, .mipCount = 1}); // WebGPU stores into mip 0 regardless, and says so
		auto heap_gpu = gpuHostToDevicePointer(queue, heap);

		auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
			import noapi;
			[shader("compute")] [numthreads(8, 8, 1)]
			void main(uint3 thread : SV_DispatchThreadID) { gpuStoreTexture2D(0, thread.xy, float4(0, 0, 1, 1)); }
		)slang"));
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, heap_gpu);
		gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
		gpuSetPipeline(cmd, pipeline);
		gpuDispatch(cmd, heap_gpu, {8, 8, 1});
		gpuBarrier(cmd, STAGE_COMPUTE, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(pixels[64 * 64 - 1] == 0xffff0000); // Opaque blue, as RGBA bytes

		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, pixels);
		gpuFree(queue, heap);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuRWTextureViewDescriptor: every storable format can be stored into") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
			import noapi;
			[shader("compute")] [numthreads(4, 4, 1)]
			void main(uint3 thread : SV_DispatchThreadID) { gpuStoreTexture2D(0, thread.xy, float4(1, 0, 0, 1)); }
		)slang"));
		auto heap = gpuMalloc<GpuTextureDescriptor>(queue, 1, MEMORY_DESCRIPTOR_HEAP);
		auto heap_gpu = gpuHostToDevicePointer(queue, heap);
		for(int format = FORMAT_R8_UNORM; format <= FORMAT_D32_FLOAT_S8_UINT; ++format) {
			GpuTextureDesc desc = {.dimensions = {4, 4, 1}, .format = (FORMAT)format, .usage = USAGE_STORAGE};
			assert((gpuTextureSizeAlign(queue, desc).size > 0) == gpuFormatIsStorableEXT(desc.format));
			if(!gpuFormatIsStorableEXT(desc.format)) continue;
			auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
			heap[0] = gpuRWTextureViewDescriptor(queue, gpuCreateTexture(queue, desc, memory), {});
			auto cmd = gpuStartCommandRecording(queue);
			gpuSyncMemoryEXT(cmd, heap_gpu);
			gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
			gpuSetPipeline(cmd, pipeline);
			gpuDispatch(cmd, heap_gpu, {1, 1, 1});
			gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
			gpuFree(queue, memory);
		}
		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, heap);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCreateComputePipeline fails on ambiguous entry points and on imports that don't compile or link") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		std::string messages;
		gpuSetShaderDiagnosticCallbackEXT([](GpuStringView message, void* all) { *(std::string*)all += std::string_view(message); }, &messages);
		assert(!gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
			[shader("compute")] [numthreads(1, 1, 1)] void first() {}
			[shader("compute")] [numthreads(1, 1, 1)] void second() {}
		)slang"))); // Two compute entry points
		assert(!gpuCreateComputePipeline(queue, string_to_bytes("void helper() {}"))); // None
		assert(!gpuCreateComputePipeline(queue, string_to_bytes("not a shader")));
		const char* declares = "module noapi_example_declares; public extern uint missing();";
		gpuAddShaderModuleEXT("noapi_example_declares", {declares, strlen(declares)});
		assert(!gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
			import noapi_example_declares;
			[shader("compute")] [numthreads(1, 1, 1)] void main() { missing(); }
		)slang"))); // Declared by a module, never defined, which only linking finds out
		assert(gpuRemoveShaderModuleEXT("noapi_example_declares"));

		const char* broken = "module noapi_example_broken; not slang";
		gpuAddShaderModuleEXT("noapi_example_broken", {broken, strlen(broken)});
		assert(!gpuCreateComputePipeline(queue, string_to_bytes(R"slang([shader("compute")] [numthreads(1, 1, 1)] void main() {})slang")));
		assert(gpuRemoveShaderModuleEXT("noapi_example_broken"));
		assert(messages.find("more than one compute entry point") != std::string::npos && messages.find("missing") != std::string::npos);
		gpuSetShaderDiagnosticCallbackEXT(nullptr, nullptr);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuMemCpy copies every byte it is asked to") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		uint32_t* source = (uint32_t*)gpuMalloc(queue, 4 * sizeof(uint32_t), 16, MEMORY_DEFAULT);
		uint32_t* destination = (uint32_t*)gpuMalloc(queue, 4 * sizeof(uint32_t), 16, MEMORY_READBACK);
		for(uint32_t i = 0; i < 4; ++i) source[i] = i + 1;
		gpu* source_gpu = gpuHostToDevicePointer(queue, source);
		gpu* destination_gpu = gpuHostToDevicePointer(queue, destination);

		GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, source_gpu);
		gpuMemCpy(cmd, destination_gpu, source_gpu, 4 * sizeof(uint32_t));
		gpuSyncMemoryEXT(cmd, destination_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		for(uint32_t i = 0; i < 4; ++i) assert(destination[i] == i + 1);

		gpuFree(queue, source);
		gpuFree(queue, destination);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCopyFromTexture reads back a cleared render target") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment color = {.texture = texture, .loadOp = LOAD_OP_CLEAR, .clearValue = {0, 1, 0, 1}};
		auto cmd = gpuStartCommandRecording(queue);
		gpuBeginRenderPass(cmd, {.colorAttachments = {&color, 1}});
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(pixels[0] == 0xff00ff00 && pixels[64 * 64 - 1] == 0xff00ff00); // Opaque green, as RGBA bytes

		gpuFree(queue, pixels);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCopyToTexture round trips through gpuCopyFromTexture") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_R32_UINT, .usage = TEXTURE_USAGE_FLAGS(USAGE_TRANSFER_SRC | USAGE_TRANSFER_DST)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		auto upload = gpuMalloc<uint32_t>(queue, 64 * 64);
		auto download = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		for(uint32_t i = 0; i < 64 * 64; ++i) upload[i] = i;
		auto upload_gpu = gpuHostToDevicePointer(queue, upload);
		auto download_gpu = gpuHostToDevicePointer(queue, download);

		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, upload_gpu);
		gpuCopyToTexture(cmd, memory, upload_gpu, texture);
		gpuBarrier(cmd, STAGE_TRANSFER, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, download_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, download_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(download[64 * 64 - 1] == 64 * 64 - 1);

		gpuFree(queue, upload);
		gpuFree(queue, download);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCopyToTexture copies every layer of an array") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc desc = {.type = TEXTURE_2D_ARRAY, .dimensions = {4, 4, 1}, .layerCount = 2, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_TRANSFER_SRC | USAGE_TRANSFER_DST)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		auto upload = gpuMalloc<uint32_t>(queue, 2 * 4 * 4);
		auto download = gpuMalloc<uint32_t>(queue, 2 * 4 * 4, MEMORY_READBACK);
		for(uint32_t i = 0; i < 2 * 4 * 4; ++i) upload[i] = i;
		auto upload_gpu = gpuHostToDevicePointer(queue, upload);
		auto download_gpu = gpuHostToDevicePointer(queue, download);

		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, upload_gpu);
		gpuCopyToTexture(cmd, memory, upload_gpu, texture);
		gpuBarrier(cmd, STAGE_TRANSFER, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, download_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, download_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		for(uint32_t i = 0; i < 2 * 4 * 4; ++i) assert(download[i] == i);

		gpuFree(queue, upload);
		gpuFree(queue, download);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCopyFromTexture reads back depth and stencil") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc depth_desc = {.dimensions = {64, 64, 1}, .format = FORMAT_D32_FLOAT, .usage = TEXTURE_USAGE_FLAGS(USAGE_DEPTH_STENCIL_ATTACHMENT | USAGE_TRANSFER_SRC)};
		GpuTextureDesc stencil_desc = {.dimensions = {256, 4, 1}, .format = FORMAT_S8_UINT, .usage = TEXTURE_USAGE_FLAGS(USAGE_DEPTH_STENCIL_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto depth_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, depth_desc));
		auto stencil_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, stencil_desc));
		auto depth = gpuCreateTexture(queue, depth_desc, depth_memory);
		auto stencil = gpuCreateTexture(queue, stencil_desc, stencil_memory);
		auto depths = gpuMalloc<float>(queue, 64 * 64, MEMORY_READBACK);
		auto stencils = gpuMalloc<uint8_t>(queue, 256 * 4, MEMORY_READBACK);
		auto depths_gpu = gpuHostToDevicePointer(queue, depths);
		auto stencils_gpu = gpuHostToDevicePointer(queue, stencils);

		auto cmd = gpuStartCommandRecording(queue);
		gpuBeginRenderPass(cmd, {.depthAttachment = GpuDepthStencilAttachment{.texture = depth, .loadOp = LOAD_OP_CLEAR, .clearValue = 0.5}});
		gpuEndRenderPass(cmd);
		gpuBeginRenderPass(cmd, {.stencilAttachment = GpuDepthStencilAttachment{.texture = stencil, .loadOp = LOAD_OP_CLEAR, .clearValue = 7}});
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_DEPTH_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, depths_gpu, depth_memory, depth);
		gpuCopyFromTexture(cmd, stencils_gpu, stencil_memory, stencil);
		gpuSyncMemoryEXT(cmd, depths_gpu);
		gpuSyncMemoryEXT(cmd, stencils_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(depths[0] == 0.5f && stencils[256 * 4 - 1] == 7);

		gpuFree(queue, stencils);
		gpuFree(queue, depths);
		gpuFree(queue, stencil_memory);
		gpuFree(queue, depth_memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuBlitTextureEXT rescales and converts formats") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc source_desc = {.dimensions = {128, 128, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_COLOR_ATTACHMENT)};
		GpuTextureDesc destination_desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA16_FLOAT, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto source_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, source_desc));
		auto destination_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, destination_desc));
		auto source = gpuCreateTexture(queue, source_desc, source_memory);
		auto destination = gpuCreateTexture(queue, destination_desc, destination_memory);
		auto pixels = gpuMalloc<uint16_t>(queue, 64 * 64 * 4, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment color = {.texture = source, .loadOp = LOAD_OP_CLEAR, .clearValue = {1, 1, 1, 1}};
		GpuRenderPassDesc pass = {.colorAttachments = {&color, 1}};
		auto cmd = gpuStartCommandRecording(queue);
		gpuBeginRenderPass(cmd, pass);
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_PIXEL_SHADER);
		gpuBlitTextureEXT(cmd, destination, source); // Half the size, and converted to half floats
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, destination_memory, destination);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(pixels[0] == 0x3c00); // 1.0 as a half float

		gpuFree(queue, pixels);
		gpuFree(queue, destination_memory);
		gpuFree(queue, source_memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuBlitTextureEXT round trips with nearest sampling") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		gpu* memory[2];
		GpuTexture* textures[2];
		for(int i = 0; i < 2; ++i) {
			memory[i] = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
			textures[i] = gpuCreateTexture(queue, desc, memory[i]);
		}
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment red = {.texture = textures[0], .loadOp = LOAD_OP_CLEAR, .clearValue = {1, 0, 0, 1}};
		auto there = gpuStartCommandRecording(queue);
		gpuBeginRenderPass(there, {.colorAttachments = {&red, 1}});
		gpuEndRenderPass(there);
		gpuBarrier(there, STAGE_RASTER_COLOR_OUT, STAGE_PIXEL_SHADER);
		gpuBlitTextureEXT(there, textures[1], textures[0]);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&there, 1}));

		auto back = gpuStartCommandRecording(queue);
		gpuBlitTextureEXT(back, textures[0], textures[1], false);
		gpuBarrier(back, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(back, pixels_gpu, memory[0], textures[0]);
		gpuSyncMemoryEXT(back, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&back, 1}));
		assert(pixels[0] == 0xff0000ff && pixels[64 * 64 - 1] == 0xff0000ff);

		gpuFree(queue, pixels);
		for(auto block: memory) gpuFree(queue, block);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuBlitTextureEXT keeps the source's orientation") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc source_desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_TRANSFER_DST)};
		GpuTextureDesc destination_desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto source_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, source_desc));
		auto destination_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, destination_desc));
		auto source = gpuCreateTexture(queue, source_desc, source_memory);
		auto destination = gpuCreateTexture(queue, destination_desc, destination_memory);

		// Red, green, blue and white quadrants, from the top left in reading order (RGBA bytes)
		constexpr uint32_t quadrants[4] = {0xff0000ff, 0xff00ff00, 0xffff0000, 0xffffffff};
		auto texels = gpuMalloc<uint32_t>(queue, 64 * 64);
		auto texels_gpu = gpuHostToDevicePointer(queue, texels);
		for(uint32_t y = 0; y < 64; ++y)
			for(uint32_t x = 0; x < 64; ++x)
				texels[y * 64 + x] = quadrants[(y / 32) * 2 + x / 32];
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, texels_gpu);
		gpuCopyToTexture(cmd, source_memory, texels_gpu, source);
		gpuBarrier(cmd, STAGE_TRANSFER, STAGE_PIXEL_SHADER);
		gpuBlitTextureEXT(cmd, destination, source, false);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, destination_memory, destination);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		CHECK_EQ(pixels[0], quadrants[0]);
		CHECK_EQ(pixels[63], quadrants[1]);
		CHECK_EQ(pixels[63 * 64], quadrants[2]);
		CHECK_EQ(pixels[64 * 64 - 1], quadrants[3]);

		gpuFree(queue, pixels);
		gpuFree(queue, texels);
		gpuFree(queue, destination_memory);
		gpuFree(queue, source_memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuBlitTextureEXT falls back to nearest for unfilterable formats") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc source_desc = {.dimensions = {16, 16, 1}, .format = FORMAT_RGBA32_FLOAT, .usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_COLOR_ATTACHMENT)};
		GpuTextureDesc destination_desc = {.dimensions = {16, 16, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_COLOR_ATTACHMENT};
		auto source_memory = (gpu*)gpuMalloc(queue, 16 * 16 * 16, 256, MEMORY_TEXTURE);
		auto destination_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, destination_desc));
		auto destination = gpuCreateTexture(queue, destination_desc, destination_memory);
		if(auto source = gpuCreateTexture(queue, source_desc, source_memory)) {
			auto cmd = gpuStartCommandRecording(queue);
			gpuBlitTextureEXT(cmd, destination, source); // Linear asked for, nearest taken
			gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		}
		gpuFree(queue, destination_memory);
		gpuFree(queue, source_memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuSetActiveTextureHeapPtr: a dispatch samples a texture through the heap") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_TRANSFER_DST)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		auto texels = gpuMalloc<uint32_t>(queue, 64 * 64);
		for(uint32_t i = 0; i < 64 * 64; ++i) texels[i] = 0xff0000ff; // Opaque red, as RGBA bytes
		auto texels_gpu = gpuHostToDevicePointer(queue, texels);

		auto heap = gpuMalloc<GpuTextureDescriptor>(queue, 1, MEMORY_DESCRIPTOR_HEAP);
		heap[0] = gpuTextureViewDescriptor(queue, texture, {});
		auto heap_gpu = gpuHostToDevicePointer(queue, heap);

		auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
			import noapi;
			[shader("compute")] [numthreads(1, 1, 1)]
			void main() { gpuComputeData<float4>()[0] = gpuSample2D(0, gpuGetSamplerIndex(gpuDefaultSampler()), float2(0.5, 0.5)); }
		)slang"));
		auto color = gpuMalloc<float>(queue, 4, MEMORY_READBACK);
		auto color_gpu = gpuHostToDevicePointer(queue, color);

		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, texels_gpu);
		gpuCopyToTexture(cmd, memory, texels_gpu, texture);
		gpuBarrier(cmd, STAGE_TRANSFER, STAGE_COMPUTE);
		gpuSyncMemoryEXT(cmd, heap_gpu);
		gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
		gpuSetPipeline(cmd, pipeline);
		gpuDispatch(cmd, color_gpu, {1, 1, 1});
		gpuSyncMemoryEXT(cmd, color_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(color[0] == 1.0f && color[1] == 0.0f && color[3] == 1.0f);

		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, color);
		gpuFree(queue, heap);
		gpuFree(queue, texels);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuSetActiveTextureHeapPtr snapshots a heap in ordinary memory each time it is set") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_TRANSFER_DST)};
		gpu* memory[2];
		GpuTexture* textures[2];
		auto texels = gpuMalloc<uint32_t>(queue, 2 * 64 * 64);
		for(uint32_t i = 0; i < 64 * 64; ++i) texels[i] = 0xff0000ff, texels[64 * 64 + i] = 0xff00ff00; // Red, then green
		auto texels_gpu = gpuHostToDevicePointer(queue, texels);
		auto heap = gpuMalloc<GpuTextureDescriptor>(queue, 1, MEMORY_DEFAULT);
		auto heap_gpu = gpuHostToDevicePointer(queue, heap);
		auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
			import noapi;
			[shader("compute")] [numthreads(1, 1, 1)]
			void main() { gpuComputeData<float4>()[0] = gpuSample2D(0, gpuGetSamplerIndex(gpuDefaultSampler()), float2(0.5, 0.5)); }
		)slang"));
		auto color = gpuMalloc<float>(queue, 4, MEMORY_READBACK);
		auto color_gpu = gpuHostToDevicePointer(queue, color);

		auto upload = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(upload, texels_gpu);
		for(int i = 0; i < 2; ++i) {
			memory[i] = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
			textures[i] = gpuCreateTexture(queue, desc, memory[i]);
			gpuCopyToTexture(upload, memory[i], (gpu*)((char*)texels_gpu + i * 64 * 64 * 4), textures[i]);
		}
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&upload, 1}));

		for(int i = 0; i < 2; ++i) {
			heap[0] = gpuTextureViewDescriptor(queue, textures[i], {});
			auto cmd = gpuStartCommandRecording(queue);
			gpuSyncMemoryEXT(cmd, heap_gpu);
			gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
			gpuSetPipeline(cmd, pipeline);
			gpuDispatch(cmd, color_gpu, {1, 1, 1});
			gpuSyncMemoryEXT(cmd, color_gpu);
			gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
			assert(color[0] == (i == 0 ? 1.0f : 0.0f) && color[1] == (i == 0 ? 0.0f : 1.0f));
		}

		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, color);
		gpuFree(queue, heap);
		gpuFree(queue, texels);
		for(auto block: memory) gpuFree(queue, block);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuSignalAfter always sets its counter, and takes the max of it where split_barrier_signals says it can") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto counter = gpuMalloc<uint64_t>(queue, 1, MEMORY_READBACK);
		auto counter_gpu = gpuHostToDevicePointer(queue, counter);

		auto cmd = gpuStartCommandRecording(queue);
		gpuSignalAfter(cmd, STAGE_COMPUTE, counter_gpu, 7, SIGNAL_ATOMIC_SET);
		gpuWaitBefore(cmd, STAGE_COMPUTE, counter_gpu, 7, OP_GREATER_EQUAL);
		gpuSignalAfter(cmd, STAGE_COMPUTE, counter_gpu, 9, SIGNAL_ATOMIC_MAX);
		gpuWaitBefore(cmd, STAGE_TRANSFER, counter_gpu, 9, OP_GREATER_EQUAL);
		gpuSyncMemoryEXT(cmd, counter_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(*counter == (gpuGetCapabilitiesEXT(queue).split_barrier_signals ? 9 : 7));

		gpuFree(queue, counter);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuSignalAfter and gpuWaitBefore report what they can't honor inside a render pass") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		int reports = 0;
		gpuSetDiagnosticCallbackEXT(queue, [](GpuQueue*, GPU_DIAGNOSTIC, GpuStringView, void* count) { ++*(int*)count; }, &reports);
		auto counter = gpuMalloc<uint64_t>(queue);
		auto counter_gpu = gpuHostToDevicePointer(queue, counter);
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = USAGE_COLOR_ATTACHMENT};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);

		GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
		auto cmd = gpuStartCommandRecording(queue);
		gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
		gpuSignalAfter(cmd, STAGE_RASTER_COLOR_OUT, counter_gpu, 1, SIGNAL_ATOMIC_SET);
		gpuWaitBefore(cmd, STAGE_PIXEL_SHADER, counter_gpu, 1, OP_GREATER_EQUAL);
		gpuEndRenderPass(cmd);
		gpuSignalAfter(cmd, STAGE_COMPUTE, counter_gpu, 2, SIGNAL_ATOMIC_SET);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(reports > 0);

		gpuSetDiagnosticCallbackEXT(queue, nullptr, nullptr);
		gpuFree(queue, memory);
		gpuFree(queue, counter);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuDispatch hands the shader its root data struct") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		struct Root { gpu* values; };
		auto values = gpuMalloc<uint32_t>(queue, 16, MEMORY_READBACK);
		auto root = gpuMalloc<Root>(queue);
		root->values = gpuHostToDevicePointer(queue, values);
		auto root_gpu = gpuHostToDevicePointer(queue, root);

		auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
			import noapi;
			struct Root : IGpuLoadable {
				GpuPtr<uint> values;
				static const uint gpuStride = 8;
				static Root gpuLoad(GpuAddress at) { var from = GpuReader(at); Root out; out.values = from.next<GpuPtr<uint> >(); return out; }
			}
			[shader("compute")] [numthreads(16, 1, 1)]
			void main(uint3 thread : SV_DispatchThreadID) { let root = *gpuComputeData<Root>(); root.values[thread.x] = thread.x * 2; }
		)slang"));
		assert(pipeline);

		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, root_gpu);
		gpuSetPipeline(cmd, pipeline);
		gpuDispatch(cmd, root_gpu, {1, 1, 1});
		gpuSyncMemoryEXT(cmd, root->values);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(values[5] == 10);

		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, root);
		gpuFree(queue, values);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuDispatch keeps the pipeline bound across copies") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto counter = gpuMalloc<uint32_t>(queue);
		*counter = 0;
		auto counter_gpu = gpuHostToDevicePointer(queue, counter);
		auto counts = gpuMalloc<uint32_t>(queue, 2, MEMORY_READBACK);
		auto counts_gpu = gpuHostToDevicePointer(queue, counts);
		auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
			import noapi;
			[shader("compute")] [numthreads(1, 1, 1)] void main() { gpuComputeData<uint>()[0] += 1; }
		)slang"));

		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, counter_gpu);
		gpuSetPipeline(cmd, pipeline);
		for(int i = 0; i < 2; ++i) {
			gpuDispatch(cmd, counter_gpu, {1, 1, 1});
			gpuBarrier(cmd, STAGE_COMPUTE, STAGE_TRANSFER);
			gpuMemCpy(cmd, (gpu*)((char*)counts_gpu + i * 4), counter_gpu, 4); // The count so far
			gpuBarrier(cmd, STAGE_TRANSFER, STAGE_COMPUTE);
		}
		gpuSyncMemoryEXT(cmd, counts_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(counts[0] == 1 && counts[1] == 2);

		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, counts);
		gpuFree(queue, counter);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuDispatchIndirect launches the grid it reads from GPU memory") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto values = gpuMalloc<uint32_t>(queue, 64, MEMORY_READBACK);
		auto values_gpu = gpuHostToDevicePointer(queue, values);
		auto grid = gpuMalloc<uvec3>(queue);
		*grid = {4, 1, 1};
		auto grid_gpu = gpuHostToDevicePointer(queue, grid);
		auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
			import noapi;
			[shader("compute")] [numthreads(16, 1, 1)]
			void main(uint3 thread : SV_DispatchThreadID) { gpuComputeData<uint>()[thread.x] = 1; }
		)slang"));
		assert(pipeline);

		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, grid_gpu);
		gpuSetPipeline(cmd, pipeline);
		gpuDispatchIndirect(cmd, values_gpu, grid_gpu);
		gpuSyncMemoryEXT(cmd, values_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(values[0] == 1 && values[63] == 1); // Four groups of 16

		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, grid);
		gpuFree(queue, values);
	}
	noapi_doctest_end();
}

// graphics.h

TEST_CASE("CULL: each mode culls the windings it names") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			// Counter-clockwise: (-1, -1), then (3, -1), then (-1, 3)
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
		)slang");
		GpuTextureDesc desc = {.dimensions = {4, 4, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		auto indices = gpuMalloc<uint32_t>(queue, 3);
		for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
		auto indices_gpu = gpuHostToDevicePointer(queue, indices);
		auto pixels = gpuMalloc<uint32_t>(queue, 4 * 4, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};

		for(auto [cull, drawn]: {std::pair{CULL_NONE, true}, {CULL_CW, true}, {CULL_CCW, false}, {CULL_ALL, false}}) {
			auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.cull = cull, .colorTargets = {&target, 1}});
			GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
			auto cmd = gpuStartCommandRecording(queue);
			gpuSyncMemoryEXT(cmd, indices_gpu);
			gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
			gpuSetPipeline(cmd, pipeline);
			gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
			gpuEndRenderPass(cmd);
			gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
			gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
			gpuSyncMemoryEXT(cmd, pixels_gpu);
			gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
			assert((pixels[0] == 0xffffffff) == drawn);
			gpuFreePipeline(queue, pipeline);
		}

		gpuFree(queue, pixels);
		gpuFree(queue, indices);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("STENCIL_OP: each op leaves the value it names") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
		)slang");
		std::pair<STENCIL_OP, uint32_t> cases[] = {{STENCIL_OP_KEEP, 5}, {STENCIL_OP_ZERO, 0}, {STENCIL_OP_REPLACE, 9}, {STENCIL_OP_INCR_SAT, 6},
			{STENCIL_OP_DECR_SAT, 4}, {STENCIL_OP_INVERT, 250}, {STENCIL_OP_INCR_WRAP, 6}, {STENCIL_OP_DECR_WRAP, 4}};
		GpuTextureDesc color_desc = {.dimensions = {64, 1, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		GpuTextureDesc stencil_desc = {.dimensions = {64, 1, 1}, .format = FORMAT_S8_UINT, .usage = USAGE_DEPTH_STENCIL_ATTACHMENT};
		auto color_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, color_desc));
		auto stencil_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, stencil_desc));
		auto color = gpuCreateTexture(queue, color_desc, color_memory);
		auto stencil = gpuCreateTexture(queue, stencil_desc, stencil_memory);
		assert(stencil);
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.stencilFormat = FORMAT_S8_UINT, .colorTargets = {&target, 1}});
		auto indices = gpuMalloc<uint32_t>(queue, 3);
		for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
		auto indices_gpu = gpuHostToDevicePointer(queue, indices);
		auto pixels = gpuMalloc<uint32_t>(queue, 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment color_attachment = {.texture = color, .loadOp = LOAD_OP_CLEAR};
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, indices_gpu);
		gpuBeginRenderPass(cmd, {.stencilAttachment = GpuDepthStencilAttachment{.texture = stencil, .loadOp = LOAD_OP_CLEAR, .clearValue = 5}, .colorAttachments = {&color_attachment, 1}});
		gpuSetPipeline(cmd, pipeline);
		std::vector<GpuDepthStencilState*> states;
		for(size_t i = 0; i < std::size(cases); ++i) {
			auto [op, expected] = cases[i];
			// No back faces are drawn, so their reference is never used; WebGPU, which has only the one, reports that it ignores it
			states.push_back(gpuCreateDepthStencilState(queue, {.stencilFront = {.passOp = op, .reference = 9}, .stencilBack = {.passOp = op, .reference = 1}}));
			states.push_back(gpuCreateDepthStencilState(queue, {.stencilWriteMask = 0, .stencilFront = {.test = OP_EQUAL, .reference = uint8_t(expected)}, .stencilBack = {.test = OP_EQUAL, .reference = uint8_t(expected)}}));
			gpuSetViewportEXT(cmd, {1, 1}, {int32_t(i), 0});
			gpuSetDepthStencilState(cmd, states[states.size() - 2]);
			gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
			gpuSetDepthStencilState(cmd, states.back());
			gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
		}
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, color_memory, color);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		for(size_t i = 0; i < std::size(cases); ++i)
			assert(pixels[i] == 0xffffffff);

		for(auto state: states) gpuFreeDepthStencilState(queue, state);
		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, pixels);
		gpuFree(queue, indices);
		gpuFree(queue, stencil_memory);
		gpuFree(queue, color_memory);
	}
	noapi_doctest_end();
}

TEST_CASE("INDEX_TYPE_EXT: 8 bit indices draw, or report themselves unsupported") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			// A quad as a strip: (-1, -1), (1, -1), (-1, 1), (1, 1)
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2(index & 1, index >> 1) * 2 - 1, 0, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
		)slang");
		int unsupported = 0;
		gpuSetDiagnosticCallbackEXT(queue, [](GpuQueue*, GPU_DIAGNOSTIC kind, GpuStringView, void* count) {
			if(kind == GPU_DIAGNOSTIC_UNSUPPORTED) ++*(int*)count;
		}, &unsupported);
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.topology = TOPOLOGY_TRIANGLE_STRIP, .colorTargets = {&target, 1}});
		auto indices = gpuMalloc<uint8_t>(queue, 4);
		for(uint8_t i = 0; i < 4; ++i) indices[i] = i;
		auto indices_gpu = gpuHostToDevicePointer(queue, indices);
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, indices_gpu);
		gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
		gpuSetPipeline(cmd, pipeline);
		gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 4, 1, INDEX_TYPE_UINT8);
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert((pixels[0] == 0xffffffff && pixels[64 * 64 - 1] == 0xffffffff) == (unsupported == 0));

		gpuSetDiagnosticCallbackEXT(queue, nullptr, nullptr);
		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, pixels);
		gpuFree(queue, indices);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("GpuDepthStencilDesc: every comparison as a depth test") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, gpuVertexData<float>()[0], 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
		)slang");
		OP ops[] = {OP_NEVER, OP_LESS, OP_EQUAL, OP_LESS_EQUAL, OP_GREATER, OP_NOT_EQUAL, OP_GREATER_EQUAL, OP_ALWAYS};
		const char* passes[] = {"---", "x--", "-x-", "xx-", "--x", "x-x", "-xx", "xxx"}; // For 0.25, 0.5 and 0.75
		GpuTextureDesc color_desc = {.dimensions = {64, 1, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		GpuTextureDesc depth_desc = {.dimensions = {64, 1, 1}, .format = FORMAT_D32_FLOAT, .usage = USAGE_DEPTH_STENCIL_ATTACHMENT};
		auto color_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, color_desc));
		auto depth_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, depth_desc));
		auto color = gpuCreateTexture(queue, color_desc, color_memory);
		auto depth = gpuCreateTexture(queue, depth_desc, depth_memory);
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.depthFormat = FORMAT_D32_FLOAT, .colorTargets = {&target, 1}});
		auto depths = gpuMalloc<float>(queue, 3 * 4); // One 16 byte root data struct per depth
		auto indices = gpuMalloc<uint32_t>(queue, 3);
		for(uint32_t i = 0; i < 3; ++i) depths[i * 4] = 0.25f * (i + 1), indices[i] = i;
		auto depths_gpu = gpuHostToDevicePointer(queue, depths);
		auto indices_gpu = gpuHostToDevicePointer(queue, indices);
		auto pixels = gpuMalloc<uint32_t>(queue, 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment color_attachment = {.texture = color, .loadOp = LOAD_OP_CLEAR};
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, depths_gpu);
		gpuSyncMemoryEXT(cmd, indices_gpu);
		gpuBeginRenderPass(cmd, {.depthAttachment = GpuDepthStencilAttachment{.texture = depth, .loadOp = LOAD_OP_CLEAR, .clearValue = 0.5}, .colorAttachments = {&color_attachment, 1}});
		gpuSetPipeline(cmd, pipeline);
		std::vector<GpuDepthStencilState*> states;
		for(size_t op = 0; op < std::size(ops); ++op) {
			states.push_back(gpuCreateDepthStencilState(queue, {.depthMode = DEPTH_READ, .depthTest = ops[op]}));
			gpuSetDepthStencilState(cmd, states.back());
			for(int k = 0; k < 3; ++k) {
				gpuSetViewportEXT(cmd, {1, 1}, {int32_t(op * 3 + k), 0});
				gpuDrawIndexedInstanced(cmd, (gpu*)((char*)depths_gpu + k * 16), indices_gpu, indices_gpu, 3, 1);
			}
		}
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, color_memory, color);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		for(size_t op = 0; op < std::size(ops); ++op)
			for(int k = 0; k < 3; ++k)
				assert((pixels[op * 3 + k] == 0xffffffff) == (passes[op][k] == 'x'));

		for(auto state: states) gpuFreeDepthStencilState(queue, state);
		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, pixels);
		gpuFree(queue, indices);
		gpuFree(queue, depths);
		gpuFree(queue, depth_memory);
		gpuFree(queue, color_memory);
	}
	noapi_doctest_end();
}

TEST_CASE("GpuBlendDesc: every equation against every factor") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(0.75, 0, 0, 0.5); }
		)slang");
		BLEND ops[] = {BLEND_ADD, BLEND_SUBTRACT, BLEND_REV_SUBTRACT, BLEND_MIN, BLEND_MAX};
		FACTOR factors[] = {FACTOR_ZERO, FACTOR_ONE, FACTOR_SRC_COLOR, FACTOR_ONE_MINUS_SRC_COLOR, FACTOR_DST_COLOR, FACTOR_ONE_MINUS_DST_COLOR,
			FACTOR_SRC_ALPHA, FACTOR_ONE_MINUS_SRC_ALPHA, FACTOR_DST_ALPHA, FACTOR_ONE_MINUS_DST_ALPHA};
		const float src = 0.75f, src_alpha = 0.5f, dst = 0.5f, dst_alpha = 0.25f;
		const float weight[] = {0, 1, src, 1 - src, dst, 1 - dst, src_alpha, 1 - src_alpha, dst_alpha, 1 - dst_alpha}; // Each factor's value for red

		GpuTextureDesc desc = {.dimensions = {64, 1, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
		auto indices = gpuMalloc<uint32_t>(queue, 3);
		for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
		auto indices_gpu = gpuHostToDevicePointer(queue, indices);
		auto pixels = gpuMalloc<uint32_t>(queue, 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR, .clearValue = {dst, 0, 0, dst_alpha}};
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, indices_gpu);
		gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
		std::vector<GpuBlendState*> states;
		for(size_t op = 0; op < std::size(ops); ++op)
			for(size_t f = 0; f < std::size(factors); ++f) {
				states.push_back(gpuCreateBlendState(queue, {.colorOp = ops[op], .srcColorFactor = factors[f], .dstColorFactor = factors[(f + 3) % std::size(factors)]}));
				gpuSetBlendState(cmd, states.back()); // Before the pipeline, which keeps it rather than putting back its own
				gpuSetPipeline(cmd, pipeline);
				gpuSetViewportEXT(cmd, {1, 1}, {int32_t(op * std::size(factors) + f), 0});
				gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
			}
		for(int i = 0; i < 2; ++i) { // The same color equation, so only the alpha half tells these two apart
			states.push_back(gpuCreateBlendState(queue, {.srcAlphaFactor = i ? FACTOR_ZERO : FACTOR_ONE}));
			gpuSetBlendState(cmd, states.back());
			gpuSetViewportEXT(cmd, {1, 1}, {int32_t(std::size(ops) * std::size(factors) + i), 0});
			gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
		}
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));

		for(size_t op = 0; op < std::size(ops); ++op)
			for(size_t f = 0; f < std::size(factors); ++f) {
				float s = src * weight[f], d = dst * weight[(f + 3) % std::size(factors)];
				float expected[] = {s + d, s - d, d - s, std::min(src, dst), std::max(src, dst)}; // MIN and MAX ignore the factors
				float red = (pixels[op * std::size(factors) + f] & 0xff) / 255.0f;
				assert(std::abs(red - std::clamp(expected[op], 0.0f, 1.0f)) < 0.02f);
			}
		uint32_t last = std::size(ops) * std::size(factors);
		assert((pixels[last] >> 24) == 0x80 && (pixels[last + 1] >> 24) == 0); // Alpha written as is, then zeroed

		for(auto state: states) gpuFreeBlendState(queue, state);
		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, pixels);
		gpuFree(queue, indices);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("GpuRasterDesc: dual source blending weighs the destination by the second output") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			struct Output { [[vk::location(0), vk::index(0)]] float4 color : SV_Target0; [[vk::location(0), vk::index(1)]] float4 weight : SV_Target1; }
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
			[shader("fragment")] Output fragmentMain() { return {float4(1, 0, 0, 1), float4(0.25)}; }
		)slang");
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.supportDualSourceBlending = true, .colorTargets = {&target, 1}});
		assert((pipeline != nullptr) == gpuGetCapabilitiesEXT(queue).dual_source_blending);
		if(pipeline) {
			GpuTextureDesc desc = {.dimensions = {64, 1, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
			auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
			auto texture = gpuCreateTexture(queue, desc, memory);
			auto indices = gpuMalloc<uint32_t>(queue, 3);
			for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
			auto indices_gpu = gpuHostToDevicePointer(queue, indices);
			auto pixels = gpuMalloc<uint32_t>(queue, 64, MEMORY_READBACK);
			auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);
			GpuBlendState* states[] = {
				gpuCreateBlendState(queue, {.dstColorFactor = FACTOR_SRC1_COLOR, .srcAlphaFactor = FACTOR_SRC1_ALPHA, .dstAlphaFactor = FACTOR_ONE_MINUS_SRC1_ALPHA}),
				gpuCreateBlendState(queue, {.dstColorFactor = FACTOR_ONE_MINUS_SRC1_COLOR}),
			};

			GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR, .clearValue = {0, 0, 1, 1}};
			auto cmd = gpuStartCommandRecording(queue);
			gpuSyncMemoryEXT(cmd, indices_gpu);
			gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
			gpuSetPipeline(cmd, pipeline);
			for(int i = 0; i < 2; ++i) {
				gpuSetBlendState(cmd, states[i]);
				gpuSetViewportEXT(cmd, {1, 1}, {i, 0});
				gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
			}
			gpuEndRenderPass(cmd);
			gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
			gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
			gpuSyncMemoryEXT(cmd, pixels_gpu);
			gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
			assert((pixels[0] & 0xff) == 0xff && ((pixels[0] >> 16) & 0xff) == 0x40 && (pixels[0] >> 24) == 0xff); // Red, a quarter of the blue, opaque
			assert(((pixels[1] >> 16) & 0xff) == 0xbf); // Three quarters of the blue

			for(auto state: states) gpuFreeBlendState(queue, state);
			gpuFreePipeline(queue, pipeline);
			gpuFree(queue, pixels);
			gpuFree(queue, indices);
			gpuFree(queue, memory);
		}
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCreateGraphicsPipeline fails without both entry points") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return gpuFragmentData<float4>()[0]; }
		)slang");
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {
			.colorTargets = {&target, 1},
			.blendstate = GpuBlendDesc{.srcColorFactor = FACTOR_SRC_ALPHA, .dstColorFactor = FACTOR_ONE_MINUS_SRC_ALPHA},
		});
		assert(pipeline);
		assert(!gpuCreateGraphicsPipeline(queue, string_to_bytes("not a shader"), shader, {.colorTargets = {&target, 1}}));
		auto vertex_only = string_to_bytes(R"slang([shader("vertex")] float4 vertexMain() : SV_Position { return float4(0, 0, 0, 1); })slang");
		auto fragment_only = string_to_bytes(R"slang([shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); })slang");
		assert(!gpuCreateGraphicsPipeline(queue, fragment_only, fragment_only, {.colorTargets = {&target, 1}})); // No vertex entry point
		assert(!gpuCreateGraphicsPipeline(queue, vertex_only, vertex_only, {.colorTargets = {&target, 1}})); // No fragment entry point
		gpuFreePipeline(queue, pipeline);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCreateGraphicsMeshletPipeline follows the mesh shader capability") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			struct Vertex { float4 position : SV_Position; }
			[shader("mesh")] [outputtopology("triangle")] [numthreads(1, 1, 1)]
			void meshMain(out vertices Vertex corners[3], out indices uint3 triangles[1]) {
				SetMeshOutputCounts(3, 1);
				corners[0].position = float4(-1, -1, 0, 1); corners[1].position = float4(3, -1, 0, 1); corners[2].position = float4(-1, 3, 0, 1);
				triangles[0] = uint3(0, 1, 2);
			}
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
		)slang");
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsMeshletPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
		assert((pipeline != nullptr) == gpuGetCapabilitiesEXT(queue).mesh_shaders);
		if(pipeline) gpuFreePipeline(queue, pipeline);
		assert(!gpuCreateGraphicsMeshletPipeline(queue, string_to_bytes("void helper() {}"), shader, {.colorTargets = {&target, 1}})); // No mesh entry point
	}
	noapi_doctest_end();
}

TEST_CASE("gpuSetDepthStencilState: a triangle behind the cleared depth is not drawn") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0.5, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
		)slang");
		GpuTextureDesc color_desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		GpuTextureDesc depth_desc = {.dimensions = {64, 64, 1}, .format = FORMAT_D32_FLOAT, .usage = USAGE_DEPTH_STENCIL_ATTACHMENT};
		auto color_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, color_desc));
		auto depth_memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, depth_desc));
		auto color = gpuCreateTexture(queue, color_desc, color_memory);
		auto depth = gpuCreateTexture(queue, depth_desc, depth_memory);
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.depthFormat = FORMAT_D32_FLOAT, .colorTargets = {&target, 1}});
		auto closer = gpuCreateDepthStencilState(queue, {.depthMode = DEPTH_READ, .depthTest = OP_LESS}); // 0.5 is not closer than the 0.25 cleared
		auto indices = gpuMalloc<uint32_t>(queue, 3);
		for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
		auto indices_gpu = gpuHostToDevicePointer(queue, indices);
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment color_attachment = {.texture = color, .loadOp = LOAD_OP_CLEAR};
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, indices_gpu);
		gpuBeginRenderPass(cmd, {.depthAttachment = GpuDepthStencilAttachment{.texture = depth, .loadOp = LOAD_OP_CLEAR, .clearValue = 0.25}, .colorAttachments = {&color_attachment, 1}});
		gpuSetPipeline(cmd, pipeline);
		gpuSetDepthStencilState(cmd, closer);
		gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, color_memory, color);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(pixels[0] == 0xff000000); // Still the clear color: the triangle failed the depth test

		gpuFreeDepthStencilState(queue, closer);
		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, pixels);
		gpuFree(queue, indices);
		gpuFree(queue, depth_memory);
		gpuFree(queue, color_memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuSetBlendState overrides the pipeline's blend state") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 0, 0, 1); }
		)slang");
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
		auto additive = gpuCreateBlendState(queue, {.srcColorFactor = FACTOR_ONE, .dstColorFactor = FACTOR_ONE});
		auto indices = gpuMalloc<uint32_t>(queue, 3);
		for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
		auto indices_gpu = gpuHostToDevicePointer(queue, indices);
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR, .clearValue = {0, 0, 1, 1}};
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, indices_gpu);
		gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
		gpuSetPipeline(cmd, pipeline);
		gpuSetBlendState(cmd, additive);
		gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(pixels[0] == 0xffff00ff); // Red added onto the blue it was cleared to

		gpuFreeBlendState(queue, additive);
		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, pixels);
		gpuFree(queue, indices);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuSetViewportEXT limits drawing to the viewport") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
		)slang");
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
		auto indices = gpuMalloc<uint32_t>(queue, 3);
		for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
		auto indices_gpu = gpuHostToDevicePointer(queue, indices);
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, indices_gpu);
		gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
		gpuSetViewportEXT(cmd, {32, 64}); // The left half
		gpuSetPipeline(cmd, pipeline);
		gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(pixels[0] == 0xffffffff && pixels[63] == 0xff000000);

		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, pixels);
		gpuFree(queue, indices);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuSetScissorRectEXT clips drawing to the scissor") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
		)slang");
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
		auto indices = gpuMalloc<uint32_t>(queue, 3);
		for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
		auto indices_gpu = gpuHostToDevicePointer(queue, indices);
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, indices_gpu);
		gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
		gpuSetScissorRectEXT(cmd, {32, 64}, {32, 0}); // Only the right half is drawn
		gpuSetPipeline(cmd, pipeline);
		gpuDrawIndexedInstanced(cmd, indices_gpu, indices_gpu, indices_gpu, 3, 1);
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(pixels[0] == 0xff000000 && pixels[63] == 0xffffffff);

		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, pixels);
		gpuFree(queue, indices);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuBeginRenderPass: LOAD_OP_LOAD keeps what the last pass cleared") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment clear = {.texture = texture, .loadOp = LOAD_OP_CLEAR, .clearValue = {1, 0, 0, 1}};
		GpuColorAttachment keep = {.texture = texture, .loadOp = LOAD_OP_LOAD};
		auto cmd = gpuStartCommandRecording(queue);
		gpuBeginRenderPass(cmd, {.colorAttachments = {&clear, 1}});
		gpuEndRenderPass(cmd);
		gpuBeginRenderPass(cmd, {.colorAttachments = {&keep, 1}}); // Picks up where the last pass left off
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(pixels[0] == 0xff0000ff);

		gpuFree(queue, pixels);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuDrawIndexedInstanced draws with 16 bit indices and fragment data") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return gpuFragmentData<float4>()[0]; }
		)slang");
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});

		struct Color { float r, g, b, a; };
		auto color = gpuMalloc<Color>(queue);
		*color = {1, 0, 1, 1};
		auto indices = gpuMalloc<uint16_t>(queue, 4);
		for(uint16_t i = 0; i < 3; ++i) indices[i] = i;
		auto color_gpu = gpuHostToDevicePointer(queue, color);
		auto indices_gpu = gpuHostToDevicePointer(queue, indices);
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, color_gpu);
		gpuSyncMemoryEXT(cmd, indices_gpu);
		gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
		gpuSetPipeline(cmd, pipeline);
		gpuDrawIndexedInstanced(cmd, color_gpu, color_gpu, indices_gpu, 3, 1, INDEX_TYPE_UINT16);
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(pixels[0] == 0xffff00ff);

		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, pixels);
		gpuFree(queue, indices);
		gpuFree(queue, color);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuDrawIndexedInstancedIndirect reads its arguments from GPU memory") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(float2((index << 1) & 2, index & 2) * 2 - 1, 0, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(0, 0, 1, 1); }
		)slang");
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});

		struct DrawArguments { uint32_t index_count, instance_count, first_index; int32_t vertex_offset; uint32_t first_instance; };
		auto arguments = gpuMalloc<DrawArguments>(queue);
		*arguments = {3, 1, 0, 0, 0};
		auto indices = gpuMalloc<uint32_t>(queue, 3);
		for(uint32_t i = 0; i < 3; ++i) indices[i] = i;
		auto arguments_gpu = gpuHostToDevicePointer(queue, arguments);
		auto indices_gpu = gpuHostToDevicePointer(queue, indices);
		auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
		auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

		GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, arguments_gpu);
		gpuSyncMemoryEXT(cmd, indices_gpu);
		gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
		gpuSetPipeline(cmd, pipeline);
		gpuDrawIndexedInstancedIndirect(cmd, indices_gpu, indices_gpu, indices_gpu, arguments_gpu);
		gpuEndRenderPass(cmd);
		gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
		gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
		gpuSyncMemoryEXT(cmd, pixels_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(pixels[0] == 0xffff0000);

		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, pixels);
		gpuFree(queue, indices);
		gpuFree(queue, arguments);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuDrawMeshlets draws a meshlet") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			struct Vertex { float4 position : SV_Position; }
			[shader("mesh")] [outputtopology("triangle")] [numthreads(1, 1, 1)]
			void meshMain(out vertices Vertex corners[3], out indices uint3 triangles[1]) {
				SetMeshOutputCounts(3, 1);
				corners[0].position = float4(-1, -1, 0, 1); corners[1].position = float4(3, -1, 0, 1); corners[2].position = float4(-1, 3, 0, 1);
				triangles[0] = uint3(0, 1, 2);
			}
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
		)slang");
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsMeshletPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
		if(pipeline) {
			GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
			auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
			auto texture = gpuCreateTexture(queue, desc, memory);
			auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
			auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

			GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
			auto cmd = gpuStartCommandRecording(queue);
			gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
			gpuSetPipeline(cmd, pipeline);
			gpuDrawMeshlets(cmd, pixels_gpu, pixels_gpu, {1, 1, 1});
			gpuEndRenderPass(cmd);
			gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
			gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
			gpuSyncMemoryEXT(cmd, pixels_gpu);
			gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
			assert(pixels[0] == 0xffffffff);

			gpuFreePipeline(queue, pipeline);
			gpuFree(queue, pixels);
			gpuFree(queue, memory);
		}
	}
	noapi_doctest_end();
}

TEST_CASE("gpuDrawMeshletsIndirect reads its grid from GPU memory") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto shader = string_to_bytes(R"slang(
			import noapi;
			struct Vertex { float4 position : SV_Position; }
			[shader("mesh")] [outputtopology("triangle")] [numthreads(1, 1, 1)]
			void meshMain(out vertices Vertex corners[3], out indices uint3 triangles[1]) {
				SetMeshOutputCounts(3, 1);
				corners[0].position = float4(-1, -1, 0, 1); corners[1].position = float4(3, -1, 0, 1); corners[2].position = float4(-1, 3, 0, 1);
				triangles[0] = uint3(0, 1, 2);
			}
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
		)slang");
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto pipeline = gpuCreateGraphicsMeshletPipeline(queue, shader, shader, {.colorTargets = {&target, 1}});
		if(pipeline) {
			GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_SRC)};
			auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
			auto texture = gpuCreateTexture(queue, desc, memory);
			auto grid = gpuMalloc<uvec3>(queue);
			*grid = {1, 1, 1};
			auto grid_gpu = gpuHostToDevicePointer(queue, grid);
			auto pixels = gpuMalloc<uint32_t>(queue, 64 * 64, MEMORY_READBACK);
			auto pixels_gpu = gpuHostToDevicePointer(queue, pixels);

			GpuColorAttachment attachment = {.texture = texture, .loadOp = LOAD_OP_CLEAR};
			auto cmd = gpuStartCommandRecording(queue);
			gpuSyncMemoryEXT(cmd, grid_gpu);
			gpuBeginRenderPass(cmd, {.colorAttachments = {&attachment, 1}});
			gpuSetPipeline(cmd, pipeline);
			gpuDrawMeshletsIndirect(cmd, grid_gpu, grid_gpu, grid_gpu);
			gpuEndRenderPass(cmd);
			gpuBarrier(cmd, STAGE_RASTER_COLOR_OUT, STAGE_TRANSFER);
			gpuCopyFromTexture(cmd, pixels_gpu, memory, texture);
			gpuSyncMemoryEXT(cmd, pixels_gpu);
			gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
			assert(pixels[0] == 0xffffffff);

			gpuFreePipeline(queue, pipeline);
			gpuFree(queue, pixels);
			gpuFree(queue, grid);
			gpuFree(queue, memory);
		}
	}
	noapi_doctest_end();
}

// samplers.h

TEST_CASE("gpuSetEnabledSamplersEXT: shaders sample with the samplers enabled") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_TRANSFER_DST)};
		auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
		auto texture = gpuCreateTexture(queue, desc, memory);
		auto texels = gpuMalloc<uint32_t>(queue, 64 * 64);
		for(uint32_t i = 0; i < 64 * 64; ++i) texels[i] = i % 64 < 32 ? 0xff0000ff : 0xff00ff00; // Red on the left, green on the right
		auto texels_gpu = gpuHostToDevicePointer(queue, texels);
		auto heap = gpuMalloc<GpuTextureDescriptor>(queue, 1, MEMORY_DESCRIPTOR_HEAP);
		heap[0] = gpuTextureViewDescriptor(queue, texture, {});
		auto heap_gpu = gpuHostToDevicePointer(queue, heap);

		auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
			import noapi;
			[shader("compute")] [numthreads(1, 1, 1)]
			void main() {
				let clamp = GpuSamplerDesc(ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, FILTER_NEAREST, FILTER_NEAREST, FILTER_NEAREST);
				let mirror = GpuSamplerDesc(ADDRESS_MODE_MIRROR_REPEAT, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, FILTER_NEAREST, FILTER_NEAREST, FILTER_NEAREST);
				gpuComputeData<float4>()[0] = gpuSample2D(0, gpuGetSamplerIndex(clamp), float2(1.25, 0.5));
				gpuComputeData<float4>()[1] = gpuSample2D(0, gpuGetSamplerIndex(mirror), float2(1.25, 0.5));
			}
		)slang"));
		auto color = gpuMalloc<float>(queue, 8, MEMORY_READBACK);
		auto color_gpu = gpuHostToDevicePointer(queue, color);

		GpuSamplerDesc samplers[] = {
			{ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, FILTER_NEAREST, FILTER_NEAREST, FILTER_NEAREST},
			{ADDRESS_MODE_MIRROR_REPEAT, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, FILTER_NEAREST, FILTER_NEAREST, FILTER_NEAREST},
		};
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, texels_gpu);
		gpuCopyToTexture(cmd, memory, texels_gpu, texture);
		gpuBarrier(cmd, STAGE_TRANSFER, STAGE_COMPUTE);
		gpuSyncMemoryEXT(cmd, heap_gpu);
		gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
		gpuSetEnabledSamplersEXT(cmd, samplers);
		gpuSetPipeline(cmd, pipeline);
		gpuDispatch(cmd, color_gpu, {1, 1, 1});
		gpuSyncMemoryEXT(cmd, color_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		assert(color[0] == 0.0f && color[1] == 1.0f); // Clamped to the green edge, where repeating would have wrapped round to red
		assert(color[4] == 0.0f && color[5] == 1.0f); // Mirrored back to 0.75, still green

		gpuFreePipeline(queue, pipeline);
		gpuFree(queue, color);
		gpuFree(queue, heap);
		gpuFree(queue, texels);
		gpuFree(queue, memory);
	}
	noapi_doctest_end();
}

// shaders.h

TEST_CASE("gpuSetShaderDiagnosticCallbackEXT receives a failed compile's diagnostics") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		std::string last;
		gpuSetShaderDiagnosticCallbackEXT([](GpuStringView message, void* userdata) { *(std::string*)userdata = std::string_view(message); }, &last);
		assert(!gpuCreateComputePipeline(queue, string_to_bytes("not a shader")));
		assert(!last.empty());
		gpuSetShaderDiagnosticCallbackEXT(nullptr, nullptr);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuAddShaderModuleEXT replaces a registered module and refuses the API's own") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		const char* draft = "module noapi_example_scale; public uint exampleScale() { return 2; }";
		GpuStringView draft_source = {draft, strlen(draft)};
		gpuAddShaderModuleEXT("noapi_example_scale", draft_source);
		const char* module = "module noapi_example_scale; public uint exampleScale() { return 3; }";
		GpuStringView source = {module, strlen(module)};
		gpuAddShaderModuleEXT("noapi_example_scale", source); // Replaces the draft
		gpuAddShaderModuleEXT("noapi", source); // Reported through the diagnostic callback and ignored: noapi is the API's own

		const char* text = "import noapi; import noapi_example_scale;\n"
			"[shader(\"compute\")] [numthreads(1, 1, 1)] void main() { gpuComputeData<uint>()[0] = exampleScale(); }";
		GpuByteSpan shader = {text, strlen(text)};
		GpuPipeline* pipeline = gpuCreateComputePipeline(queue, shader);
		assert(pipeline);

		gpuFreePipeline(queue, pipeline);
		gpuRemoveShaderModuleEXT("noapi_example_scale");
	}
	noapi_doctest_end();
}

TEST_CASE("gpuAddShaderModuleEXT reaches a pipeline whose compiler session was cached before another stage's") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		const char* draft = "module noapi_example_cached; public uint exampleValue() { return 2; }";
		gpuAddShaderModuleEXT("noapi_example_cached", {draft, strlen(draft)});
		auto compute = string_to_bytes(R"slang(
			import noapi; import noapi_example_cached;
			[shader("compute")] [numthreads(1, 1, 1)] void main() { gpuComputeData<uint>()[0] = exampleValue(); }
		)slang");
		auto draft_pipeline = gpuCreateComputePipeline(queue, compute);
		assert(draft_pipeline);

		// A graphics compile, so the compute session is no longer the most recently used one
		auto graphics = string_to_bytes(R"slang(
			import noapi;
			[shader("vertex")] float4 vertexMain(uint index : SV_VertexID) : SV_Position { return float4(0, 0, 0, 1); }
			[shader("fragment")] float4 fragmentMain() : SV_Target { return float4(1, 1, 1, 1); }
		)slang");
		GpuColorTarget target = {.format = FORMAT_RGBA8_UNORM};
		auto graphics_pipeline = gpuCreateGraphicsPipeline(queue, graphics, graphics, {.colorTargets = {&target, 1}});
		assert(graphics_pipeline);

		const char* module = "module noapi_example_cached; public uint exampleValue() { return 3; }";
		gpuAddShaderModuleEXT("noapi_example_cached", {module, strlen(module)});
		auto pipeline = gpuCreateComputePipeline(queue, compute);
		assert(pipeline);

		auto value = gpuMalloc<uint32_t>(queue);
		*value = 0;
		auto value_gpu = gpuHostToDevicePointer(queue, value);
		auto result = gpuMalloc<uint32_t>(queue, 1, MEMORY_READBACK);
		auto result_gpu = gpuHostToDevicePointer(queue, result);
		auto cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, value_gpu);
		gpuSetPipeline(cmd, pipeline);
		gpuDispatch(cmd, value_gpu, {1, 1, 1});
		gpuBarrier(cmd, STAGE_COMPUTE, STAGE_TRANSFER);
		gpuMemCpy(cmd, result_gpu, value_gpu, 4);
		gpuSyncMemoryEXT(cmd, result_gpu);
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
		CHECK_EQ(*result, 3u);

		gpuFree(queue, result);
		gpuFree(queue, value);
		gpuFreePipeline(queue, pipeline);
		gpuFreePipeline(queue, graphics_pipeline);
		gpuFreePipeline(queue, draft_pipeline);
		assert(gpuRemoveShaderModuleEXT("noapi_example_cached"));
	}
	noapi_doctest_end();
}

TEST_CASE("gpuRemoveShaderModuleEXT reports a module that is already gone") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		const char* module = "module noapi_example_empty;";
		gpuAddShaderModuleEXT("noapi_example_empty", {module, strlen(module)});
		assert(gpuRemoveShaderModuleEXT("noapi_example_empty"));
		assert(!gpuRemoveShaderModuleEXT("noapi_example_empty"));
	}
	noapi_doctest_end();
}

TEST_CASE("gpuResetShaderConfigurationEXT drops every registered module") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		const char* module = "module noapi_example_reset;";
		gpuAddShaderModuleEXT("noapi_example_reset", {module, strlen(module)});
		gpuResetShaderConfigurationEXT();
		assert(!gpuRemoveShaderModuleEXT("noapi_example_reset"));
	}
	noapi_doctest_end();
}

// surface.h

TEST_CASE("gpuSurfaceReconfigureEXT resizes mid-frame and refuses depth usage") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuSurface* surface = gpuCreateSurfaceEXT(queue, backend, {.texture = {.dimensions = {640, 480, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
		if(surface) { // NULL without a window, as on a headless device
			auto config = gpuSurfaceGetConfigurationEXT(surface);
			config.texture.dimensions = {1280, 720, 1}; // The window was resized
			gpuSurfaceReconfigureEXT(queue, surface, config);

			assert(gpuSurfaceNextTextureEXT(queue, surface));
			config.texture.dimensions = {800, 600, 1}; // Resized again mid-frame: the acquired texture is given up unrendered
			gpuSurfaceReconfigureEXT(queue, surface, config);

			// No surface renders depth into its textures: Vulkan keeps the configuration it had, WebGPU grants the color part alone
			config.texture.usage = TEXTURE_USAGE_FLAGS(USAGE_COLOR_ATTACHMENT | USAGE_DEPTH_STENCIL_ATTACHMENT);
			gpuSurfaceReconfigureEXT(queue, surface, config);
			assert(!(gpuSurfaceGetConfigurationEXT(surface).texture.usage & USAGE_DEPTH_STENCIL_ATTACHMENT));
			gpuFreeSurfaceEXT(queue, surface);
		}
	}
	noapi_doctest_end();
}

TEST_CASE("gpuSurfaceNextTextureEXT acquires one texture at a time") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuSurface* surface = gpuCreateSurfaceEXT(queue, backend, {.texture = {.dimensions = {640, 480, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
		if(surface) { // NULL without a window, as on a headless device
			if(auto target = gpuSurfaceNextTextureEXT(queue, surface)) {
				assert(!gpuSurfaceNextTextureEXT(queue, surface)); // The first one hasn't been presented yet
				GpuColorAttachment color = {.texture = target, .loadOp = LOAD_OP_CLEAR};
				GpuRenderPassDesc pass = {.colorAttachments = {&color, 1}};
				auto cmd = gpuStartCommandRecording(queue);
				gpuBeginRenderPass(cmd, pass);
				gpuEndRenderPass(cmd, pass);
				gpuSurfacePresentEXT(queue, surface, gpuSubmit(queue, {&cmd, 1}));
			}
			gpuWaitIdleEXT(queue);
			gpuFreeSurfaceEXT(queue, surface);
		}
	}
	noapi_doctest_end();
}

TEST_CASE("gpuSurfacePresentEXT reports presenting with nothing acquired") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuSurface* surface = gpuCreateSurfaceEXT(queue, backend, {.texture = {.dimensions = {640, 480, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
		if(surface) { // NULL without a window, as on a headless device
			int unsupported = 0;
			gpuSetDiagnosticCallbackEXT(queue, [](GpuQueue*, GPU_DIAGNOSTIC, GpuStringView, void* count) { ++*(int*)count; }, &unsupported);
			gpuSurfacePresentEXT(queue, surface);
			assert(unsupported == 1);
			if(gpuSurfaceNextTextureEXT(queue, surface)) gpuSurfacePresentEXT(queue, surface); // Handed back unrendered, with nothing to wait for
			assert(unsupported == 1);
			gpuSetDiagnosticCallbackEXT(queue, nullptr, nullptr);
			gpuFreeSurfaceEXT(queue, surface);
		}
	}
	noapi_doctest_end();
}

// sync.h

TEST_CASE("gpuSyncMemoryEXT syncs both ways and ignores GPU only memory") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		uint32_t* values = (uint32_t*)gpuMalloc(queue, 4 * sizeof(uint32_t), 16, MEMORY_DEFAULT);
		uint32_t* copy = (uint32_t*)gpuMalloc(queue, 4 * sizeof(uint32_t), 16, MEMORY_READBACK);
		values[2] = 42;
		gpu* values_gpu = gpuHostToDevicePointer(queue, values);
		gpu* copy_gpu = gpuHostToDevicePointer(queue, copy);

		gpu* scratch = (gpu*)gpuMalloc(queue, 16, 16, MEMORY_GPU);

		GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
		gpuSyncMemoryEXT(cmd, scratch); // GPU only, so there is nothing to sync
		gpuSyncMemoryEXT(cmd, values_gpu); // The CPU wrote it, so before the copy reads it
		gpuMemCpy(cmd, copy_gpu, values_gpu, 4 * sizeof(uint32_t));
		gpuSyncMemoryEXT(cmd, copy_gpu); // The GPU wrote it, so after
		GpuCommandBufferSpan buffers = {&cmd, 1};
		gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, buffers, NULL, 0), UINT64_MAX);
		assert(copy[2] == 42);

		gpuFree(queue, values);
		gpuFree(queue, copy);
		gpuFreeDevicePointerEXT(queue, scratch);
	}
	noapi_doctest_end();
}

// vulkan/noapi.h
#ifdef NOAPI_BACKEND_VULKAN

TEST_CASE("gpuSetupDefaultVulkanEXT says why it failed, cut to fit, and leaves nothing behind") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuCStringSpan none = {NULL, 0};
		const char* missing[] = {"VK_NOAPI_example_missing"};
		GpuCStringSpan missing_names = {missing, 1};
		GpuVulkanDefault vulkan;
		char error[8];
		assert(!gpuSetupDefaultVulkanEXT(NULL, NULL, gpuDefaultErrorCallbackEXT, VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
			none, missing_names, none, false, &vulkan, error, sizeof(error))); // No such layer
		assert(strlen(error) == sizeof(error) - 1);
		assert(!gpuSetupDefaultVulkanEXT(NULL, NULL, gpuDefaultErrorCallbackEXT, VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
			none, none, missing_names, false, &vulkan, NULL, 0)); // No device with such an extension
	}
	noapi_doctest_end();
}

TEST_CASE("gpuSetupDefaultVulkanEXT reports validation messages at or above the severity filter") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		static int messages = 0;
		auto count = [](void*, int, GpuStringView) { ++messages; };
		auto invalid_buffer = [](const GpuVulkanDefault& vulkan) {
			auto create = (PFN_vkCreateBuffer)vkGetDeviceProcAddr(vulkan.device, "vkCreateBuffer");
			auto destroy = (PFN_vkDestroyBuffer)vkGetDeviceProcAddr(vulkan.device, "vkDestroyBuffer");
			VkBufferCreateInfo info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 0, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}; // A size of 0 is invalid
			VkBuffer buffer = VK_NULL_HANDLE;
			if(create(vulkan.device, &info, nullptr, &buffer) == VK_SUCCESS) destroy(vulkan.device, buffer, nullptr);
			((PFN_vkDestroyDevice)vkGetDeviceProcAddr(vulkan.device, "vkDestroyDevice"))(vulkan.device, nullptr);
			vkDestroyDebugUtilsMessengerEXT(vulkan.instance, vulkan.messenger, nullptr);
			vkDestroyInstance(vulkan.instance, nullptr);
		};

		GpuCStringSpan none = {};
		GpuVulkanDefault vulkan;
		assert(gpuSetupDefaultVulkanEXT(nullptr, nullptr, count, VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT, none, none, none, true, &vulkan, nullptr, 0));
		invalid_buffer(vulkan);
		assert(messages == 0); // Nothing is severe enough
		assert(gpuSetupDefaultVulkanEXT(nullptr, nullptr, count, VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, none, none, none, true, &vulkan, nullptr, 0));
		invalid_buffer(vulkan);
		assert(messages > 0);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuRequiredVulkanDeviceExtensionsEXT names the descriptor heap extension") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		auto required = gpuRequiredVulkanDeviceExtensionsEXT();
		assert(std::ranges::any_of(std::span(required.ptr, required.count), [](const char* name) { return strcmp(name, "VK_EXT_descriptor_heap") == 0; }));
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCreateQueue looks up the device's queue, and a compute only one where there is a family for it") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuQueue* second = gpuCreateQueue(backend.instance, backend.gpu, backend.device, backend.graphics_queue, backend.graphics_queue_family, true, gpuDefaultCpuAllocator, NULL);
		assert(second);
		gpuFreeQueue(second);

		GpuQueue* found = gpuCreateQueue(backend.instance, backend.gpu, backend.device, VK_NULL_HANDLE, (uint32_t)-1, true, gpuDefaultCpuAllocator, NULL);
		assert(found); // The device's graphics queue, looked up
		gpuFreeQueue(found);

		GpuQueue* compute = gpuCreateQueue(backend.instance, backend.gpu, backend.device, VK_NULL_HANDLE, (uint32_t)-1, false, gpuDefaultCpuAllocator, NULL);
		assert(compute || errno == VK_ERROR_FEATURE_NOT_PRESENT); // Only where the device has a compute family of its own
		if(compute) gpuFreeQueue(compute);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCreateQueue works on a device of the program's own alongside the shared one") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		float priority = 1;
		VkDeviceQueueCreateInfo queue_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = backend.graphics_queue_family, .queueCount = 1, .pQueuePriorities = &priority};
		auto features14 = gpuEnableRequiredVulkan14FeaturesEXT({});
		features14.pNext = gpuRequiredVulkanDeviceCreateInfoPnextEXT(); // Linked in front of, never after
		auto features13 = gpuEnableRequiredVulkan13FeaturesEXT({});
		features13.pNext = &features14;
		auto features12 = gpuEnableRequiredVulkan12FeaturesEXT({});
		features12.pNext = &features13;
		auto features11 = gpuEnableRequiredVulkan11FeaturesEXT({});
		features11.pNext = &features12;
		VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features11, .features = gpuEnableRequiredVulkanFeaturesEXT({})};
		auto extensions = gpuRequiredVulkanDeviceExtensionsEXT();
		VkDeviceCreateInfo info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &features, .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
			.enabledExtensionCount = uint32_t(extensions.count), .ppEnabledExtensionNames = extensions.ptr};
		VkDevice device;
		assert(vkCreateDevice(backend.gpu, &info, nullptr, &device) == VK_SUCCESS);
		VkQueue device_queue;
		vkGetDeviceQueue(device, backend.graphics_queue_family, 0, &device_queue);
		GpuQueue* own = gpuCreateQueue(backend.instance, backend.gpu, device, device_queue, backend.graphics_queue_family);
		assert(own && !gpuGetCapabilitiesEXT(own).mesh_shaders); // VK_EXT_mesh_shader wasn't enabled on it

		// The same work on each device, one after the other
		for(GpuQueue* each: {queue, own}) {
			auto pipeline = gpuCreateComputePipeline(each, string_to_bytes(R"slang(
				import noapi;
				[shader("compute")] [numthreads(1, 1, 1)] void main() { gpuComputeData<uint>()[0] = 7; }
			)slang"));
			auto value = gpuMalloc<uint32_t>(each, 1, MEMORY_READBACK);
			auto value_gpu = gpuHostToDevicePointer(each, value);
			auto cmd = gpuStartCommandRecording(each);
			gpuSetPipeline(cmd, pipeline);
			gpuDispatch(cmd, value_gpu, {1, 1, 1});
			gpuSyncMemoryEXT(cmd, value_gpu);
			gpuWaitSemaphore(each, gpuGetSubmissionSemaphoreEXT(each), gpuSubmit(each, {&cmd, 1}));
			assert(*value == 7);
			gpuFreePipeline(each, pipeline);
			gpuFree(each, value);
		}

		gpuFreeQueue(own);
		vkDestroyDevice(device, nullptr);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCreateSurfaceEXT reports a missing VkSurfaceKHR as lost") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuSurfaceDescriptor desc = GPU_SURFACE_DESCRIPTOR_DEFAULT;
		desc.texture.usage = USAGE_COLOR_ATTACHMENT;
		assert(!gpuCreateSurfaceEXT(queue, VK_NULL_HANDLE, desc) && errno == VK_ERROR_SURFACE_LOST_KHR);
	}
	noapi_doctest_end();
}

#endif

// webgpu/noapi.h
#ifdef NOAPI_BACKEND_WEBGPU

TEST_CASE("gpuSetupDefaultWebGPUEXT reports device errors to the error callback") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		static int errors = 0;
		GpuWebGPUDefault webgpu;
		assert(gpuSetupDefaultWebGPUEXT(nullptr, nullptr, [](void*, int, GpuStringView) { ++errors; }, true, &webgpu, nullptr, 0));
		WGPUBufferDescriptor invalid = {.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_Storage, .size = 16}; // Mapped for reading, a buffer can only be copied into
		wgpuBufferRelease(wgpuDeviceCreateBuffer(webgpu.device, &invalid));
		wgpuInstanceProcessEvents(webgpu.instance);
		assert(errors == 1);

		wgpuDeviceRelease(webgpu.device);
		wgpuAdapterRelease(webgpu.adapter);
		wgpuInstanceRelease(webgpu.instance);
	}
	noapi_doctest_end();
}

TEST_CASE("gpuCreateSurfaceEXT reports a missing WGPUSurface as a validation error") {
	GpuQueue* queue = noapi_doctest_queue();
	GpuDoctestBackend backend = noapi_doctest_backend();
	(void)queue;
	(void)backend;
	noapi_doctest_begin();
	{
		GpuSurfaceDescriptor desc = GPU_SURFACE_DESCRIPTOR_DEFAULT;
		desc.texture.usage = USAGE_COLOR_ATTACHMENT;
		assert(!gpuCreateSurfaceEXT(queue, nullptr, desc) && errno == WGPUErrorType_Validation);
	}
	noapi_doctest_end();
}

#endif
