#pragma once

#include "../surface.hpp"
#include "../sync.hpp"
#include "../samplers.hpp"
#include "../allocator.hpp"

#include <expected>
#include <functional>
#include <string>
#include <unordered_map>
#include <webgpu/webgpu.h>

namespace GPU {
#ifdef __cpp_lib_function_ref
	template<typename T>
	using function_t = std::function_ref<T>;
#else 
	template<typename T>
	using function_t = std::function<T>;
#endif

	namespace default_ {
		void error_callback(void* queue, int type, std::string_view message);
	}
}

struct GpuWebGPUDefault {
	WGPUInstance instance;
	WGPUAdapter adapter;
	WGPULimits limits;
	WGPUDevice device;
	WGPUSurface surface;
};
std::expected<GpuWebGPUDefault, std::string> gpuSetupDefaultWebGPUEXT(GPU::function_t<WGPUSurface(WGPUInstance)> surface_loader, void(*error_callback)(void* queue, int type, std::string_view message) = GPU::default_::error_callback, bool prefer_high_power = true);

gpu* gpuEncodeWebGPUAddressEXT(uint8_t monobuffer, uint64_t address);
std::pair<uint8_t, uint64_t> gpuDecodeWebGPUAddressEXT(gpu* addr);

struct GpuQueue {
	WGPUAdapter adapter;
	WGPUDevice device;
	WGPULimits limits;
	WGPUQueue queue;
	CpuAllocatorFunc cpu_allocator;

	size_t current_submission_index = 0;

	size_t monobuffer_size = 0, monobuffer_capacity = 0;
	WGPUBuffer empty_monobuffer = nullptr;
	std::array<WGPUBuffer, 6> monobuffers;
	uint8_t active_monobuffer = 0;

	struct MonobufferRange {
		uint8_t buffer;
		size_t start, end;
		size_t size() { return end - start; }
	};
	std::unordered_map<gpu*, std::pair<MonobufferRange, void*>> allocations;
	std::unordered_map<gpu*, void*> gpu2cpu;
	std::unordered_map<void*, gpu*> cpu2gpu;
	std::vector<MonobufferRange> freelist;

	std::vector<std::pair<WGPUBuffer, size_t>> buffers_pending_free;
};

GpuQueue* gpuCreateQueue(WGPUAdapter adapter, WGPUDevice device, WGPULimits limits, CpuAllocatorFunc allocator = default_::cpu_allocator);
inline GpuQueue* gpuCreateQueue(GpuWebGPUDefault def, CpuAllocatorFunc allocator = default_::cpu_allocator) {
	return gpuCreateQueue(def.adapter, def.device, def.limits, allocator);
}

struct GpuCommandBuffer {
	GpuQueue* queue;
	WGPUCommandEncoder encoder;
	// WGPURenderPassEncoder render_pass;
};