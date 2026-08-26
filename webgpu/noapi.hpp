#pragma once

#include "../surface.hpp"
#include "../sync.hpp"
#include "../samplers.hpp"
#include "../allocator.hpp"

#include <expected>
#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <webgpu/webgpu.h>
#include <array>

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
std::pair<uint8_t, uint64_t> gpuDecodeWebGPUAddressEXT(gpu* addr);\

struct GpuSemaphore {
	WGPUBuffer buffer;			// 8 bytes, Storage | CopySrc | CopyDst
	WGPUBuffer readback_buffer;	// 8 bytes, MapRead | CopyDst
	WGPUBuffer upload_buffer;	// 8 bytes, Uniform | CopyDst (input to cs_set)
	WGPUBindGroup bind_group;	// binds {buffer, valueUniformBuffer}
};

struct GpuQueue {
	WGPUAdapter adapter;
	WGPUDevice device;
	WGPULimits limits;
	WGPUQueue queue;
	CpuAllocatorFunc cpu_allocator;

	WGPUBindGroupLayout semaphore_bind_group_layout = nullptr;
	WGPUComputePipeline semaphore_increment_pipeline = nullptr;
	WGPUComputePipeline semaphore_set_pipeline = nullptr;
	WGPUComputePipeline semaphore_set_max_pipeline = nullptr;

	size_t next_submission_index = 1;
	GpuSemaphore current_submission_timeline_semaphore;

	size_t monobuffer_size = 0, monobuffer_capacity = 0;
	WGPUBuffer empty_monobuffer = nullptr;
	std::array<WGPUBuffer, 6> monobuffers;
	uint8_t active_monobuffer = 0;

	struct MonobufferRange {
		uint8_t buffer;
		size_t start, end;
		size_t size() { return end - start; }
	};
	std::unordered_map<gpu*, std::tuple<MonobufferRange, void*, MEMORY>> allocations;
	std::unordered_map<void*, gpu*> cpu2gpu;
	std::vector<MonobufferRange> freelist;

	std::vector<std::pair<std::function<void()>, size_t>> code_pending_submission_finished;
};

GpuQueue* gpuCreateQueue(WGPUAdapter adapter, WGPUDevice device, WGPULimits limits, CpuAllocatorFunc allocator = default_::cpu_allocator);
inline GpuQueue* gpuCreateQueue(GpuWebGPUDefault def, CpuAllocatorFunc allocator = default_::cpu_allocator) {
	return gpuCreateQueue(def.adapter, def.device, def.limits, allocator);
}

struct GpuCommandBuffer {
	GpuQueue* queue;
	WGPUCommandEncoder encoder;
	// WGPURenderPassEncoder render_pass;

	std::vector<std::function<void()>> code_pending_submission_finished;
};