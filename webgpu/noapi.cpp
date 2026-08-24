
#include "noapi.hpp"
#include "compute.hpp"
#include "webgpu/webgpu.h"

#include <cassert>
#include <iostream>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

void GPU::default_::error_callback(void* queue, int type, std::string_view message) {
	std::cerr << "WGPU Device Error " << type << ": " << message << std::endl;
#ifdef _WIN32
	system("pause");
#endif
	exit(-1);
}

std::expected<GpuWebGPUDefault, std::string> gpuSetupDefaultWebGPUEXT(GPU::function_t<WGPUSurface(WGPUInstance)> surface_loader, void(*error_callback)(void* queue, int type, std::string_view message) /* = GPU::default_::error_callback */, bool prefer_high_power /* = true */){
	GpuWebGPUDefault out;

	{ // Instance
		WGPUInstanceDescriptor d {
		
		};
		out.instance = wgpuCreateInstance(&d);
	}
	if(!out.instance) return std::unexpected("Failld to create WebGPU instance");

	out.surface = surface_loader(out.instance);

	{ // Adapter
		WGPURequestAdapterOptions adapter_opts {
			.powerPreference = prefer_high_power ? WGPUPowerPreference_HighPerformance : WGPUPowerPreference_LowPower,
			.compatibleSurface = out.surface,
		};

		struct RequestAdapterResult {
			WGPUAdapter adapter;
			bool done;
		} result = {};
		wgpuInstanceRequestAdapter(out.instance, &adapter_opts, {
			.mode = WGPUCallbackMode_AllowSpontaneous,
			.callback = +[](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void* userdata2) {
				RequestAdapterResult* result = (RequestAdapterResult*)userdata1;
				auto error_callback = *(void(**)(void* queue, int type, std::string_view message))userdata2;
				result->done = true;
				if (status != WGPURequestAdapterStatus_Success) 
					error_callback(nullptr, WGPUErrorType_Validation, {message.data, message.length});
				result->adapter = adapter;
			},
			.userdata1 = &result,
			.userdata2 = &error_callback,
		});
		while (!result.done) {
			wgpuInstanceProcessEvents(out.instance);
		#ifdef __EMSCRIPTEN__
			emscripten_sleep(1); // yields back to the browser event loop
		#endif
		}
		out.adapter = result.adapter;
	}
	if(!out.adapter) return std::unexpected("Failed to create WebGPU adapter");

	out.limits = {}; // Emscripten apparently asserts that the limits have been zeroed out!
	wgpuAdapterGetLimits(out.adapter, &out.limits);

	{ // Device
		WGPUDeviceDescriptor device_desc = {
			.requiredLimits = &out.limits,
			.deviceLostCallbackInfo = {
				.mode = WGPUCallbackMode_AllowSpontaneous,
				.callback = +[](WGPUDevice const* device, WGPUDeviceLostReason reason, WGPUStringView message, void* userdata1, void* _) {
					if(reason == WGPUDeviceLostReason_Destroyed) return;

					auto callback = (void(*)(WGPUDevice const* device, WGPUErrorType type, std::string_view message))userdata1;
					callback(device, (WGPUErrorType)reason, {message.data, message.length});
				},
				.userdata1 = (void*)error_callback
			},
			.uncapturedErrorCallbackInfo = {
				.callback = +[](WGPUDevice const* device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* _){
					auto callback = (void(*)(WGPUDevice const* device, WGPUErrorType type, std::string_view message))userdata1;
					callback(device, type, {message.data, message.length});
				},
				.userdata1 = (void*)error_callback
			}
		};

		struct RequestDeviceResult {
			WGPUDevice device;
			bool done;
		} result = {};
		wgpuAdapterRequestDevice(out.adapter, &device_desc, {
			.mode = WGPUCallbackMode_AllowSpontaneous,
			.callback = +[](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void* userdata2) {
				RequestDeviceResult* result = (RequestDeviceResult*)userdata1;
				auto& error_callback = *(GPU::function_t<void(WGPUDevice const* device, WGPUErrorType type, std::string_view message)>*)userdata2;
				result->done = true;
				if (status != WGPURequestDeviceStatus_Success)
					error_callback(nullptr, WGPUErrorType_Validation, {message.data, message.length});
				result->device = device;
			},
			.userdata1 = &result,
			.userdata2 = &error_callback,
		});
		while (!result.done) {
			wgpuInstanceProcessEvents(out.instance);
		#ifdef __EMSCRIPTEN__
			emscripten_sleep(1); // yields back to the browser event loop
		#endif
		}
		out.device = result.device;
	}
	if(!out.device) return std::unexpected("Failed to create WebGPU device");

	return out;
}

GpuQueue* gpuCreateQueue(WGPUAdapter adapter, WGPUDevice device, WGPULimits limits, CpuAllocatorFunc allocator /* = default_::gpu_allocator */) {
	auto out = (GpuQueue*)allocator(nullptr, sizeof(GpuQueue));
	new(out) GpuQueue{
		.adapter = adapter,
		.device = device,
		.limits = limits,
		.cpu_allocator = allocator,
	};

	out->queue = wgpuDeviceGetQueue(device);

	WGPUBufferDescriptor d {
		.label = {"NoAPI Empty Monobuffer", WGPU_STRLEN},
		.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
		.size = 1
	};
	out->empty_monobuffer = wgpuDeviceCreateBuffer(out->device, &d);
	std::fill(out->monobuffers.begin(), out->monobuffers.end(), out->empty_monobuffer);
	return out;
}

void gpuFreeQueue(GpuQueue* queue) {
	// TODO: Free stuff!

	auto allocator = queue->cpu_allocator;
	allocator(queue, 0);
}

constexpr static uint64_t gpu_address_max = 0x1FFFFFFFFFFFFFFF; // (2^61 - 1) aka max number storable in 60 bits

// top 3 bits encode monobuffer, rest encodes address
gpu* gpuEncodeWebGPUAddressEXT(uint8_t monobuffer, uint64_t address) {
	assert(monobuffer < GpuQueue{}.monobuffers.size());
	assert(address <= gpu_address_max);

	uint64_t debug = uint64_t(monobuffer + 1) << 61 | address;
	return (gpu*)debug;
}
std::pair<uint8_t, uint64_t> gpuDecodeWebGPUAddressEXT(gpu* addr) {
	auto address = (uint64_t)addr;
	return {(address >> 61) - 1, address & gpu_address_max};
}

void* gpuMalloc(GpuQueue* queue, size_t bytes, size_t align /* = 16 */, MEMORY memory /* = MEMORY_DEFAULT */) {
	constexpr static auto align_up = [](size_t addr, size_t align) {
		return (addr + align - 1) & ~(align - 1);
	};
	constexpr static auto allocation_bookkeeping = [](GpuQueue* queue, uint8_t active_monobuffer, size_t gpu_address, size_t bytes) {
		auto cpu = queue->cpu_allocator(nullptr, bytes);\
		auto gpu = gpuEncodeWebGPUAddressEXT(active_monobuffer, gpu_address);
		queue->allocations[gpu] = {GpuQueue::MonobufferRange{active_monobuffer, gpu_address, gpu_address + bytes}, cpu};
		queue->gpu2cpu[gpu] = cpu;
		queue->cpu2gpu[cpu] = gpu;
		return cpu;
	};

	if(bytes > queue->limits.maxStorageBufferBindingSize) {
		errno = WGPUErrorType_OutOfMemory;
		return nullptr;
	}

	for(size_t i = 0; i < queue->freelist.size(); ++i) {
		auto& [active_buffer, start, end] = queue->freelist[i];
		// Strip any empty ranges from the list
		if(start == end) {
			queue->freelist.erase(queue->freelist.begin() + i);
			--i;
			continue;
		}

		// If we find a free space big enough for the allocation... use that
		auto aligned = align_up(start, align);
		if(end - aligned > bytes) {
			start = aligned + bytes;
			return allocation_bookkeeping(queue, active_buffer, aligned, bytes);
		}

		// If each element overlaps with the previous one merge them
		// assumes that the freelist is sorted by buffer index followed by starting value
		if(i == 0) continue;

		auto [prev_buffer, prev_start, prev_end] = queue->freelist[i - 1];
		if(start > prev_start && start < prev_end && prev_buffer == active_buffer) {
			queue->freelist.erase(queue->freelist.begin() + (i - 1));
			start = prev_start;
		}
	}

	auto new_start = align_up(queue->monobuffer_size, align);
	auto new_end = new_start + bytes;
	auto new_active_monobuffer = queue->active_monobuffer;

	// Reallocate the monobuffer if necessary
	bool activate_next_monobuffer = false;
	if(new_end > queue->monobuffer_capacity) {
		if(queue->monobuffers[queue->active_monobuffer] != queue->empty_monobuffer)
			queue->buffers_pending_free.emplace_back(queue->monobuffers[queue->active_monobuffer], queue->current_submission_index);
		else queue->monobuffer_capacity = new_end;

		queue->monobuffer_capacity = std::max(queue->monobuffer_capacity * 2, new_end); // Doubles the size of the capacity each time
		if(queue->monobuffer_capacity > queue->limits.maxStorageBufferBindingSize) {
			queue->monobuffer_capacity = queue->limits.maxStorageBufferBindingSize;
			activate_next_monobuffer = true;
		}
		if(new_end > queue->limits.maxStorageBufferBindingSize) {
			new_start = 0;
			new_end = bytes;
			++new_active_monobuffer;
		}

		WGPUBufferDescriptor d {
			.label = {"NoAPI Monobuffer", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
			.size = queue->monobuffer_capacity 
		};
		auto old_monobuffer = queue->monobuffers[queue->active_monobuffer];
		queue->monobuffers[queue->active_monobuffer] = wgpuDeviceCreateBuffer(queue->device, &d);

		if(old_monobuffer != queue->empty_monobuffer) {
			auto cmd = wgpuDeviceCreateCommandEncoder(queue->device, nullptr);
			wgpuCommandEncoderCopyBufferToBuffer(cmd, old_monobuffer, 0, queue->monobuffers[queue->active_monobuffer], 0, queue->monobuffer_size);
			auto to_submit = wgpuCommandEncoderFinish(cmd, nullptr);
			wgpuQueueSubmit(queue->queue, 1, &to_submit);
			// TODO: need to add to_submit to the free queue
			wgpuCommandEncoderRelease(cmd);
		}

		if(activate_next_monobuffer) {
			// Add the rest of the old monobuffer to the freelist
			queue->freelist.emplace_back(queue->active_monobuffer, queue->monobuffer_size, queue->limits.maxStorageBufferBindingSize);

			++queue->active_monobuffer;
			if(queue->active_monobuffer >= queue->monobuffers.size()) { // Fail if monobuffer being activated is the 8th monobuffer
				errno = WGPUErrorType_OutOfMemory;
				return nullptr;
			}

			WGPUBufferDescriptor d {
				.label = {"NoAPI Monobuffer", WGPU_STRLEN},
				.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
				.size = bytes 
			};
			queue->monobuffers[queue->active_monobuffer] = wgpuDeviceCreateBuffer(queue->device, &d);

			if(new_start == 0)
				queue->monobuffer_size = 0;
			else queue->monobuffer_size = bytes;
			queue->monobuffer_capacity = bytes;
		}
	}

	if(!activate_next_monobuffer) queue->monobuffer_size = new_end;
	return allocation_bookkeeping(queue, new_active_monobuffer, new_start, new_end - new_start);
}

gpu* gpuHostToDevicePointer(GpuQueue* queue, void* ptr) {
	if(queue->cpu2gpu.contains(ptr))
		return queue->cpu2gpu[ptr];
	return nullptr;
}