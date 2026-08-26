
#include "noapi.hpp"
#include "common.hpp"

#include <cassert>
#include <cstring>
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

	out->current_submission_timeline_semaphore = GPU::semaphore_initialize(out, out->next_submission_index - 1);

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
	GPU::semaphore_destroy(queue->current_submission_timeline_semaphore);

	auto allocator = queue->cpu_allocator;
	queue->~GpuQueue();
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
	constexpr static auto allocation_bookkeeping = [](GpuQueue* queue, uint8_t active_monobuffer, size_t gpu_address, size_t bytes, MEMORY memory) {
		auto cpu = queue->cpu_allocator(nullptr, bytes);\
		auto gpu = gpuEncodeWebGPUAddressEXT(active_monobuffer, gpu_address);
		queue->allocations[gpu] = {GpuQueue::MonobufferRange{active_monobuffer, gpu_address, gpu_address + bytes}, cpu, memory};
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
			return allocation_bookkeeping(queue, active_buffer, aligned, bytes, memory);
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
			queue->code_pending_submission_finished.emplace_back([buffer = queue->monobuffers[queue->active_monobuffer]]() {
				wgpuBufferRelease(buffer);
			}, queue->next_submission_index);
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
			wgpuCommandEncoderRelease(cmd);
			wgpuCommandBufferRelease(to_submit);
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
	return allocation_bookkeeping(queue, new_active_monobuffer, new_start, new_end - new_start, memory);
}

gpu* gpuHostToDevicePointer(GpuQueue* queue, void* ptr) {
	if(queue->cpu2gpu.contains(ptr))
		return queue->cpu2gpu[ptr];
	return nullptr;
}

void* gpuDeviceToHostPointerEXT(GpuQueue* queue, gpu* ptr) {
	if(queue->allocations.contains(ptr))
		return std::get<void*>(queue->allocations[ptr]);
	return nullptr;
}

void gpuFree(GpuQueue* queue, void* ptr) {
	auto device = gpuHostToDevicePointer(queue, ptr);
	if(device) gpuFree(queue, device);
}
void gpuFree(GpuQueue* queue, gpu* ptr) {
	if(!queue->allocations.contains(ptr)) return;

	// TODO: Extra buffers go here

	auto [range, cpu, _memory_type] = queue->allocations[ptr];
	queue->cpu_allocator(cpu, 0);

	queue->allocations.erase(ptr);
	queue->cpu2gpu.erase(cpu);

	queue->freelist.push_back(range);
	// Sort the freelist so buffers are together, and then by start
	std::sort(queue->freelist.begin(), queue->freelist.end(), [](const GpuQueue::MonobufferRange& a, const GpuQueue::MonobufferRange& b) -> bool {
		if(a.buffer == b.buffer)
			return a.start < b.start;

		return a.buffer < b.buffer;
	});
}

GpuCommandBuffer* gpuStartCommandRecording(GpuQueue* queue) {
	auto out = (GpuCommandBuffer*)queue->cpu_allocator(nullptr, sizeof(GpuCommandBuffer));
	new(out) GpuCommandBuffer{
		.queue = queue,
		.encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr)
	};

	auto current_finished_submission = GPU::semaphore_value(queue, queue->current_submission_timeline_semaphore);
	if(queue->code_pending_submission_finished.size())
		for(size_t i = queue->code_pending_submission_finished.size(); i--; ) {
			auto& [code, submit] = queue->code_pending_submission_finished[i];
			if(submit <= current_finished_submission) {
				code();
				queue->code_pending_submission_finished.erase(queue->code_pending_submission_finished.begin() + i);
			}
		}

	return out;
}

void gpuFreeCommandBuffer(GpuCommandBuffer* cmd) {
	wgpuCommandEncoderRelease(cmd->encoder);

	auto allocator = cmd->queue->cpu_allocator;
	cmd->~GpuCommandBuffer();
	allocator(cmd, 0);
}

uint64_t gpuSubmitNoFree(GpuQueue* queue, std::span<GpuCommandBuffer*> command_buffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signal_value /* = 0 */) {
	std::vector<WGPUCommandBuffer> buffers; buffers.reserve(command_buffers.size() + 1);
	for(auto buffer: command_buffers)
		buffers.emplace_back(wgpuCommandEncoderFinish(buffer->encoder, nullptr));

	GpuCommandBuffer cmd {
		.queue = queue,
		.encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr),
	};
	GPU::semaphore_gpu_increment(&cmd, queue->current_submission_timeline_semaphore);

	// TODO: GpuSemaphore* semaphore /* = nullptr */ stuff

	buffers.emplace_back(wgpuCommandEncoderFinish(cmd.encoder, nullptr));
	wgpuCommandEncoderRelease(cmd.encoder);

	wgpuQueueSubmit(queue->queue, buffers.size(), buffers.data());

	for(auto buffer: command_buffers)
		for(auto code: buffer->code_pending_submission_finished)
			queue->code_pending_submission_finished.emplace_back(code, queue->next_submission_index);

	return queue->next_submission_index++;
}

uint64_t gpuSubmit(GpuQueue* queue, std::span<GpuCommandBuffer*> command_buffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signal_value /* = 0 */) {
	auto submission = gpuSubmitNoFree(queue, command_buffers, semaphore, signal_value);
	for(auto cmd: command_buffers)
		gpuFreeCommandBuffer(cmd);
	return submission;
}

void gpuWaitIdleEXT(GpuQueue* queue);

void gpuSyncMemoryEXT(GpuCommandBuffer* cmd, gpu* mem) {
	auto [range, cpu, memory_type] = cmd->queue->allocations[mem];
	auto size = range.size();

	switch (memory_type) {
	case MEMORY_DEFAULT:
	case MEMORY_GPU:
	case MEMORY_TEXTURE: {
		WGPUBufferDescriptor d{
			.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite,
			.size = size,
			.mappedAtCreation = true,
		};
		auto tmp = wgpuDeviceCreateBuffer(cmd->queue->device, &d);
		auto tmp_cpu = wgpuBufferGetMappedRange(tmp, 0, size);
		memcpy(tmp_cpu, cpu, size);
		wgpuBufferUnmap(tmp);

		wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, tmp, 0, cmd->queue->monobuffers[range.buffer], range.start, size);
		cmd->code_pending_submission_finished.emplace_back([tmp]() {
			wgpuBufferRelease(tmp);
		});
	}
	break; case MEMORY_READBACK:
	case MEMORY_TEXTURE_READBACK: {
		WGPUBufferDescriptor d{
			.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
			.size = size,
		};
		auto tmp = wgpuDeviceCreateBuffer(cmd->queue->device, &d);

		wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, cmd->queue->monobuffers[range.buffer], range.start, tmp, 0, size);
		cmd->code_pending_submission_finished.emplace_back([queue = cmd->queue, tmp, cpu, size]() {
			GPU::wait_for_buffer_map(queue, tmp, WGPUMapMode_Read, 0, size);

			auto tmp_cpu = wgpuBufferGetConstMappedRange(tmp, 0, size);
			memcpy(cpu, tmp_cpu, size);

			wgpuBufferRelease(tmp);
		});
	}
	}
}

void gpuSyncMemoryEXT(GpuQueue* queue, gpu* mem) {
	auto cmd = gpuStartCommandRecording(queue);
	gpuSyncMemoryEXT(cmd, mem);
	auto submit_index = gpuSubmit(queue, {&cmd, 1});

	GPU::semaphore_wait(queue, queue->current_submission_timeline_semaphore, submit_index);

	// TODO: Should we add an accessor function that lets us poll the pending syncs without having to use this hack?
	gpuFreeCommandBuffer(gpuStartCommandRecording(queue));
}