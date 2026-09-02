#pragma once

#include "webgpu/noapi.hpp"

#include <cstdint>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <chrono>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

namespace GPU {

	static const char* semaphore_wgsl_code = R"WGSL(
struct SemaphoreData {
	lo: atomic<u32>,
	hi: atomic<u32>,
}

struct SetValue {
	lo: u32,
	hi: u32,
}

@group(0) @binding(0) var<storage, read_write> semaphore_buf: SemaphoreData;
@group(0) @binding(1) var<uniform> set_value: SetValue;

@compute @workgroup_size(1)
fn cs_increment() {
	loop {
		let old_lo = atomicLoad(&semaphore_buf.lo);
		let new_lo = old_lo + 1u; // wraps 0xFFFFFFFF -> 0 on overflow
		let r = atomicCompareExchangeWeak(&semaphore_buf.lo, old_lo, new_lo);
		if (r.exchanged) {
			if (new_lo == 0u) {
				atomicAdd(&semaphore_buf.hi, 1u);
			}
			break;
		}
		// else: another invocation raced us on `lo`, retry.
	}
}

@compute @workgroup_size(1)
fn cs_set() {
	atomicStore(&semaphore_buf.lo, set_value.lo);
	atomicStore(&semaphore_buf.hi, set_value.hi);
}

@compute @workgroup_size(1)
fn cs_set_max() {
	let cur_lo = atomicLoad(&semaphore_buf.lo);
	let cur_hi = atomicLoad(&semaphore_buf.hi);
	let new_lo = set_value.lo;
	let new_hi = set_value.hi;

	let is_greater = (new_hi > cur_hi) || (new_hi == cur_hi && new_lo > cur_lo);
	if (is_greater) {
		atomicStore(&semaphore_buf.lo, new_lo);
		atomicStore(&semaphore_buf.hi, new_hi);
	}
})WGSL";

	inline void ensure_semaphore_pipelines(GpuQueue* queue) {
		if(queue->semaphore_bind_group_layout) return;

		std::array<WGPUBindGroupLayoutEntry, 2> entries = {
			WGPUBindGroupLayoutEntry{
				.binding = 0,
				.visibility = WGPUShaderStage_Compute,
				.buffer = {
					.type = WGPUBufferBindingType_Storage,
					.minBindingSize = sizeof(uint64_t),
				}
			}, WGPUBindGroupLayoutEntry{
				.binding = 1,
				.visibility = WGPUShaderStage_Compute,
				.buffer = {
					.type = WGPUBufferBindingType_Uniform,
					.minBindingSize = sizeof(uint64_t),
				}
			}
		};

		WGPUBindGroupLayoutDescriptor bind_group{
			.entryCount = entries.size(),
			.entries = entries.data(),
		};
		queue->semaphore_bind_group_layout = wgpuDeviceCreateBindGroupLayout(queue->device, &bind_group);

		WGPUPipelineLayoutDescriptor pipeline_layout_desc{
			.bindGroupLayoutCount = 1,
			.bindGroupLayouts = &queue->semaphore_bind_group_layout,
		};
		WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(queue->device, &pipeline_layout_desc);

		WGPUShaderSourceWGSL wgsl{
			.chain = {.sType = WGPUSType_ShaderSourceWGSL},
			.code = {semaphore_wgsl_code, WGPU_STRLEN}
		};
		WGPUShaderModuleDescriptor shader_module_desc{
			.nextInChain = &wgsl.chain
		};
		WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(queue->device, &shader_module_desc);

		WGPUComputePipelineDescriptor increment{
			.layout = pipeline_layout,
			.compute = {
				.module = shader_module,
				.entryPoint = {"cs_increment", WGPU_STRLEN},
			},
		};
		queue->semaphore_increment_pipeline = wgpuDeviceCreateComputePipeline(queue->device, &increment);

		WGPUComputePipelineDescriptor set{
			.layout = pipeline_layout,
			.compute = {
				.module = shader_module,
				.entryPoint = {"cs_set", WGPU_STRLEN},
			},
		};
		queue->semaphore_set_pipeline = wgpuDeviceCreateComputePipeline(queue->device, &set);

		WGPUComputePipelineDescriptor set_max{
			.layout = pipeline_layout,
			.compute = {
				.module = shader_module,
				.entryPoint = {"cs_set_max", WGPU_STRLEN},
			},
		};
		queue->semaphore_set_max_pipeline = wgpuDeviceCreateComputePipeline(queue->device, &set_max);

		wgpuShaderModuleRelease(shader_module);
		wgpuPipelineLayoutRelease(pipeline_layout);
	}

	inline void wait_for_buffer_map(GpuQueue* queue, WGPUBuffer buffer, WGPUMapMode mode, uint64_t offset, uint64_t size) {
		struct Wait { volatile bool done = false; };
		Wait wait;

		WGPUBufferMapCallbackInfo callback{
			.mode = WGPUCallbackMode_AllowSpontaneous,
			.callback = [](WGPUMapAsyncStatus _status, WGPUStringView _message, void* userdata1, void* _userdata2) {
				static_cast<Wait*>(userdata1)->done = true;
			},
			.userdata1 = &wait,
		};
		wgpuBufferMapAsync(buffer, mode, offset, size, callback);

		while (!wait.done) {
		#ifdef __EMSCRIPTEN__
			emscripten_sleep(1); // yields back to the browser event loop
		#else
			wgpuDeviceTick(queue->device);
		#endif
		}
	}

	inline uint64_t semaphore_cpu_set(GpuQueue* queue, GpuSemaphore sema, uint64_t value = 1) {
		wgpuQueueWriteBuffer(queue->queue, sema.buffer, 0, &value, sizeof(value));
		return value;
	}

	inline GpuSemaphore semaphore_initialize(GpuQueue* queue, uint64_t initial_value) {
		GpuSemaphore out{};

		WGPUBufferDescriptor buffer{
			.label = {"NoAPI Semaphore Storage Buffer", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
			.size = sizeof(uint64_t),
		};
		out.buffer = wgpuDeviceCreateBuffer(queue->device, &buffer);

		WGPUBufferDescriptor readback{
			.label = {"NoAPI Semaphore Value Readback Buffer", WGPU_STRLEN},
			.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
			.size = sizeof(uint64_t),
		};
		out.readback_buffer = wgpuDeviceCreateBuffer(queue->device, &readback);

		WGPUBufferDescriptor upload{
			.label = {"NoAPI Semaphore Value Upload Buffer", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
			.size = sizeof(uint64_t),
		};
		out.upload_buffer = wgpuDeviceCreateBuffer(queue->device, &upload);

		ensure_semaphore_pipelines(queue);

		std::array<WGPUBindGroupEntry, 2> entries = {
			WGPUBindGroupEntry{
				.binding = 0,
				.buffer = out.buffer,
				.size = sizeof(uint64_t),
			}, WGPUBindGroupEntry{
				.binding = 1,
				.buffer = out.upload_buffer,
				.size = sizeof(uint64_t),
			}
		};
		WGPUBindGroupDescriptor bind_group{
			.layout = queue->semaphore_bind_group_layout,
			.entryCount = entries.size(),
			.entries = entries.data(),
		};
		out.bind_group = wgpuDeviceCreateBindGroup(queue->device, &bind_group);

		semaphore_cpu_set(queue, out, initial_value);
		return out;
	}

	inline uint64_t semaphore_value(GpuQueue* queue, GpuSemaphore sema) {
		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr);
		wgpuCommandEncoderCopyBufferToBuffer(encoder, sema.buffer, 0, sema.readback_buffer, 0, sizeof(uint64_t));

		WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
		wgpuCommandEncoderRelease(encoder);

		wgpuQueueSubmit(queue->queue, 1, &cmd);
		wgpuCommandBufferRelease(cmd);

		wait_for_buffer_map(queue, sema.readback_buffer, WGPUMapMode_Read, 0, sizeof(uint64_t));

		const void* mapped = wgpuBufferGetConstMappedRange(sema.readback_buffer, 0, sizeof(uint64_t));
		uint64_t result = 0;
		std::memcpy(&result, mapped, sizeof(uint64_t));
		wgpuBufferUnmap(sema.readback_buffer);

		return result;
	}

	inline bool semaphore_wait(GpuQueue* queue, GpuSemaphore sema, uint64_t target_value, std::chrono::nanoseconds timeout = std::chrono::nanoseconds::max(), std::chrono::nanoseconds poll_interval = std::chrono::milliseconds(1)) {
		const bool infinite = (timeout == std::chrono::nanoseconds::max());
		const auto deadline = infinite ? std::chrono::steady_clock::time_point::max() : std::chrono::steady_clock::now() + timeout;

		while (true) {
			if (semaphore_value(queue, sema) >= target_value)
				return true;
			if (!infinite && std::chrono::steady_clock::now() >= deadline)
				return false;
#ifdef __EMSCRIPTEN__
			emscripten_sleep(std::chrono::duration_cast<std::chrono::milliseconds>(poll_interval).count());
#else
			std::this_thread::sleep_for(poll_interval);
#endif
		}
	}

	inline void semaphore_gpu_increment(GpuCommandBuffer* cmd, GpuSemaphore sema) {
		ensure_semaphore_pipelines(cmd->queue);

		WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(cmd->encoder, nullptr);
		wgpuComputePassEncoderSetPipeline(pass, cmd->queue->semaphore_increment_pipeline);
		wgpuComputePassEncoderSetBindGroup(pass, 0, sema.bind_group, 0, nullptr);
		wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
		wgpuComputePassEncoderEnd(pass);
		wgpuComputePassEncoderRelease(pass);
	}

	inline void semaphore_gpu_set(GpuCommandBuffer* cmd, GpuSemaphore sema, uint64_t value = 1) {
		wgpuQueueWriteBuffer(cmd->queue->queue, sema.upload_buffer, 0, &value, sizeof(value));

		ensure_semaphore_pipelines(cmd->queue);

		WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(cmd->encoder, nullptr);
		wgpuComputePassEncoderSetPipeline(pass, cmd->queue->semaphore_set_pipeline);
		wgpuComputePassEncoderSetBindGroup(pass, 0, sema.bind_group, 0, nullptr);
		wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
		wgpuComputePassEncoderEnd(pass);
		wgpuComputePassEncoderRelease(pass);
	}

	inline void semaphore_gpu_set_max(GpuCommandBuffer* cmd, GpuSemaphore sema, uint64_t value = 1) {
		wgpuQueueWriteBuffer(cmd->queue->queue, sema.upload_buffer, 0, &value, sizeof(value));

		ensure_semaphore_pipelines(cmd->queue);

		WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(cmd->encoder, nullptr);
		wgpuComputePassEncoderSetPipeline(pass, cmd->queue->semaphore_set_max_pipeline);
		wgpuComputePassEncoderSetBindGroup(pass, 0, sema.bind_group, 0, nullptr);
		wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
		wgpuComputePassEncoderEnd(pass);
		wgpuComputePassEncoderRelease(pass);
	}

	inline void semaphore_cpu_set_max(GpuQueue* queue, GpuSemaphore sema, uint64_t value = 1) {
		GpuCommandBuffer cmd {
			.queue = queue,
			.encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr)
		};
		semaphore_gpu_set_max(&cmd, sema, value);

		WGPUCommandBuffer cb = wgpuCommandEncoderFinish(cmd.encoder, nullptr);
		wgpuCommandEncoderRelease(cmd.encoder);

		wgpuQueueSubmit(queue->queue, 1, &cb);
		wgpuCommandBufferRelease(cb);
	}

	inline void semaphore_destroy(GpuSemaphore sema) {
		wgpuBindGroupRelease(sema.bind_group);
		wgpuBufferRelease(sema.upload_buffer);
		wgpuBufferRelease(sema.readback_buffer);
		wgpuBufferRelease(sema.buffer);
	}



	// Processess all of the pending code snippets associated with already finished submissions
	inline void process_pending_code(GpuQueue* queue) {
		auto current_finished_submission = GPU::semaphore_value(queue, queue->current_submission_timeline_semaphore);
		if(queue->code_pending_submission_finished.size())
			for(size_t i = queue->code_pending_submission_finished.size(); i--; ) {
				auto& [code, submit] = queue->code_pending_submission_finished[i];
				if(submit <= current_finished_submission) {
					code();
					queue->code_pending_submission_finished.erase(queue->code_pending_submission_finished.begin() + i);
				}
			}
	}

	inline void push_to_monobuffer(GpuCommandBuffer* cmd, GpuQueue::MonobufferRange range, void* cpu) {
		auto size = range.size();
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

	inline size_t push_to_monobuffer(GpuQueue* queue, GpuQueue::MonobufferRange range, void* cpu) {
		auto cmd = gpuStartCommandRecording(queue);
		push_to_monobuffer(cmd, range, cpu);
		return gpuSubmit(queue, {&cmd, 1});

		// If we are just pushing to the buffer I don't think we always care about making sure the process is 100% finished.
		// So returning the submission index which we can wait on if we do care seems fine...
		// GPU::semaphore_wait(queue, queue->current_submission_timeline_semaphore, submit_index);
		// GPU::process_pending_code(queue);
	}

	inline void pull_from_monobuffer(GpuCommandBuffer* cmd, GpuQueue::MonobufferRange range, void* cpu) {
		auto size = range.size();

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

	inline void pull_from_monobuffer(GpuQueue* queue, GpuQueue::MonobufferRange range, void* cpu) {
		auto cmd = gpuStartCommandRecording(queue);
		push_to_monobuffer(cmd, range, cpu);
		auto submit_index = gpuSubmit(queue, {&cmd, 1});

		// When we pull it seems much more likely that we want the cpu memory updated before considering our work "done"
		GPU::semaphore_wait(queue, queue->current_submission_timeline_semaphore, submit_index);
		GPU::process_pending_code(queue);
	}



	inline WGPUTextureDimension texture2wgpu(TEXTURE type){
		switch (type){
		case TEXTURE_1D:
			return WGPUTextureDimension_1D;

		case TEXTURE_2D:
		case TEXTURE_2D_ARRAY:
		case TEXTURE_CUBE:
		case TEXTURE_CUBE_ARRAY:
			return WGPUTextureDimension_2D;

		case TEXTURE_3D:
			return WGPUTextureDimension_3D;

		default:
			return WGPUTextureDimension_2D;
		}
	}

	inline WGPUTextureViewDimension texture_view2wgpu(TEXTURE type){
		switch (type){
		case TEXTURE_2D:
		case TEXTURE_2D_ARRAY:
			return WGPUTextureViewDimension_2DArray;
		case TEXTURE_CUBE_ARRAY:
			return WGPUTextureViewDimension_CubeArray;
		case TEXTURE_3D:
			return WGPUTextureViewDimension_3D;
		default:
			return WGPUTextureViewDimension_2D;
		}
	}

	inline WGPUTextureFormat format2wgpu(FORMAT format){
		switch (format) {
		case FORMAT_RGBA8_UNORM:
			return WGPUTextureFormat_RGBA8Unorm;
		case FORMAT_RGBA8_SRGB:
			return WGPUTextureFormat_RGBA8UnormSrgb;
		case FORMAT_RGBA16_FLOAT:
			return WGPUTextureFormat_RGBA16Float;
		case FORMAT_RGBA32_FLOAT:
			return WGPUTextureFormat_RGBA32Float;
		case FORMAT_RG11B10_FLOAT:
			return WGPUTextureFormat_RG11B10Ufloat;
		case FORMAT_RGB10_A2_UNORM:
			return WGPUTextureFormat_RGB10A2Unorm;
		case FORMAT_R8_UNORM:
			return WGPUTextureFormat_R8Unorm;
		case FORMAT_R16_FLOAT:
			return WGPUTextureFormat_R16Float;
		case FORMAT_R32_FLOAT:
			return WGPUTextureFormat_R32Float;
		case FORMAT_D16_UNORM:
			return WGPUTextureFormat_Depth16Unorm;
		case FORMAT_D24_UNORM_S8_UINT:
			return WGPUTextureFormat_Depth24PlusStencil8;
		case FORMAT_D32_FLOAT:
			return WGPUTextureFormat_Depth32Float;
		case FORMAT_D32_FLOAT_S8_UINT:
			return WGPUTextureFormat_Depth32FloatStencil8;
		default:
			return WGPUTextureFormat_Undefined;
		}
	}

	inline WGPUTextureUsage usage2wgpu(TEXTURE_USAGE_FLAGS flags) {
		WGPUTextureUsage usage = WGPUTextureUsage_None;

		if (flags & USAGE_TRANSFER_SRC)
			usage |= WGPUTextureUsage_CopySrc;

		if (flags & USAGE_TRANSFER_DST)
			usage |= WGPUTextureUsage_CopyDst;

		if (flags & USAGE_SAMPLED)
			usage |= WGPUTextureUsage_TextureBinding;

		if (flags & USAGE_STORAGE)
			usage |= WGPUTextureUsage_StorageBinding;

		if (flags & USAGE_COLOR_ATTACHMENT || flags & USAGE_DEPTH_STENCIL_ATTACHMENT)
			usage |= WGPUTextureUsage_RenderAttachment;

		return usage;
	}

	inline WGPUTextureDescriptor texture2wgpu(const GpuTextureDesc& src, std::string_view label = "") {
		return WGPUTextureDescriptor{
			.label = { .data = label.data(), .length = label.size() },
			.usage = usage2wgpu(src.usage),
			.dimension = texture2wgpu(src.type),
			.size = {
				.width  = src.dimensions.x,
				.height = src.dimensions.y,
				.depthOrArrayLayers = (src.type == TEXTURE_2D_ARRAY || src.type == TEXTURE_CUBE_ARRAY) ? src.layerCount : src.dimensions.z,
			},
			.format = format2wgpu(src.format),
			.mipLevelCount = src.mipCount,
			.sampleCount   = src.sampleCount,
			.viewFormatCount = 0,
			.viewFormats     = nullptr,
		};
	}
}
