#include "noapi.hpp"
#include "../slang_compiler.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <variant>

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
	// Runs whatever was deferred until the submission that still needed it finished. Reading the
	// timeline semaphore here would mean a blocking buffer readback, and in the browser blocking means
	// unwinding to the event loop: a frame that did that between acquiring the presentable texture and
	// submitting would resume in a later task, by which point the canvas texture has expired. So this
	// leans on the submission index gpuSubmitNoFree's completion callback publishes, which costs
	// nothing and never yields. The callback only arrives while the instance is being pumped, which is
	// what the browser does between frames and what wgpuDeviceTick does natively.
	inline void process_pending_code(GpuQueue* queue) {
#ifndef __EMSCRIPTEN__
		wgpuDeviceTick(queue->device);
#endif
		auto current_finished_submission = queue->last_finished_submission;
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
		pull_from_monobuffer(cmd, range, cpu);
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
		case FORMAT_R8_UNORM: return WGPUTextureFormat_R8Unorm;
		case FORMAT_R8_SNORM: return WGPUTextureFormat_R8Snorm;
		case FORMAT_R8_UINT: return WGPUTextureFormat_R8Uint;
		case FORMAT_R8_SINT: return WGPUTextureFormat_R8Sint;

		// The 16-bit normalized formats come from Dawn's unorm16/snorm16 extensions; emdawnwebgpu
		// doesn't expose them, so on the web they fall through to WGPUTextureFormat_Undefined
#ifndef __EMSCRIPTEN__
		case FORMAT_R16_UNORM: return WGPUTextureFormat_R16Unorm;
		case FORMAT_R16_SNORM: return WGPUTextureFormat_R16Snorm;
#endif
		case FORMAT_R16_UINT: return WGPUTextureFormat_R16Uint;
		case FORMAT_R16_SINT: return WGPUTextureFormat_R16Sint;
		case FORMAT_R16_FLOAT: return WGPUTextureFormat_R16Float;
		case FORMAT_RG8_UNORM: return WGPUTextureFormat_RG8Unorm;
		case FORMAT_RG8_SNORM: return WGPUTextureFormat_RG8Snorm;
		case FORMAT_RG8_UINT: return WGPUTextureFormat_RG8Uint;
		case FORMAT_RG8_SINT: return WGPUTextureFormat_RG8Sint;

		case FORMAT_R32_UINT: return WGPUTextureFormat_R32Uint;
		case FORMAT_R32_SINT: return WGPUTextureFormat_R32Sint;
		case FORMAT_R32_FLOAT: return WGPUTextureFormat_R32Float;
#ifndef __EMSCRIPTEN__
		case FORMAT_RG16_UNORM: return WGPUTextureFormat_RG16Unorm;
		case FORMAT_RG16_SNORM: return WGPUTextureFormat_RG16Snorm;
#endif
		case FORMAT_RG16_UINT: return WGPUTextureFormat_RG16Uint;
		case FORMAT_RG16_SINT: return WGPUTextureFormat_RG16Sint;
		case FORMAT_RG16_FLOAT: return WGPUTextureFormat_RG16Float;
		case FORMAT_RGBA8_UNORM: return WGPUTextureFormat_RGBA8Unorm;
		case FORMAT_RGBA8_SRGB: return WGPUTextureFormat_RGBA8UnormSrgb;
		case FORMAT_RGBA8_SNORM: return WGPUTextureFormat_RGBA8Snorm;
		case FORMAT_RGBA8_UINT: return WGPUTextureFormat_RGBA8Uint;
		case FORMAT_RGBA8_SINT: return WGPUTextureFormat_RGBA8Sint;
		case FORMAT_BGRA8_UNORM: return WGPUTextureFormat_BGRA8Unorm;
		case FORMAT_BGRA8_SRGB: return WGPUTextureFormat_BGRA8UnormSrgb;

		case FORMAT_RGB10_A2_UINT: return WGPUTextureFormat_RGB10A2Uint;
		case FORMAT_RGB10_A2_UNORM: return WGPUTextureFormat_RGB10A2Unorm;
		case FORMAT_RG11B10_UFLOAT: return WGPUTextureFormat_RG11B10Ufloat;
		case FORMAT_RGB9E5_UFLOAT: return WGPUTextureFormat_RGB9E5Ufloat;

		case FORMAT_RG32_UINT: return WGPUTextureFormat_RG32Uint;
		case FORMAT_RG32_SINT: return WGPUTextureFormat_RG32Sint;
		case FORMAT_RG32_FLOAT: return WGPUTextureFormat_RG32Float;
#ifndef __EMSCRIPTEN__
		case FORMAT_RGBA16_UNORM: return WGPUTextureFormat_RGBA16Unorm;
		case FORMAT_RGBA16_SNORM: return WGPUTextureFormat_RGBA16Snorm;
#endif
		case FORMAT_RGBA16_UINT: return WGPUTextureFormat_RGBA16Uint;
		case FORMAT_RGBA16_SINT: return WGPUTextureFormat_RGBA16Sint;
		case FORMAT_RGBA16_FLOAT: return WGPUTextureFormat_RGBA16Float;

		case FORMAT_RGBA32_UINT: return WGPUTextureFormat_RGBA32Uint;
		case FORMAT_RGBA32_SINT: return WGPUTextureFormat_RGBA32Sint;
		case FORMAT_RGBA32_FLOAT: return WGPUTextureFormat_RGBA32Float;

		case FORMAT_S8_UINT: return WGPUTextureFormat_Stencil8;
		case FORMAT_D16_UNORM: return WGPUTextureFormat_Depth16Unorm;
		case FORMAT_D24_PLUS: return WGPUTextureFormat_Depth24Plus;
		case FORMAT_D24_PLUS_S8_UINT: return WGPUTextureFormat_Depth24PlusStencil8;
		case FORMAT_D32_FLOAT: return WGPUTextureFormat_Depth32Float;
		case FORMAT_D32_FLOAT_S8_UINT: return WGPUTextureFormat_Depth32FloatStencil8;

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

	// Names a WebGPU format the API can talk about again, or FORMAT_NONE for one it can't. Only
	// needed where WebGPU picks the format rather than us — which today means a surface reporting
	// the formats it prefers (see gpuSurfaceReconfigureEXT).
	inline FORMAT wgpu2format(WGPUTextureFormat format) {
		switch (format) {
		case WGPUTextureFormat_R8Unorm: return FORMAT_R8_UNORM;
		case WGPUTextureFormat_R8Snorm: return FORMAT_R8_SNORM;
		case WGPUTextureFormat_R8Uint: return FORMAT_R8_UINT;
		case WGPUTextureFormat_R8Sint: return FORMAT_R8_SINT;

		// Dawn-only 16-bit normalized formats; see format2wgpu
#ifndef __EMSCRIPTEN__
		case WGPUTextureFormat_R16Unorm: return FORMAT_R16_UNORM;
		case WGPUTextureFormat_R16Snorm: return FORMAT_R16_SNORM;
#endif
		case WGPUTextureFormat_R16Uint: return FORMAT_R16_UINT;
		case WGPUTextureFormat_R16Sint: return FORMAT_R16_SINT;
		case WGPUTextureFormat_R16Float: return FORMAT_R16_FLOAT;
		case WGPUTextureFormat_RG8Unorm: return FORMAT_RG8_UNORM;
		case WGPUTextureFormat_RG8Snorm: return FORMAT_RG8_SNORM;
		case WGPUTextureFormat_RG8Uint: return FORMAT_RG8_UINT;
		case WGPUTextureFormat_RG8Sint: return FORMAT_RG8_SINT;

		case WGPUTextureFormat_R32Uint: return FORMAT_R32_UINT;
		case WGPUTextureFormat_R32Sint: return FORMAT_R32_SINT;
		case WGPUTextureFormat_R32Float: return FORMAT_R32_FLOAT;
#ifndef __EMSCRIPTEN__
		case WGPUTextureFormat_RG16Unorm: return FORMAT_RG16_UNORM;
		case WGPUTextureFormat_RG16Snorm: return FORMAT_RG16_SNORM;
#endif
		case WGPUTextureFormat_RG16Uint: return FORMAT_RG16_UINT;
		case WGPUTextureFormat_RG16Sint: return FORMAT_RG16_SINT;
		case WGPUTextureFormat_RG16Float: return FORMAT_RG16_FLOAT;
		case WGPUTextureFormat_RGBA8Unorm: return FORMAT_RGBA8_UNORM;
		case WGPUTextureFormat_RGBA8UnormSrgb: return FORMAT_RGBA8_SRGB;
		case WGPUTextureFormat_RGBA8Snorm: return FORMAT_RGBA8_SNORM;
		case WGPUTextureFormat_RGBA8Uint: return FORMAT_RGBA8_UINT;
		case WGPUTextureFormat_RGBA8Sint: return FORMAT_RGBA8_SINT;
		case WGPUTextureFormat_BGRA8Unorm: return FORMAT_BGRA8_UNORM;
		case WGPUTextureFormat_BGRA8UnormSrgb: return FORMAT_BGRA8_SRGB;

		case WGPUTextureFormat_RGB10A2Uint: return FORMAT_RGB10_A2_UINT;
		case WGPUTextureFormat_RGB10A2Unorm: return FORMAT_RGB10_A2_UNORM;
		case WGPUTextureFormat_RG11B10Ufloat: return FORMAT_RG11B10_UFLOAT;
		case WGPUTextureFormat_RGB9E5Ufloat: return FORMAT_RGB9E5_UFLOAT;

		case WGPUTextureFormat_RG32Uint: return FORMAT_RG32_UINT;
		case WGPUTextureFormat_RG32Sint: return FORMAT_RG32_SINT;
		case WGPUTextureFormat_RG32Float: return FORMAT_RG32_FLOAT;
#ifndef __EMSCRIPTEN__
		case WGPUTextureFormat_RGBA16Unorm: return FORMAT_RGBA16_UNORM;
		case WGPUTextureFormat_RGBA16Snorm: return FORMAT_RGBA16_SNORM;
#endif
		case WGPUTextureFormat_RGBA16Uint: return FORMAT_RGBA16_UINT;
		case WGPUTextureFormat_RGBA16Sint: return FORMAT_RGBA16_SINT;
		case WGPUTextureFormat_RGBA16Float: return FORMAT_RGBA16_FLOAT;

		case WGPUTextureFormat_RGBA32Uint: return FORMAT_RGBA32_UINT;
		case WGPUTextureFormat_RGBA32Sint: return FORMAT_RGBA32_SINT;
		case WGPUTextureFormat_RGBA32Float: return FORMAT_RGBA32_FLOAT;

		case WGPUTextureFormat_Stencil8: return FORMAT_S8_UINT;
		case WGPUTextureFormat_Depth16Unorm: return FORMAT_D16_UNORM;
		case WGPUTextureFormat_Depth24Plus: return FORMAT_D24_PLUS;
		case WGPUTextureFormat_Depth24PlusStencil8: return FORMAT_D24_PLUS_S8_UINT;
		case WGPUTextureFormat_Depth32Float: return FORMAT_D32_FLOAT;
		case WGPUTextureFormat_Depth32FloatStencil8: return FORMAT_D32_FLOAT_S8_UINT;

		default: return FORMAT_NONE;
		}
	}

	// PRESENT_MODE_BEST_AVAILABLE has no WebGPU spelling, so it is resolved against what the surface
	// supports before this is reached (see GPU::detail::pick_present_mode)
	inline WGPUPresentMode present2wgpu(PRESENT_MODE mode) {
		switch (mode) {
		case PRESENT_MODE_IMMEDIATE: return WGPUPresentMode_Immediate;
		case PRESENT_MODE_FIFO_RELAXED: return WGPUPresentMode_FifoRelaxed;
		case PRESENT_MODE_MAILBOX: return WGPUPresentMode_Mailbox;
		default: return WGPUPresentMode_Fifo;
		}
	}

	inline PRESENT_MODE wgpu2present(WGPUPresentMode mode) {
		switch (mode) {
		case WGPUPresentMode_Immediate: return PRESENT_MODE_IMMEDIATE;
		case WGPUPresentMode_FifoRelaxed: return PRESENT_MODE_FIFO_RELAXED;
		case WGPUPresentMode_Mailbox: return PRESENT_MODE_MAILBOX;
		default: return PRESENT_MODE_FIFO;
		}
	}

	inline WGPUCompareFunction op2wgpu(OP op) {
		switch (op) {
		case OP_NEVER: return WGPUCompareFunction_Never;
		case OP_LESS: return WGPUCompareFunction_Less;
		case OP_EQUAL: return WGPUCompareFunction_Equal;
		case OP_LESS_EQUAL: return WGPUCompareFunction_LessEqual;
		case OP_GREATER: return WGPUCompareFunction_Greater;
		case OP_NOT_EQUAL: return WGPUCompareFunction_NotEqual;
		case OP_GREATER_EQUAL: return WGPUCompareFunction_GreaterEqual;
		case OP_ALWAYS: return WGPUCompareFunction_Always;
		}
		return WGPUCompareFunction_Always;
	}

	inline WGPUStencilOperation stencil2wgpu(STENCIL_OP op) {
		switch (op) {
		case STENCIL_OP_KEEP: return WGPUStencilOperation_Keep;
		case STENCIL_OP_ZERO: return WGPUStencilOperation_Zero;
		case STENCIL_OP_REPLACE: return WGPUStencilOperation_Replace;
		case STENCIL_OP_INCR_SAT: return WGPUStencilOperation_IncrementClamp;
		case STENCIL_OP_DECR_SAT: return WGPUStencilOperation_DecrementClamp;
		case STENCIL_OP_INVERT: return WGPUStencilOperation_Invert;
		case STENCIL_OP_INCR_WRAP: return WGPUStencilOperation_IncrementWrap;
		case STENCIL_OP_DECR_WRAP: return WGPUStencilOperation_DecrementWrap;
		}
		return WGPUStencilOperation_Keep;
	}

	inline WGPUBlendOperation blend2wgpu(BLEND op) {
		switch (op) {
		case BLEND_ADD: return WGPUBlendOperation_Add;
		case BLEND_SUBTRACT: return WGPUBlendOperation_Subtract;
		case BLEND_REV_SUBTRACT: return WGPUBlendOperation_ReverseSubtract;
		case BLEND_MIN: return WGPUBlendOperation_Min;
		case BLEND_MAX: return WGPUBlendOperation_Max;
		}
		return WGPUBlendOperation_Add;
	}

	inline WGPUBlendFactor factor2wgpu(FACTOR factor) {
		switch (factor) {
		case FACTOR_ZERO: return WGPUBlendFactor_Zero;
		case FACTOR_ONE: return WGPUBlendFactor_One;
		case FACTOR_SRC_COLOR: return WGPUBlendFactor_Src;
		case FACTOR_ONE_MINUS_SRC_COLOR: return WGPUBlendFactor_OneMinusSrc;
		case FACTOR_DST_COLOR: return WGPUBlendFactor_Dst;
		case FACTOR_ONE_MINUS_DST_COLOR: return WGPUBlendFactor_OneMinusDst;
		case FACTOR_SRC_ALPHA: return WGPUBlendFactor_SrcAlpha;
		case FACTOR_ONE_MINUS_SRC_ALPHA: return WGPUBlendFactor_OneMinusSrcAlpha;
		case FACTOR_DST_ALPHA: return WGPUBlendFactor_DstAlpha;
		case FACTOR_ONE_MINUS_DST_ALPHA: return WGPUBlendFactor_OneMinusDstAlpha;
		case FACTOR_SRC1_COLOR: return WGPUBlendFactor_Src1;
		case FACTOR_ONE_MINUS_SRC1_COLOR: return WGPUBlendFactor_OneMinusSrc1;
		case FACTOR_SRC1_ALPHA: return WGPUBlendFactor_Src1Alpha;
		case FACTOR_ONE_MINUS_SRC1_ALPHA: return WGPUBlendFactor_OneMinusSrc1Alpha;
		}
		return WGPUBlendFactor_One;
	}

	inline WGPUColorWriteMask mask2wgpu(uint8_t mask) {
		WGPUColorWriteMask out = WGPUColorWriteMask_None;
		if (mask & 0x1) out |= WGPUColorWriteMask_Red;
		if (mask & 0x2) out |= WGPUColorWriteMask_Green;
		if (mask & 0x4) out |= WGPUColorWriteMask_Blue;
		if (mask & 0x8) out |= WGPUColorWriteMask_Alpha;
		return out;
	}

	inline WGPUPrimitiveTopology topology2wgpu(TOPOLOGY topology) {
		switch (topology) {
		case TOPOLOGY_TRIANGLE_LIST: return WGPUPrimitiveTopology_TriangleList;
		case TOPOLOGY_TRIANGLE_STRIP: return WGPUPrimitiveTopology_TriangleStrip;
		}
		return WGPUPrimitiveTopology_TriangleList;
	}

	// We define counter clockwise triangles as front facing (matching the Vulkan backend)
	inline WGPUCullMode cull2wgpu(CULL cull) {
		switch (cull) {
		case CULL_NONE: return WGPUCullMode_None;
		case CULL_CCW: return WGPUCullMode_Front;
		case CULL_CW: return WGPUCullMode_Back;
		case CULL_ALL:
			assert(false && "WebGPU can't cull both faces at once, use an empty color target list instead");
			return WGPUCullMode_Back;
		}
		return WGPUCullMode_None;
	}

	inline WGPUIndexFormat index2wgpu(INDEX_TYPE_EXT type) {
		switch (type) {
		case INDEX_TYPE_UINT16: return WGPUIndexFormat_Uint16;
		case INDEX_TYPE_UINT32: return WGPUIndexFormat_Uint32;
		case INDEX_TYPE_UINT8:
			assert(false && "8 bit indices aren't supported by WebGPU");
			return WGPUIndexFormat_Uint16;
		}
		return WGPUIndexFormat_Uint32;
	}

	// WebGPU has no "don't care", but discarding/clearing expresses the same intent to a tiler
	inline WGPULoadOp load2wgpu(LOAD_OP op) {
		switch (op) {
		case LOAD_OP_LOAD: return WGPULoadOp_Load;
		case LOAD_OP_CLEAR:
		case LOAD_OP_DONT_CARE: return WGPULoadOp_Clear;
		}
		return WGPULoadOp_Load;
	}

	inline WGPUStoreOp store2wgpu(STORE_OP op) {
		switch (op) {
		case STORE_OP_STORE: return WGPUStoreOp_Store;
		case STORE_OP_DONT_CARE: return WGPUStoreOp_Discard;
		}
		return WGPUStoreOp_Store;
	}

	inline bool operator==(const GpuStencil& a, const GpuStencil& b) {
		return a.test == b.test && a.failOp == b.failOp && a.passOp == b.passOp
			&& a.depthFailOp == b.depthFailOp && a.reference == b.reference;
	}

	inline bool operator==(const GpuDepthStencilDesc& a, const GpuDepthStencilDesc& b) {
		return a.depthMode == b.depthMode && a.depthTest == b.depthTest && a.depthBias == b.depthBias
			&& a.depthBiasSlopeFactor == b.depthBiasSlopeFactor && a.depthBiasClamp == b.depthBiasClamp
			&& a.stencilReadMask == b.stencilReadMask && a.stencilWriteMask == b.stencilWriteMask
			&& a.stencilFront == b.stencilFront && a.stencilBack == b.stencilBack;
	}

	inline bool operator==(const GpuBlendDesc& a, const GpuBlendDesc& b) {
		return a.colorOp == b.colorOp && a.srcColorFactor == b.srcColorFactor && a.dstColorFactor == b.dstColorFactor
			&& a.alphaOp == b.alphaOp && a.srcAlphaFactor == b.srcAlphaFactor && a.dstAlphaFactor == b.dstAlphaFactor
			&& a.colorWriteMask == b.colorWriteMask;
	}

	template<typename T>
	inline bool same(const std::optional<T>& a, const std::optional<T>& b) {
		if(a.has_value() != b.has_value()) return false;
		return !a.has_value() || *a == *b;
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




const GpuQueue::MonotextureRange GpuQueue::MonotextureRange::INVALID = {static_cast<uint32_t>(-1), static_cast<uint32_t>(-1), static_cast<uint32_t>(-1)};

void gpuDefaultErrorCallbackEXT(void* queue, int type, GpuStringView message) {
	std::cerr << "WGPU Device Error " << type << ": " << std::string_view(message) << std::endl;
#ifdef _WIN32
	system("pause");
#endif
	exit(-1);
}

// Copies why the setup gave up into the caller's buffer (which is allowed to be absent) and
// reports the failure
static bool setup_failed(char* out_error, size_t out_error_capacity, std::string_view message) {
	if(out_error && out_error_capacity) {
		auto length = std::min(message.size(), out_error_capacity - 1);
		memcpy(out_error, message.data(), length);
		out_error[length] = '\0';
	}
	return false;
}

bool gpuSetupDefaultWebGPUEXT(GpuWebGPUSurfaceLoaderEXT surface_loader, void* surface_loader_userdata,
	GpuErrorCallbackEXT error_callback, bool prefer_high_power,
	GpuWebGPUDefault* out_default, char* out_error, size_t out_error_capacity
) {
	GpuWebGPUDefault out;

	{ // Instance
		WGPUInstanceDescriptor d {

		};
		out.instance = wgpuCreateInstance(&d);
	}
	if(!out.instance) return setup_failed(out_error, out_error_capacity, "Failed to create WebGPU instance");

	out.surface = surface_loader(out.instance, surface_loader_userdata);

	{ // Adapter
		WGPURequestAdapterOptions adapter_opts {
			.powerPreference = prefer_high_power ? WGPUPowerPreference_HighPerformance : WGPUPowerPreference_LowPower,
			.compatibleSurface = out.surface,
		};

		struct RequestAdapterResult {
			WGPUAdapter adapter;
			volatile bool done;
		} result = {};
		wgpuInstanceRequestAdapter(out.instance, &adapter_opts, {
			.mode = WGPUCallbackMode_AllowSpontaneous,
			.callback = +[](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void* userdata2) {
				RequestAdapterResult* result = (RequestAdapterResult*)userdata1;
				auto error_callback = *(GpuErrorCallbackEXT*)userdata2;
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
	if(!out.adapter) return setup_failed(out_error, out_error_capacity, "Failed to create WebGPU adapter");

	out.limits = {}; // Emscripten apparently asserts that the limits have been zeroed out!
	wgpuAdapterGetLimits(out.adapter, &out.limits);

	{ // Device
		WGPUDeviceDescriptor device_desc = {
			.requiredLimits = &out.limits,
			.deviceLostCallbackInfo = {
				.mode = WGPUCallbackMode_AllowSpontaneous,
				.callback = +[](WGPUDevice const* device, WGPUDeviceLostReason reason, WGPUStringView message, void* userdata1, void* _) {
					if(reason == WGPUDeviceLostReason_Destroyed) return;

					auto callback = (void(*)(WGPUDevice const* device, WGPUErrorType type, GpuStringView message))userdata1;
					callback(device, (WGPUErrorType)reason, {message.data, message.length});
				},
				.userdata1 = (void*)error_callback
			},
			.uncapturedErrorCallbackInfo = {
				.callback = +[](WGPUDevice const* device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* _){
					auto callback = (void(*)(WGPUDevice const* device, WGPUErrorType type, GpuStringView message))userdata1;
					callback(device, type, {message.data, message.length});
				},
				.userdata1 = (void*)error_callback
			}
		};

		struct RequestDeviceResult {
			WGPUDevice device;
			volatile bool done;
		} result = {};
		wgpuAdapterRequestDevice(out.adapter, &device_desc, {
			.mode = WGPUCallbackMode_AllowSpontaneous,
			.callback = +[](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void* userdata2) {
				RequestDeviceResult* result = (RequestDeviceResult*)userdata1;
				auto error_callback = *(GpuErrorCallbackEXT*)userdata2;
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
	if(!out.device) return setup_failed(out_error, out_error_capacity, "Failed to create WebGPU device");

	*out_default = out;
	return true;
}

// Group 0 describes the monobuffers, the active texture heap, the sampler/texture metadata buffer,
// and the shader data uniform. Its layout never changes, but compute and graphics need separate ones
// since WebGPU forbids writable storage buffers in the vertex stage (so graphics binds the
// monobuffers read only).
WGPUBindGroupLayout create_buffer_bind_group_layout(GpuQueue* queue, bool compute) {
	auto visibility = compute ? WGPUShaderStage_Compute : WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;

	uint32_t binding = 0;
	std::vector<WGPUBindGroupLayoutEntry> group0;

	for(uint32_t i = 0; i < queue->monobuffers.size(); ++i)
		group0.push_back({
			.binding = binding++,
			.visibility = visibility,
			.buffer = {
				.type = compute ? WGPUBufferBindingType_Storage : WGPUBufferBindingType_ReadOnlyStorage,
				.hasDynamicOffset = false,
				.minBindingSize = 0,
			}
		});

	// The texture heap, then the sampler lookup map and monotexture sizes; both read only
	for(uint32_t i = 0; i < 2; ++i)
		group0.push_back({
			.binding = binding++,
			.visibility = visibility,
			.buffer = {
				.type = WGPUBufferBindingType_ReadOnlyStorage,
				.hasDynamicOffset = false,
				.minBindingSize = 0,
			}
		});

	group0.push_back({
		.binding = binding++,
		.visibility = visibility,
		.buffer = {
			.type = WGPUBufferBindingType_Uniform,
			.hasDynamicOffset = false,
			.minBindingSize = 0,
		}
	});

	WGPUBindGroupLayoutDescriptor d{
		.entryCount = static_cast<uint32_t>(group0.size()),
		.entries = group0.data(),
	};
	return wgpuDeviceCreateBindGroupLayout(queue->device, &d);
}

// Group 3 is one binding per sampler slot. Its shape is fixed, so unlike the texture groups it is
// built once and shared by every sampler set.
WGPUBindGroupLayout create_sampler_bind_group_layout(GpuQueue* queue) {
	std::vector<WGPUBindGroupLayoutEntry> group3;
	for(uint32_t i = 0; i < gpu_sampler_slot_count; ++i)
		group3.push_back({
			.binding = i,
			.visibility = WGPUShaderStage_Compute | WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
			.sampler = {
				// Filtering accepts a nearest sampler too, and group 2 already declares every sampled
				// monotexture as a filterable float texture
				.type = WGPUSamplerBindingType_Filtering,
			}
		});

	WGPUBindGroupLayoutDescriptor d{
		.entryCount = static_cast<uint32_t>(group3.size()),
		.entries = group3.data(),
	};
	return wgpuDeviceCreateBindGroupLayout(queue->device, &d);
}

namespace GPU::detail {
	// Sampler descriptions pack into 16 bits but only ever reach max_packed(), so the lookup map
	// needs one word per value in [0, max_packed()]
	constexpr static uint32_t sampler_lookup_size = GpuSamplerDesc::max_packed() + 1;

	constexpr static WGPUAddressMode address2wgpu(ADDRESS_MODE mode) {
		switch (mode) {
		case ADDRESS_MODE_CLAMP: return WGPUAddressMode_ClampToEdge;
		case ADDRESS_MODE_MIRROR_REPEAT: return WGPUAddressMode_MirrorRepeat;
		case ADDRESS_MODE_REPEAT: return WGPUAddressMode_Repeat;
		}
		return WGPUAddressMode_Repeat;
	}

	constexpr static WGPUFilterMode filter2wgpu(FILTER filter) {
		return filter == FILTER_NEAREST ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
	}

	constexpr static WGPUMipmapFilterMode mip_filter2wgpu(FILTER filter) {
		return filter == FILTER_NEAREST ? WGPUMipmapFilterMode_Nearest : WGPUMipmapFilterMode_Linear;
	}

	// Looks up (or builds) the sampler set for a list of enabled samplers. Slot 0 is always the
	// default sampler, matching the Vulkan backend, so a packed description that was never enabled
	// still resolves to something valid.
	inline GpuQueue::SamplerSet* ensure_sampler_set(GpuQueue* queue, std::span<const GpuSamplerDesc> requested) {
		// One slot is spent on the default sampler, so the caller gets the rest
		assert(requested.size() < gpu_sampler_slot_count && "More samplers enabled at once than there are sampler slots");
		// The assert is gone in a release build, where the clamp below would otherwise hand every
		// sampler past the last slot the default sampler without a word about it
		if(requested.size() >= gpu_sampler_slot_count) errno = WGPUErrorType_Validation;
		auto count = std::min<size_t>(requested.size(), gpu_sampler_slot_count - 1);

		std::vector<GpuSamplerDesc> enabled(count + 1, GpuSamplerDesc{});
		std::copy_n(requested.begin(), count, enabled.begin() + 1);

		if(auto found = queue->sampler_cache.find(enabled); found != queue->sampler_cache.end())
			return &found->second;

		auto& set = queue->sampler_cache[enabled];

		// Every slot has to be filled for the bind group to match the layout, so the ones the caller
		// didn't ask for get a copy of the default sampler
		set.samplers.reserve(gpu_sampler_slot_count);
		for(uint32_t i = 0; i < gpu_sampler_slot_count; ++i) {
			auto& desc = enabled[i < enabled.size() ? i : 0];
			WGPUSamplerDescriptor d {
				.label = {"NoAPI Sampler", WGPU_STRLEN},
				.addressModeU = address2wgpu(desc.address_mode_u),
				.addressModeV = address2wgpu(desc.address_mode_v),
				.addressModeW = address2wgpu(desc.address_mode_w),
				.magFilter = filter2wgpu(desc.mag_filter),
				.minFilter = filter2wgpu(desc.min_filter),
				.mipmapFilter = mip_filter2wgpu(desc.mip_filter),
				.lodMinClamp = 0,
				.lodMaxClamp = 32, // Well past the mip count of any texture WebGPU can allocate
				.maxAnisotropy = 1,
			};
			set.samplers.emplace_back(wgpuDeviceCreateSampler(queue->device, &d));
		}

		std::vector<WGPUBindGroupEntry> entries; entries.reserve(set.samplers.size());
		for(uint32_t i = 0; i < set.samplers.size(); ++i)
			entries.push_back({
				.binding = i,
				.sampler = set.samplers[i],
			});
		WGPUBindGroupDescriptor bind_group {
			.label = {"NoAPI Samplers", WGPU_STRLEN},
			.layout = queue->current_bind_group_layout3,
			.entryCount = static_cast<uint32_t>(entries.size()),
			.entries = entries.data(),
		};
		set.bind_group = wgpuDeviceCreateBindGroup(queue->device, &bind_group);

		// Walked backwards so that the lowest slot wins if the same description was enabled twice
		std::vector<uint32_t> lookup(sampler_lookup_size, 0);
		for(size_t i = enabled.size(); i--; )
			lookup[enabled[i].pack()] = i;

		WGPUBufferDescriptor buffer {
			.label = {"NoAPI Sampler Map", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
			.size = lookup.size() * sizeof(uint32_t),
		};
		set.lookup_buffer = wgpuDeviceCreateBuffer(queue->device, &buffer);
		wgpuQueueWriteBuffer(queue->queue, set.lookup_buffer, 0, lookup.data(), buffer.size);

		return &set;
	}

	inline void free_sampler_set(GpuQueue::SamplerSet& set) {
		if(set.lookup_buffer) wgpuBufferRelease(set.lookup_buffer);
		if(set.bind_group) wgpuBindGroupRelease(set.bind_group);
		for(auto sampler: set.samplers)
			wgpuSamplerRelease(sampler);
	}
}

void update_pipeline_layouts(GpuQueue* queue) {
	auto create_layout_and_group = [&](std::vector<WGPUBindGroupLayoutEntry>& layoutEntries, std::vector<WGPUBindGroupEntry>& entries) -> std::pair<WGPUBindGroupLayout, WGPUBindGroup> {
		WGPUBindGroupLayoutDescriptor layout{
			.entryCount = static_cast<uint32_t>(layoutEntries.size()),
			.entries = layoutEntries.data(),
		};

		WGPUBindGroupDescriptor d {
			.layout = wgpuDeviceCreateBindGroupLayout(queue->device, &layout),
			.entryCount = static_cast<uint32_t>(entries.size()),
			.entries = entries.data(),
		};
		return {d.layout, wgpuDeviceCreateBindGroup(queue->device, &d)};
	};

	//
	// Group 0: buffers (one flavor per pipeline type, neither of which is dynamic)
	//
	if(!queue->current_compute_bind_group_layout0)
		queue->current_compute_bind_group_layout0 = create_buffer_bind_group_layout(queue, true);
	if(!queue->current_graphics_bind_group_layout0)
		queue->current_graphics_bind_group_layout0 = create_buffer_bind_group_layout(queue, false);

	//
	// Group 1: storage textures
	//
	// Groups 1 and 2 are shared between the compute and graphics pipeline layouts (the bind groups
	// built from them get bound to both), so their visibility has to name every stage allowed to
	// touch them. Writable storage textures are illegal in the vertex stage, hence the asymmetry.
	//
	std::vector<WGPUBindGroupLayoutEntry> group1;
	std::vector<WGPUBindGroupEntry> group1entries;

	uint32_t binding = 0;

	for(auto const& [_cap, texture, view, desc] : queue->storage_monotextures) {
		group1.push_back({
			.binding = binding,
			.visibility = WGPUShaderStage_Compute | WGPUShaderStage_Fragment,
			.storageTexture = {
				.access = WGPUStorageTextureAccess_ReadWrite,
				.format = GPU::format2wgpu(desc.format),
				.viewDimension = GPU::texture_view2wgpu(desc.type),
			}
		});
		group1entries.push_back({
			.binding = binding++,
			.textureView = view,
		});
	}

	//
	// Group 2: sampled textures
	//
	std::vector<WGPUBindGroupLayoutEntry> group2;
	std::vector<WGPUBindGroupEntry> group2entries;

	binding = 0;

	for(auto const& [_cap, texture, view, desc] : queue->sampled_monotextures) {
		group2.push_back({
			.binding = binding,
			.visibility = WGPUShaderStage_Compute | WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
			.texture = {
				.sampleType = WGPUTextureSampleType_Float,
				.viewDimension = GPU::texture_view2wgpu(desc.type),
				.multisampled = desc.sampleCount > 1,
			}
		});
		group2entries.push_back({
			.binding = binding++,
			.textureView = view,
		});
	}

	if(queue->current_bind_group_layout1) queue->code_pending_submission_finished.emplace_back([l = queue->current_bind_group_layout1, gp = queue->current_bind_group1](){
		wgpuBindGroupLayoutRelease(l);
		wgpuBindGroupRelease(gp);
	}, queue->next_submission_index);
	std::tie(queue->current_bind_group_layout1, queue->current_bind_group1) = create_layout_and_group(group1, group1entries);

	if(queue->current_bind_group_layout2) queue->code_pending_submission_finished.emplace_back([l = queue->current_bind_group_layout2, gp = queue->current_bind_group2](){
		wgpuBindGroupLayoutRelease(l);
		wgpuBindGroupRelease(gp);
	}, queue->next_submission_index);
	std::tie(queue->current_bind_group_layout2, queue->current_bind_group2) = create_layout_and_group(group2, group2entries);

	//
	// Group 3: samplers
	//
	// Only its contents ever change, never its shape, so the layout outlives every sampler set
	//
	if(!queue->current_bind_group_layout3)
		queue->current_bind_group_layout3 = create_sampler_bind_group_layout(queue);

	auto create_pipeline_layout = [&](WGPUBindGroupLayout bgl0) {
		WGPUBindGroupLayout layouts[] = {
			bgl0,
			queue->current_bind_group_layout1,
			queue->current_bind_group_layout2,
			queue->current_bind_group_layout3
		};

		WGPUPipelineLayoutDescriptor pd{
			.bindGroupLayoutCount = 4,
			.bindGroupLayouts = layouts,
		};
		return wgpuDeviceCreatePipelineLayout(queue->device, &pd);
	};

	if(queue->current_compute_pipeline_layout) wgpuPipelineLayoutRelease(queue->current_compute_pipeline_layout);
	queue->current_compute_pipeline_layout = create_pipeline_layout(queue->current_compute_bind_group_layout0);

	if(queue->current_graphics_pipeline_layout) wgpuPipelineLayoutRelease(queue->current_graphics_pipeline_layout);
	queue->current_graphics_pipeline_layout = create_pipeline_layout(queue->current_graphics_bind_group_layout0);
}



// gpuEncodeWebGPUAddressEXT packs a monobuffer tag above a 61 bit offset and hands the result back as
// a gpu*, and the prologue below has the shader read that pointer as a vec2<u32>. A narrower host
// pointer would truncate the tag away and leave every root data struct half the size its shader
// expects, so the web build is compiled for wasm64 (see -sMEMORY64 in the top level CMakeLists).
static_assert(sizeof(gpu*) == sizeof(uint64_t), "A gpu* has to be 64 bits wide to carry a device address");

constexpr static uint64_t gpu_address_max = 0x1FFFFFFFFFFFFFFF; // (2^61 - 1) aka max number storable in 60 bits
// A gpu pointer reaches WGSL as a vec2<u32>, so the shader works with the halves of those: the bits of
// the high word that still belong to the address, and how far up that word the monobuffer tag sits.
constexpr static uint32_t gpu_address_max_hi = static_cast<uint32_t>(gpu_address_max >> 32);
constexpr static uint32_t gpu_address_tag_shift = std::countr_one(gpu_address_max_hi); // 61 - 32

/**
 * generate_backend_module – This backend's half of the shader ABI, as the Slang module
 * that the portable `noapi` module (see slang_compiler.hpp) is written against.
 *
 * Unlike Vulkan's, this cannot be a fixed string. WGSL has no pointers, so a gpu pointer is
 * the tagged {offset, monobuffer} pair gpuEncodeWebGPUAddressEXT produces and reaching one
 * means switching over every monobuffer the queue has. WGSL also cannot index an array of
 * textures or of samplers, so sampling means switching over every monotexture and every
 * sampler slot. All of those counts are queue state, which is why this is regenerated — and
 * the shaders built against it recompiled — whenever that state changes.
 *
 * The bindings it declares have to match what create_buffer_bind_group_layout and
 * create_sampler_bind_group_layout put in the layouts; Slang maps [[vk::binding(b, s)]]
 * onto WGSL's @binding(b) @group(s), so they are spelled here in the same order.
 */
std::string generate_backend_module(GpuQueue* queue, bool compute) {
	// Has to name the same view dimension GPU::texture_view2wgpu puts in the bind group
	// layouts, or the declaration won't match the view that gets bound. A plain 2D texture is
	// still an array view, since that is how several of them get packed into one monotexture.
	constexpr static auto sampled_type = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_1D: return "Texture1D";
		case WGPUTextureViewDimension_3D: return "Texture3D";
		case WGPUTextureViewDimension_Cube: return "TextureCube";
		case WGPUTextureViewDimension_CubeArray: return "TextureCubeArray";
		case WGPUTextureViewDimension_2DArray: return "Texture2DArray";
		default: return "Texture2D";
		}
	};
	constexpr static auto storage_type = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_1D: return "RWTexture1D";
		case WGPUTextureViewDimension_3D: return "RWTexture3D";
		case WGPUTextureViewDimension_Cube: return "RWTextureCube";
		case WGPUTextureViewDimension_CubeArray: return "RWTextureCubeArray";
		case WGPUTextureViewDimension_2DArray: return "RWTexture2DArray";
		default: return "RWTexture2D";
		}
	};

	// Slang spells storage texture formats the GLSL way and lowers them to the WGSL ones. The
	// set below is exactly the set WGSL can name; everything left out is unreachable as a
	// storage texture whatever the device supports (see the note in wgsl_format's predecessor:
	// no sRGB storage format exists, bgra8unorm needs bgra8unorm-storage, and the narrow and
	// packed formats need texture-formats-tier1, which this backend never asks for).
	constexpr static auto slang_format = [](FORMAT f) {
		switch (f) {
		case FORMAT_R32_UINT: return "r32ui";
		case FORMAT_R32_SINT: return "r32i";
		case FORMAT_R32_FLOAT: return "r32f";
		case FORMAT_RGBA8_UNORM: return "rgba8";
		case FORMAT_RGBA8_SNORM: return "rgba8_snorm";
		case FORMAT_RGBA8_UINT: return "rgba8ui";
		case FORMAT_RGBA8_SINT: return "rgba8i";
		case FORMAT_RG32_UINT: return "rg32ui";
		case FORMAT_RG32_SINT: return "rg32i";
		case FORMAT_RG32_FLOAT: return "rg32f";
		case FORMAT_RGBA16_UINT: return "rgba16ui";
		case FORMAT_RGBA16_SINT: return "rgba16i";
		case FORMAT_RGBA16_FLOAT: return "rgba16f";
		case FORMAT_RGBA32_UINT: return "rgba32ui";
		case FORMAT_RGBA32_SINT: return "rgba32i";
		case FORMAT_RGBA32_FLOAT: return "rgba32f";
		default:
			// Only reached by a texture the validation layer would reject anyway; naming the
			// most common storage format keeps the generated module parseable
			assert(false && "Unsupported storage texture format");
			return "rgba8";
		}
	};

	// A texture only owns part of the monotexture holding it, so its [0, 1] coordinates are scaled
	// onto its share of the atlas before they are sampled, and pinned half a texel inside it so
	// that a bilinear tap can't reach the leftovers past the image. Both numbers come out of the
	// heap entry (see GpuTextureDescriptorImpl), which is what spares this a textureDimensions()
	// call and a divide on every sample. Each of these declares `at`, in whatever type the sample
	// call below wants. Cube coordinates are a direction rather than a position, so they pass
	// through untouched.
	constexpr static auto sample_remap = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_1D:
			return "\t\tlet at = min(uv.x * scale.x, limit.x);\n";
		case WGPUTextureViewDimension_3D:
			return "\t\tlet at = min(uv * scale, limit);\n";
		case WGPUTextureViewDimension_Cube:
			return "\t\tlet at = uv;\n";
		case WGPUTextureViewDimension_CubeArray:
			return "\t\tlet at = uv;\n";
		default:
			return "\t\tlet at = min(uv.xy * scale.xy, limit.xy);\n";
		}
	};

	// The coordinate SampleLevel is actually given. The dimensions that have no array index
	// simply drop it.
	constexpr static auto sample_coordinate = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_2DArray: return "float3(at, float(layer))";
		case WGPUTextureViewDimension_CubeArray: return "float4(at, float(layer))";
		default: return "at";
		}
	};

	std::string out = "module noapi_backend;\n\n";

	//
	// Buffers. WebGPU forbids writable storage buffers in the vertex stage, so the rasterizer
	// only ever gets the monobuffers read only (matching create_buffer_bind_group_layout).
	//
	uint32_t binding = 0;
	for(uint32_t i = 0; i < queue->monobuffers.size(); ++i)
		out += std::format("[[vk::binding({}, 0)]] public {}<uint> mono{};\n",
			binding++, compute ? "RWStructuredBuffer" : "StructuredBuffer", i);

	out += std::format("\n"
		"[[vk::binding({}, 0)]] public StructuredBuffer<uint> texture_heap;\n"
		"[[vk::binding({}, 0)]] public StructuredBuffer<uint> gpu_sampler_map;\n\n", binding, binding + 1);
	binding += 2;

	out += std::string(compute
		? "public struct GpuShaderData {\n\tpublic uint2 compute;\n}\n"
		: "public struct GpuShaderData {\n\tpublic uint2 vertex;\n\tpublic uint2 fragment;\n\tpublic uint2 index;\n}\n")
		+ std::format("[[vk::binding({}, 0)]] public ConstantBuffer<GpuShaderData> shader_data;\n\n", binding++);

	//
	// Pointers. A gpu pointer arrives as the uint2 holding the 64 bits gpuEncodeWebGPUAddressEXT
	// produced: an offset into a monobuffer, tagged in the top bits with which monobuffer that
	// is. Slang has no pointers on this target either, so reaching one is the same switch the
	// hand written WGSL used to spell out -- and, because the portable module builds every
	// wider access out of the one word load, it is the only switch that has to be generated.
	//
	std::string load_cases, store_cases;
	for(uint32_t i = 1; i < queue->monobuffers.size(); ++i) {
		load_cases += std::format("\tcase {}u: return mono{}[index];\n", i, i);
		store_cases += std::format("\tcase {}u: mono{}[index] = value; break;\n", i, i);
	}

	out += std::format(
		"static const uint GPU_ADDRESS_MAX_HI = {:#x}u; // The bits of the high word that are still address\n"
		"static const uint GPU_ADDRESS_TAG_SHIFT = {}u; // Where the monobuffer tag starts in that word\n"
		"\n"
		"public uint2 gpuBackendEncodeAddress(uint monobuffer, uint2 address) {{\n"
		"\tlet tag = monobuffer + 1u; // Zero is left meaning \"no pointer\"\n"
		"\treturn uint2(address.x, (address.y & GPU_ADDRESS_MAX_HI) | (tag << GPU_ADDRESS_TAG_SHIFT));\n"
		"}}\n"
		"\n"
		"// Which monobuffer an address names. The tag is the index plus one, so that a zeroed\n"
		"// pointer (tag 0) is \"no pointer\" rather than a valid address into the first one; it\n"
		"// wraps to a tag no case matches and lands on monobuffer 0, which is the empty buffer.\n"
		"public uint gpuBackendMonobuffer(uint2 address) {{\n"
		"\treturn (address.y >> GPU_ADDRESS_TAG_SHIFT) - 1u;\n"
		"}}\n"
		"\n"
		"// The load half of the memory ABI: one word at the address the portable module hands\n"
		"// over as the 64 bits of the pointer, low word first. Everything wider than a word --\n"
		"// pointers, vectors, matrices, a shader's own structs -- is built out of this one call\n"
		"// by the portable module, so this switch is the whole of what the emulation costs.\n"
		"public uint gpuBackendLoadU32(uint2 address) {{\n"
		"\tlet index = address.x / 4u;\n"
		"\tswitch(gpuBackendMonobuffer(address)) {{\n"
		"{}"
		"\tdefault: return mono0[index];\n"
		"\t}}\n"
		"}}\n"
		"\n", gpu_address_max_hi, gpu_address_tag_shift, load_cases);

	// The rasterizer only gets the monobuffers read only, so a store is compute only. The
	// portable module hides this behind GPU_STORES, so a graphics shader that tries to write
	// gets a compile error rather than a store that quietly goes nowhere.
	if(compute)
		out += std::format(
			"public void gpuBackendStoreU32(uint2 address, uint value) {{\n"
			"\tlet index = address.x / 4u;\n"
			"\tswitch(gpuBackendMonobuffer(address)) {{\n"
			"{}"
			"\tdefault: mono0[index] = value; break;\n"
			"\t}}\n"
			"}}\n"
			"\n", store_cases);

	//
	// Root data
	//
	out += compute
		? "public uint2 gpuBackendRootCompute() { return shader_data.compute; }\n\n"
		: "public uint2 gpuBackendRootVertex() { return shader_data.vertex; }\n"
			"public uint2 gpuBackendRootFragment() { return shader_data.fragment; }\n"
			"public uint2 gpuBackendRootIndex() { return shader_data.index; }\n\n";

	//
	// Textures
	//
	binding = 0;
	for (auto const& [_cap, texture, view, desc] : queue->storage_monotextures) {
		out += std::format("[[vk::binding({}, 1)]] [format(\"{}\")] public {}<float4> storage_tex_{};\n",
			binding, slang_format(desc.format), storage_type(desc.type), binding);
		++binding;
	}

	binding = 0;
	for (auto const& [_cap, texture, view, desc] : queue->sampled_monotextures) {
		out += std::format("[[vk::binding({}, 2)]] public {}<float4> sampled_tex_{};\n",
			binding, sampled_type(desc.type), binding);
		++binding;
	}

	//
	// Samplers, and the metadata buffer that makes them addressable
	//
	out += "\n";
	for (uint32_t i = 0; i < gpu_sampler_slot_count; ++i)
		out += std::format("[[vk::binding({}, 3)]] public SamplerState gpu_sampler_{};\n", i, i);

	auto storage_count = static_cast<uint32_t>(queue->storage_monotextures.size());
	auto sampled_count = static_cast<uint32_t>(queue->sampled_monotextures.size());

	out += std::format("\n"
		"public static const uint GPU_SAMPLER_SLOT_COUNT = {}u;\n"
		"public static const uint GPU_SAMPLER_LOOKUP_SIZE = {}u;\n"
		"public static const uint GPU_STORAGE_TEXTURE_COUNT = {}u;\n"
		"public static const uint GPU_SAMPLED_TEXTURE_COUNT = {}u;\n"
		"\n"
		"// Turns a packed GpuSamplerDesc into the slot holding it, or slot 0 (the default\n"
		"// sampler) when that description was never handed to gpuSetEnabledSamplersEXT\n"
		"public uint gpuBackendSamplerSlot(uint packed) {{\n"
		"\tif(packed >= GPU_SAMPLER_LOOKUP_SIZE) return 0u;\n"
		"\treturn gpu_sampler_map[packed];\n"
		"}}\n"
		"\n"
		"static const uint GPU_MONOTEXTURE_STORAGE_BIT = 0x80000000u;\n"
		"static const uint GPU_MONOTEXTURE_INDEX_MASK = 0x7FFFFFFFu;\n"
		"\n"
		"// One entry of the texture heap, as gpuTextureViewDescriptor lays it out: sixteen words,\n"
		"// one per field. `size` is the texture's own rather than the monotexture's, and `uv_scale`\n"
		"// and `uv_max` are its share of that monotexture and the coordinate a bilinear tap has to\n"
		"// stay inside of -- WGSL can report the monotexture's own size, but not how much of it this\n"
		"// texture owns, so both are worked out on the host and read back here.\n"
		"public struct GpuTextureDescriptor {{\n"
		"\tpublic uint texture_type;\n"
		"\tpublic uint base_mip;\n"
		"\tpublic uint mip_count;\n"
		"\tpublic uint3 size; // width, height, depth\n"
		"\tpublic uint monotexture; // Top bit set means a storage monotexture, the rest is the index\n"
		"\tpublic uint first_layer; // Absolute layer within that monotexture\n"
		"\tpublic uint end_layer;\n"
		"\tpublic float3 uv_scale;\n"
		"\tpublic float3 uv_max;\n"
		"}}\n"
		"\n"
		"public GpuTextureDescriptor gpuBackendTextureDescriptor(uint heap_index) {{\n"
		"\tlet at = heap_index * 16u;\n"
		"\treturn GpuTextureDescriptor(texture_heap[at], texture_heap[at + 1u], texture_heap[at + 2u],\n"
		"\t\tuint3(texture_heap[at + 4u], texture_heap[at + 5u], texture_heap[at + 6u]),\n"
		"\t\ttexture_heap[at + 7u], texture_heap[at + 8u], texture_heap[at + 9u],\n"
		"\t\tfloat3(asfloat(texture_heap[at + 10u]), asfloat(texture_heap[at + 11u]), asfloat(texture_heap[at + 12u])),\n"
		"\t\tfloat3(asfloat(texture_heap[at + 13u]), asfloat(texture_heap[at + 14u]), asfloat(texture_heap[at + 15u])));\n"
		"}}\n"
		"\n"
		"public uint gpuBackendMonotextureIndex(GpuTextureDescriptor descriptor) {{\n"
		"\treturn descriptor.monotexture & GPU_MONOTEXTURE_INDEX_MASK;\n"
		"}}\n"
		"\n"
		"public bool gpuBackendMonotextureIsStorage(GpuTextureDescriptor descriptor) {{\n"
		"\treturn (descriptor.monotexture & GPU_MONOTEXTURE_STORAGE_BIT) != 0u;\n"
		"}}\n"
		"\n",
		gpu_sampler_slot_count, GPU::detail::sampler_lookup_size, storage_count, sampled_count);

	// WGSL can't index an array of samplers or of textures, so the only way to pick either one
	// at runtime is a switch over every combination the queue currently has.
	binding = 0;
	for (auto const& [_cap, texture, view, desc] : queue->sampled_monotextures) {
		auto coordinate = sample_coordinate(desc.type);
		out += std::format("float4 gpu_sample_tex_{}(uint slot, float3 scale, float3 limit, float3 uv, uint layer, float mip) {{\n{}"
			"\tswitch(slot) {{\n", binding, sample_remap(desc.type));
		for (uint32_t i = 1; i < gpu_sampler_slot_count; ++i)
			out += std::format("\tcase {}u: return sampled_tex_{}.SampleLevel(gpu_sampler_{}, {}, mip);\n",
				i, binding, i, coordinate);
		out += std::format("\tdefault: return sampled_tex_{}.SampleLevel(gpu_sampler_0, {}, mip);\n"
			"\t}}\n"
			"}}\n"
			"\n", binding, coordinate);
		++binding;
	}

	out += "// Samples monotexture `texture` on `layer`, through the sampler in `slot` (see\n"
		"// gpuBackendSamplerSlot). `uv` runs 0..1 over the texture itself and is scaled by `scale`\n"
		"// onto whatever part of the monotexture that texture occupies, then held to `limit` so it\n"
		"// stays inside it. `layer` and `mip` are absolute.\n"
		"public float4 gpuBackendSampleTexture(uint texture, uint slot, float3 scale, float3 limit, float3 uv, uint layer, float mip) {\n"
		"\tswitch(texture) {\n";
	for (uint32_t i = 0; i < sampled_count; ++i)
		out += std::format("\tcase {}u: return gpu_sample_tex_{}(slot, scale, limit, uv, layer, mip);\n", i, i);
	out += "\tdefault: return float4(0.0);\n"
		"\t}\n"
		"}\n"
		"\n"
		"// Samples a texture straight out of the heap. `uv` runs 0..1 over the texture, and\n"
		"// `layer` and `mip` are relative to the view the descriptor describes rather than to the\n"
		"// monotexture.\n"
		"public float4 gpuBackendSample(uint heap_index, uint slot, float3 uv, uint layer, float mip) {\n"
		"\tlet descriptor = gpuBackendTextureDescriptor(heap_index);\n"
		"\tlet level = float(descriptor.base_mip) + clamp(mip, 0.0, max(float(descriptor.mip_count), 1.0) - 1.0);\n"
		"\tlet at = descriptor.first_layer + min(layer, descriptor.end_layer - descriptor.first_layer - 1u);\n"
		"\treturn gpuBackendSampleTexture(gpuBackendMonotextureIndex(descriptor), slot,\n"
		"\t\tdescriptor.uv_scale, descriptor.uv_max, uv, at, level);\n"
		"}\n";

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
	assert(limits.maxSamplersPerShaderStage >= gpu_sampler_slot_count && "Device is below the sampler limit WebGPU guarantees");

	out->queue = wgpuDeviceGetQueue(device);

	out->current_submission_timeline_semaphore = GPU::semaphore_initialize(out, out->next_submission_index - 1);

	WGPUBufferDescriptor d {
		.label = {"NoAPI Empty Monobuffer", WGPU_STRLEN},
		.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Index | WGPUBufferUsage_Indirect,
		.size = 4
	};
	out->empty_buffer = wgpuDeviceCreateBuffer(out->device, &d);
	for(auto& buffer: out->monobuffers)
		buffer = wgpuDeviceCreateBuffer(out->device, &d);
	std::fill(out->monobuffer_sizes.begin(), out->monobuffer_sizes.end(), d.size);

	update_pipeline_layouts(out);

	// A command buffer that never calls gpuSetEnabledSamplersEXT still has to bind something, and
	// every set resolves an unknown description to slot 0 anyway, so the default is a set of one
	out->default_sampler_set = GPU::detail::ensure_sampler_set(out, {});
	return out;
}

void gpuFreeQueue(GpuQueue* queue) {
	GPU::semaphore_destroy(queue->current_submission_timeline_semaphore);

	for(auto& [_enabled, set]: queue->sampler_cache)
		GPU::detail::free_sampler_set(set);
	if(queue->current_bind_group_layout3) wgpuBindGroupLayoutRelease(queue->current_bind_group_layout3);

	for(auto [_key, pipeline]: queue->blit_pipelines)
		wgpuRenderPipelineRelease(pipeline);
	for(size_t i = 0; i < queue->blit_bind_group_layouts.size(); ++i) {
		if(queue->blit_samplers[i]) wgpuSamplerRelease(queue->blit_samplers[i]);
		if(queue->blit_pipeline_layouts[i]) wgpuPipelineLayoutRelease(queue->blit_pipeline_layouts[i]);
		if(queue->blit_bind_group_layouts[i]) wgpuBindGroupLayoutRelease(queue->blit_bind_group_layouts[i]);
	}
	if(queue->blit_shader) wgpuShaderModuleRelease(queue->blit_shader);

	auto allocator = queue->cpu_allocator;
	queue->~GpuQueue();
	allocator(queue, 0);
}

// top 3 bits encode monobuffer, rest encodes address
gpu* gpuEncodeWebGPUAddressEXT(uint8_t monobuffer, uint64_t address) {
	assert(monobuffer < GpuQueue{}.monobuffers.size());
	assert(address <= gpu_address_max);

	uint64_t debug = uint64_t(monobuffer + 1) << 61 | address;
	return (gpu*)debug;
}
GpuWebGPUAddressEXT gpuDecodeWebGPUAddressEXT(gpu* addr) {
	auto address = (uint64_t)addr;
	return {(uint8_t)((address >> 61) - 1), address & gpu_address_max};
}

void* gpuMalloc(GpuQueue* queue, size_t bytes, size_t align /* = 16 */, MEMORY memory /* = MEMORY_DEFAULT */) {
	constexpr static auto align_up = [](size_t addr, size_t align) {
		return (addr + align - 1) & ~(align - 1);
	};
	constexpr static auto allocation_bookkeeping = [](GpuQueue* queue, uint8_t active_monobuffer, size_t gpu_address, size_t bytes, MEMORY memory) {
		auto cpu = queue->cpu_allocator(nullptr, bytes);\
		auto gpu = gpuEncodeWebGPUAddressEXT(active_monobuffer, gpu_address);
		queue->allocations[gpu] = {GpuQueue::MonobufferRange{active_monobuffer, static_cast<uint32_t>(gpu_address), static_cast<uint32_t>(gpu_address + bytes)}, cpu, memory};
		queue->cpu2gpu[cpu] = gpu;
		return cpu;
	};

	if(bytes > queue->limits.maxStorageBufferBindingSize) {
		errno = WGPUErrorType_OutOfMemory;
		return nullptr;
	}

	for(size_t i = 0; i < queue->buffer_freelist.size(); ++i) {
		auto& [active_buffer, start, end] = queue->buffer_freelist[i];
		// Strip any empty ranges from the list
		if(start == end) {
			queue->buffer_freelist.erase(queue->buffer_freelist.begin() + i);
			--i;
			continue;
		}

		// If we find a free space big enough for the allocation... use that
		auto aligned = align_up(start, align);
		// >= rather than >, so a hole the allocation exactly fills is still reused. The first half
		// of the test matters because aligning up can walk past the end of a small hole, which
		// would otherwise wrap around into a very large unsigned size.
		if(aligned <= end && size_t(end) - aligned >= bytes) {
			start = aligned + bytes;
			return allocation_bookkeeping(queue, active_buffer, aligned, bytes, memory);
		}

		// If each element overlaps with the previous one merge them
		// assumes that the freelist is sorted by buffer index followed by starting value
		if(i == 0) continue;

		auto [prev_buffer, prev_start, prev_end] = queue->buffer_freelist[i - 1];
		if(start > prev_start && start < prev_end && prev_buffer == active_buffer) {
			queue->buffer_freelist.erase(queue->buffer_freelist.begin() + (i - 1));
			start = prev_start;
		}
	}

	auto new_start = align_up(queue->active_monobuffer_size, align);
	auto new_end = new_start + bytes;
	auto new_active_monobuffer = queue->active_monobuffer;

	// Reallocate the monobuffer if necessary
	bool activate_next_monobuffer = false;
	if(new_end > queue->active_monobuffer_capacity) {
		if(queue->monobuffers[queue->active_monobuffer] != queue->empty_buffer)
			queue->code_pending_submission_finished.emplace_back([buffer = queue->monobuffers[queue->active_monobuffer]]() {
				wgpuBufferRelease(buffer);
			}, queue->next_submission_index);
		else queue->active_monobuffer_capacity = new_end;

		queue->active_monobuffer_capacity = std::max(queue->active_monobuffer_capacity * 2, new_end); // Doubles the size of the capacity each time
		if(queue->active_monobuffer_capacity > queue->limits.maxStorageBufferBindingSize) {
			queue->active_monobuffer_capacity = queue->limits.maxStorageBufferBindingSize;
			activate_next_monobuffer = true;
		}
		if(new_end > queue->limits.maxStorageBufferBindingSize) {
			new_start = 0;
			new_end = bytes;
			++new_active_monobuffer;
		}

		WGPUBufferDescriptor d {
			.label = {"NoAPI Monobuffer", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Index | WGPUBufferUsage_Indirect,
			.size = queue->active_monobuffer_capacity
		};
		auto old_monobuffer = queue->monobuffers[queue->active_monobuffer];
		queue->monobuffers[queue->active_monobuffer] = wgpuDeviceCreateBuffer(queue->device, &d);

		if(old_monobuffer != queue->empty_buffer) {
			auto cmd = wgpuDeviceCreateCommandEncoder(queue->device, nullptr);
			wgpuCommandEncoderCopyBufferToBuffer(cmd, old_monobuffer, 0, queue->monobuffers[queue->active_monobuffer], 0, queue->active_monobuffer_size);
			auto to_submit = wgpuCommandEncoderFinish(cmd, nullptr);
			wgpuQueueSubmit(queue->queue, 1, &to_submit);
			wgpuCommandEncoderRelease(cmd);
			wgpuCommandBufferRelease(to_submit);
		}

		if(activate_next_monobuffer) {
			// Add the rest of the old monobuffer to the freelist
			queue->buffer_freelist.emplace_back(queue->active_monobuffer, queue->active_monobuffer_size, queue->limits.maxStorageBufferBindingSize);

			++queue->active_monobuffer;
			if(queue->active_monobuffer >= queue->monobuffers.size()) { // Fail if monobuffer being activated is the 8th monobuffer
				errno = WGPUErrorType_OutOfMemory;
				return nullptr;
			}

			WGPUBufferDescriptor d {
				.label = {"NoAPI Monobuffer", WGPU_STRLEN},
				.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Index | WGPUBufferUsage_Indirect,
				.size = bytes
			};
			queue->monobuffers[queue->active_monobuffer] = wgpuDeviceCreateBuffer(queue->device, &d);

			if(new_start == 0)
				queue->monobuffer_sizes[queue->active_monobuffer] = queue->active_monobuffer_size = 0;
			else queue->monobuffer_sizes[queue->active_monobuffer] = queue->active_monobuffer_size = bytes;
			queue->active_monobuffer_capacity = bytes;
		}
	}

	if(!activate_next_monobuffer) queue->monobuffer_sizes[queue->active_monobuffer] = queue->active_monobuffer_size = new_end;
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
void gpuFreeDevicePointerEXT(GpuQueue* queue, gpu* ptr) {
	if(!queue->allocations.contains(ptr)) return;

	// TODO: Extra buffers go here
	if(queue->gpu2textures.contains(ptr)) {
		auto texture = queue->gpu2textures[ptr];
		if(texture->range) {
			queue->texture_freelist.emplace_back(*texture->range);
			// Sort the freelist so entries are grouped by monotexture (storage flag + index) and then by start,
			// mirroring buffer_freelist's sort so gpuCreateTexture's freelist search can merge adjacent free ranges
			std::sort(queue->texture_freelist.begin(), queue->texture_freelist.end(), [](const GpuQueue::MonotextureRange& a, const GpuQueue::MonotextureRange& b) -> bool {
				if(a.storage() != b.storage())
					return a.storage() < b.storage();
				if(a._index == b._index)
					return a.start < b.start;
				return a._index < b._index;
			});
		} else {
			wgpuTextureRelease(texture->texture);
		}

		queue->cpu_allocator(texture, 0);
	}

	auto [range, cpu, _memory_type] = queue->allocations[ptr];
	queue->cpu_allocator(cpu, 0);

	queue->allocations.erase(ptr);
	queue->cpu2gpu.erase(cpu);

	queue->buffer_freelist.push_back(range);
	// Sort the freelist so buffers are together, and then by start
	std::sort(queue->buffer_freelist.begin(), queue->buffer_freelist.end(), [](const GpuQueue::MonobufferRange& a, const GpuQueue::MonobufferRange& b) -> bool {
		if(a.buffer == b.buffer)
			return a.start < b.start;

		return a.buffer < b.buffer;
	});
}



GpuTextureSizeAlign gpuTextureSizeAlign(GpuQueue* queue, const GpuTextureDesc& desc) {
	return {16, 16}; // Waste the minimum amount of space dealing with an allocation
}

GpuTexture* gpuCreateTexture(GpuQueue* queue, const GpuTextureDesc& desc, gpu* memory) {
	// Only one texture can be associated with a memory
	if(queue->gpu2textures.contains(memory)) {
		errno = WGPUErrorType_Validation;
		return nullptr;
	}

	auto hash = GpuQueue::TextureHash::from_descriptor(desc);
	std::unordered_map<GpuQueue::TextureHash, size_t, GpuQueue::TextureHash::Hasher>* monotextures_lookup = nullptr;
	std::vector<std::tuple<uint32_t, WGPUTexture, WGPUTextureView, GpuQueue::TextureHash>>* monotextures = nullptr;
	bool is_storage_texture = desc.usage & USAGE_STORAGE;
	if(is_storage_texture) {
		monotextures = &queue->storage_monotextures;
		monotextures_lookup = &queue->storage_monotextures_lookup;
	} else if(desc.usage & USAGE_SAMPLED) {
		monotextures = &queue->sampled_monotextures;
		monotextures_lookup = &queue->sampled_monotextures_lookup;
	}

	std::optional<GpuQueue::MonotextureRange> range = {};

	if(monotextures != nullptr) {
		if(!monotextures_lookup->contains(hash)) {
			if(queue->storage_monotextures.size() >= queue->limits.maxStorageTexturesPerShaderStage) {
				errno = WGPUErrorType_OutOfMemory;
				return nullptr;
			}
			if(queue->storage_monotextures.size() >= queue->limits.maxSampledTexturesPerShaderStage) {
				errno = WGPUErrorType_OutOfMemory;
				return nullptr;
			}
			assert(desc.type != TEXTURE_1D && "1D textures aren't supported by WebGPU");
			assert(desc.type != TEXTURE_3D || desc.layerCount == 1 && "3D textures can't be arrays in WebGPU");

			(*monotextures_lookup)[hash] = monotextures->size();
			auto& [size, texture, full_view, description] = monotextures->emplace_back();
			description = hash;
			size = 0;
			{
				WGPUTextureDescriptor d {
					.usage = GPU::usage2wgpu(desc.usage),
					.dimension = GPU::texture2wgpu(desc.type),
					.size = {desc.dimensions.x, desc.dimensions.y, desc.type == TEXTURE_3D ? desc.dimensions.z : desc.layerCount * 2},
					.format = GPU::format2wgpu(desc.format),
					.mipLevelCount = static_cast<uint32_t>(std::log2(std::max(desc.dimensions.x, desc.dimensions.y)) + 1),
					.sampleCount = desc.sampleCount,
				};
				texture = wgpuDeviceCreateTexture(queue->device, &d);
			}{
				WGPUTextureViewDescriptor d {
					.format = GPU::format2wgpu(desc.format),
					.dimension = GPU::texture_view2wgpu(desc.type),
					.baseMipLevel = 0,
					.mipLevelCount = static_cast<uint32_t>(std::log2(std::max(desc.dimensions.x, desc.dimensions.y)) + 1), // TODO: Does this need a -1
					.baseArrayLayer = 0,
					.arrayLayerCount = desc.type == TEXTURE_3D ? 1 : desc.layerCount * 2,
					.aspect = WGPUTextureAspect_All, // TODO: Do we need to deal with depthonly or stencilonly?
					.usage = GPU::usage2wgpu(desc.usage),
				};
				full_view = wgpuTextureCreateView(texture, &d);
			}

			// We have a new monotexture the pipeline layouts should be aware of
			update_pipeline_layouts(queue);
		}

		auto texture_index = monotextures_lookup->at(hash);
		auto& [size, texture, full_view, _description] = monotextures->at(texture_index);

		for(size_t i = 0; i < queue->texture_freelist.size(); ++i) {
			auto& free = queue->texture_freelist[i];
			// Strip any empty ranges from the list
			if(free.start == free.end) {
				queue->texture_freelist.erase(queue->texture_freelist.begin() + i);
				--i;
				continue;
			}

			// Only consider free ranges belonging to this exact monotexture atlas
			if(free.storage() != is_storage_texture || free.index() != texture_index) continue;

			// If we find a free space big enough for the allocation... use that
			if(free.end - free.start >= desc.layerCount) {
				range = {._index = free._index, .start = free.start, .end = free.start + desc.layerCount};
				free.start += desc.layerCount;
				break;
			}

			// If each element overlaps with the previous one merge them
			// assumes that the freelist is sorted by storage then index then starting value
			if(i == 0) continue;

			auto prev = queue->texture_freelist[i - 1];
			if(free.start > prev.start && free.start < prev.end && prev._index == free._index) {
				queue->texture_freelist.erase(queue->texture_freelist.begin() + (i - 1));
				free.start = prev.start;
			}
		}

		if(!range) {
			auto capacity = desc.type == TEXTURE_3D ? 1 : wgpuTextureGetDepthOrArrayLayers(texture);
			if(size + desc.layerCount > capacity) {
				{
					queue->code_pending_submission_finished.emplace_back([texture]() {
						wgpuTextureRelease(texture);
					}, queue->next_submission_index);

					WGPUTextureDescriptor d {
						.usage = GPU::usage2wgpu(desc.usage),
						.dimension = GPU::texture2wgpu(desc.type),
						.size = {desc.dimensions.x, desc.dimensions.y, desc.type == TEXTURE_3D ? desc.dimensions.z : size * 2},
						.format = GPU::format2wgpu(desc.format),
						.mipLevelCount = static_cast<uint32_t>(std::log2(std::max(desc.dimensions.x, desc.dimensions.y)) + 1),
						.sampleCount = desc.sampleCount,
					};
					texture = wgpuDeviceCreateTexture(queue->device, &d);
				}{
					queue->code_pending_submission_finished.emplace_back([full_view]() {
						wgpuTextureViewRelease(full_view);
					}, queue->next_submission_index);

					WGPUTextureViewDescriptor d {
						.format = GPU::format2wgpu(desc.format),
						.dimension = GPU::texture_view2wgpu(desc.type),
						.baseMipLevel = 0,
						.mipLevelCount = static_cast<uint32_t>(std::log2(std::max(desc.dimensions.x, desc.dimensions.y)) + 1), // TODO: Does this need a -1
						.baseArrayLayer = 0,
						.arrayLayerCount = static_cast<uint32_t>(size * 2),
						.aspect = WGPUTextureAspect_All, // TODO: Do we need to deal with depthonly or stencilonly?
						.usage = GPU::usage2wgpu(desc.usage),
					};
					full_view = wgpuTextureCreateView(texture, &d);
				}
			}

			range = {.start = size};
			range->set_storage(is_storage_texture);
			range->set_index(texture_index);
			size += desc.layerCount;
			range->end = size; // TODO: Plus 1?
		}
	}

	auto out = (GpuTexture*)queue->cpu_allocator(nullptr, sizeof(GpuTexture));
	new(out) GpuTexture{.descriptor = desc, .range = range};
	if(range) {
		out->texture = std::get<WGPUTexture>((*monotextures)[range->index()]);
		// TODO: Texture view?
	} else { // If we are creating a purely bound texture rather than a lookup texture we are fine
		WGPUTextureDescriptor d {
			.usage = GPU::usage2wgpu(desc.usage),
			.dimension = GPU::texture2wgpu(desc.type),
			.size = {desc.dimensions.x, desc.dimensions.y, desc.type == TEXTURE_3D ? desc.dimensions.z : desc.layerCount},
			.format = GPU::format2wgpu(desc.format),
			.mipLevelCount = static_cast<uint32_t>(std::log2(std::max(desc.dimensions.x, desc.dimensions.y)) + 1),
			.sampleCount = desc.sampleCount,
		};
		out->texture = wgpuDeviceCreateTexture(queue->device, &d);
		// TODO: Texture view?
	}

	queue->gpu2textures[memory] = out;
	return out;
}

GpuTextureDescriptor gpuTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc) {
	// A shader reaches a texture by indexing the monotexture holding it, so a texture that never landed
	// in one can't be described. That is every texture created without USAGE_SAMPLED or USAGE_STORAGE,
	// and a surface's presentable texture, which WebGPU allocates itself and never lets us place in an
	// atlas. Blit it into a sampled texture (gpuBlitTextureEXT) and describe that instead.
	assert(texture->range && "This texture has no monotexture slot, so no descriptor can name it");

	GpuTextureDescriptorImpl out {
		.type = static_cast<uint32_t>(texture->descriptor.type),
		.baseMip = desc.baseMip,
		.mipCount = desc.mipCount == ALL_MIPS ? texture->descriptor.mipCount - desc.baseMip : desc.mipCount,
		.width = texture->descriptor.dimensions.x,
		.height = texture->descriptor.dimensions.y,
		.depth = texture->descriptor.dimensions.z,
		.range = texture->range ? *texture->range : GpuQueue::MonotextureRange::INVALID
	};

	// The texture's share of the monotexture holding it, which is what the shader scales its
	// coordinates by, and how far into that share a bilinear tap may reach. A texture that fills
	// its slot -- which is every power of two one, since the bucket rounds dimensions up to a power
	// of two -- gets a scale of exactly 1 and no limit at all, so the sampler's own addressing mode
	// still applies across the whole range and REPEAT tiles the way it was asked to. One that
	// doesn't fill it is held half a texel inside its image instead, which is what keeps the
	// leftovers of the slot out of the filter and is why nothing has to write them.
	//
	// Layers are addressed separately, so only a 3D texture scales its third axis; for everything
	// else the image and its slot are one layer deep and the axis falls out unscaled.
	const uint32_t image[3] = {
		texture->descriptor.dimensions.x, texture->descriptor.dimensions.y, texture->descriptor.dimensions.z
	};
	const uint32_t slot[3] = {
		std::max(wgpuTextureGetWidth(texture->texture), 1u),
		std::max(wgpuTextureGetHeight(texture->texture), 1u),
		texture->descriptor.type == TEXTURE_3D
			? std::max(wgpuTextureGetDepthOrArrayLayers(texture->texture), 1u) : std::max(image[2], 1u),
	};
	for(int axis = 0; axis < 3; ++axis) {
		out.uvScale[axis] = float(image[axis]) / float(slot[axis]);
		out.uvMax[axis] = image[axis] >= slot[axis]
			? GPU_UV_UNCLAMPED
			: (float(image[axis]) - 0.5f) / float(slot[axis]);
	}

	if(texture->range) {
		auto check = texture->range;
		out.range.start += desc.baseLayer;
		if(desc.layerCount != ALL_LAYERS)
			out.range.end = out.range.start + desc.layerCount;
		assert(out.range.start >= check->start);
		assert(out.range.end <= check->end);
	}
	return (GpuTextureDescriptor&)out;
}
GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc) {
	return gpuTextureViewDescriptor(queue, texture, desc);
}

/**
 * create_shader_module – Compile the Slang source a pipeline was handed into WGSL for one
 * stage and wrap the result as a shader module.
 *
 * The module it is linked against is the one generated for the queue's *current* bindings,
 * which is why this is called again (rather than the pipeline being reused) whenever the
 * monobuffer or monotexture set moves underneath it.
 *
 * @return The shader module, or null, having reported why, if it did not compile.
 */
WGPUShaderModule create_shader_module(GpuQueue* queue, std::string_view IR, GPU::shaders::SHADER_STAGE stage) {
	auto compute = stage == GPU::shaders::SHADER_STAGE::COMPUTE;

	std::string error;
	auto wgsl = GPU::shaders::compile(SLANG_WGSL, generate_backend_module(queue, compute), IR, stage, {
		{"GPU_COMPUTE", compute ? "1" : "0"},
		{"GPU_GRAPHICS", compute ? "0" : "1"},
		// The rasterizer only gets the monobuffers read only (see create_buffer_bind_group_layout),
		// so the portable module only offers stores to a compute shader
		{"GPU_STORES", compute ? "1" : "0"},
	}, error);

	if(!wgsl) {
		GPU::shaders::report_diagnostic(error);
		errno = WGPUErrorType_Validation;
		return nullptr;
	}

	WGPUShaderSourceWGSL source {
		.chain {.sType = WGPUSType_ShaderSourceWGSL },
		.code = {wgsl->data(), wgsl->size()}
	};
	WGPUShaderModuleDescriptor module {
		.nextInChain = &source.chain
	};
	return wgpuDeviceCreateShaderModule(queue->device, &module);
}

void update_compute_pipeline(GpuQueue* queue, const GpuPipeline* pipeline) {
	auto& cache = std::get<GpuPipeline::ComputeCache>(pipeline->cache);
	if(cache.pipeline) wgpuComputePipelineRelease(cache.pipeline);

	WGPUComputePipelineDescriptor d {
		.layout = queue->current_compute_pipeline_layout,
		.compute = {
			.module = create_shader_module(queue, cache.IR, GPU::shaders::SHADER_STAGE::COMPUTE),
			// Leaving the entry point undefined picks the module's only one, which is all Slang
			// emitted for this stage
			.entryPoint = WGPU_STRING_VIEW_INIT,
		}
	};
	cache.pipeline = wgpuDeviceCreateComputePipeline(queue->device, &d);
	pipeline->reference_layout = queue->current_compute_pipeline_layout;

	wgpuShaderModuleRelease(d.compute.module);
}

// Rebuilds the cached variant of a graphics pipeline against the queue's current pipeline layout
// and the depth/stencil, blend, and index state that is about to be drawn with. WebGPU bakes all
// of them into the PSO, so every one of them is recorded alongside the pipeline for comparison.
void update_render_pipeline(GpuQueue* queue, const GpuPipeline* pipeline, const std::optional<GpuDepthStencilDesc>& depth_stencil, const std::optional<GpuBlendDesc>& blend, INDEX_TYPE_EXT index_type) {
	auto& cache = std::get<GpuPipeline::RenderCache>(pipeline->cache);
	if(cache.pipeline) wgpuRenderPipelineRelease(cache.pipeline);

	auto& desc = cache.descriptor;

	// A dynamically applied blend state takes precedence over one baked into the rasterizer description
	auto effective_blend = blend ? blend : std::optional<GpuBlendDesc>(desc.blendstate);
	WGPUBlendState blend_state {
		.color = {
			.operation = GPU::blend2wgpu(effective_blend.value_or(GpuBlendDesc{}).colorOp),
			.srcFactor = GPU::factor2wgpu(effective_blend.value_or(GpuBlendDesc{}).srcColorFactor),
			.dstFactor = GPU::factor2wgpu(effective_blend.value_or(GpuBlendDesc{}).dstColorFactor),
		},
		.alpha = {
			.operation = GPU::blend2wgpu(effective_blend.value_or(GpuBlendDesc{}).alphaOp),
			.srcFactor = GPU::factor2wgpu(effective_blend.value_or(GpuBlendDesc{}).srcAlphaFactor),
			.dstFactor = GPU::factor2wgpu(effective_blend.value_or(GpuBlendDesc{}).dstAlphaFactor),
		},
	};

	std::vector<WGPUColorTargetState> targets; targets.reserve(desc.colorTargets.size());
	for(auto& target: desc.colorTargets) {
		// The target's baked write mask and the blend state's write mask both have to be satisfied
		uint8_t mask = effective_blend ? (target.writeMask & effective_blend->colorWriteMask) : target.writeMask;
		targets.push_back({
			.format = GPU::format2wgpu(target.format),
			.blend = effective_blend ? &blend_state : nullptr,
			.writeMask = GPU::mask2wgpu(mask),
		});
	}

	auto depth_stencil_format = desc.depthFormat != FORMAT_NONE ? desc.depthFormat : desc.stencilFormat;
	auto state = depth_stencil.value_or(GpuDepthStencilDesc{});
	auto has_depth = gpuFormatIsDepthEXT(depth_stencil_format);
	auto has_stencil = gpuFormatIsStencilEXT(depth_stencil_format);

	auto stencil2wgpu = [](const GpuStencil& stencil) {
		return WGPUStencilFaceState {
			.compare = GPU::op2wgpu(stencil.test),
			.failOp = GPU::stencil2wgpu(stencil.failOp),
			.depthFailOp = GPU::stencil2wgpu(stencil.depthFailOp),
			.passOp = GPU::stencil2wgpu(stencil.passOp),
		};
	};
	// WebGPU rejects any non default state for an aspect the attachment format doesn't have
	constexpr static WGPUStencilFaceState stencil_disabled {
		.compare = WGPUCompareFunction_Always,
		.failOp = WGPUStencilOperation_Keep,
		.depthFailOp = WGPUStencilOperation_Keep,
		.passOp = WGPUStencilOperation_Keep,
	};
	WGPUDepthStencilState depth_stencil_state {
		.format = GPU::format2wgpu(depth_stencil_format),
		.depthWriteEnabled = (has_depth && (state.depthMode & DEPTH_WRITE)) ? WGPUOptionalBool_True : WGPUOptionalBool_False,
		.depthCompare = (has_depth && (state.depthMode & DEPTH_READ)) ? GPU::op2wgpu(state.depthTest) : WGPUCompareFunction_Always,
		.stencilFront = has_stencil ? stencil2wgpu(state.stencilFront) : stencil_disabled,
		.stencilBack = has_stencil ? stencil2wgpu(state.stencilBack) : stencil_disabled,
		.stencilReadMask = has_stencil ? static_cast<uint32_t>(state.stencilReadMask) : 0xFFFFFFFFu,
		.stencilWriteMask = has_stencil ? static_cast<uint32_t>(state.stencilWriteMask) : 0xFFFFFFFFu,
		// WebGPU counts the constant bias in units of the smallest representable depth value
		.depthBias = static_cast<int32_t>(state.depthBias),
		.depthBiasSlopeScale = state.depthBiasSlopeFactor,
		.depthBiasClamp = state.depthBiasClamp,
	};

	auto vertex_module = create_shader_module(queue, cache.vertexIR, GPU::shaders::SHADER_STAGE::VERTEX);
	auto fragment_module = cache.fragmentIR.empty() ? nullptr
		: create_shader_module(queue, cache.fragmentIR, GPU::shaders::SHADER_STAGE::FRAGMENT);

	WGPUFragmentState fragment {
		.module = fragment_module,
		// Undefined rather than left alone: a zeroed WGPUStringView is the empty string, which Dawn
		// then goes looking for as an entry point name
		.entryPoint = WGPU_STRING_VIEW_INIT,
		.targetCount = targets.size(),
		.targets = targets.data(),
	};
	WGPURenderPipelineDescriptor d {
		.layout = queue->current_graphics_pipeline_layout,
		.vertex = {
			.module = vertex_module,
			// Leaving the entry point undefined picks the module's only entry point for the stage,
			// which lets a single blob holding both shaders be handed to us for both stages
			.entryPoint = WGPU_STRING_VIEW_INIT,
		},
		.primitive = {
			.topology = GPU::topology2wgpu(desc.topology),
			// Strips need to know how the index buffer restarts primitives, lists must not name a format
			.stripIndexFormat = desc.topology == TOPOLOGY_TRIANGLE_STRIP ? GPU::index2wgpu(index_type) : WGPUIndexFormat_Undefined,
			.frontFace = WGPUFrontFace_CCW,
			.cullMode = GPU::cull2wgpu(desc.cull),
		},
		.depthStencil = depth_stencil_format == FORMAT_NONE ? nullptr : &depth_stencil_state,
		.multisample = {
			.count = desc.sampleCount,
			.mask = 0xFFFFFFFF,
			.alphaToCoverageEnabled = desc.alphaToCoverage,
		},
		.fragment = fragment_module ? &fragment : nullptr,
	};
	cache.pipeline = wgpuDeviceCreateRenderPipeline(queue->device, &d);

	pipeline->reference_layout = queue->current_graphics_pipeline_layout;
	cache.depth_stencil = depth_stencil;
	cache.blend = blend;
	cache.index_type = index_type;

	wgpuShaderModuleRelease(vertex_module);
	if(fragment_module) wgpuShaderModuleRelease(fragment_module);
}

GpuPipeline* gpuCreateComputePipeline(GpuQueue* queue, GpuByteSpan computeIR) {
	auto out = (GpuPipeline*)queue->cpu_allocator(nullptr, sizeof(GpuPipeline));
	new(out) GpuPipeline{
		.reference_layout = nullptr,
		.cache = GpuPipeline::ComputeCache {
			.IR = {(char*)computeIR.data(), (char*)(computeIR.data() + computeIR.size())},
		}
	};

	update_compute_pipeline(queue, out);
	return out;
}

GpuPipeline* gpuCreateGraphicsPipeline(GpuQueue* queue, GpuByteSpan vertexIR, GpuByteSpan fragmentIR, const GpuRasterDesc& desc) {
	auto out = (GpuPipeline*)queue->cpu_allocator(nullptr, sizeof(GpuPipeline));
	new(out) GpuPipeline{
		.reference_layout = nullptr,
		.cache = GpuPipeline::RenderCache {
			.vertexIR = {(char*)vertexIR.data(), (char*)(vertexIR.data() + vertexIR.size())},
			.fragmentIR = {(char*)fragmentIR.data(), (char*)(fragmentIR.data() + fragmentIR.size())},
			.color_targets = {desc.colorTargets.begin(), desc.colorTargets.end()},
			.descriptor = desc,
		}
	};

	// The descriptor only borrows its color targets, so point it at our copy of them
	auto& cache = std::get<GpuPipeline::RenderCache>(out->cache);
	cache.descriptor.colorTargets = cache.color_targets;

	// Unlike a compute pipeline we can't build anything yet: the depth/stencil state, the blend
	// state, and the index format are all baked into a WebGPU PSO but none of them are bound until
	// a draw is recorded. The first draw builds the initial variant.
	return out;
}

void gpuFreePipeline(GpuQueue* queue, GpuPipeline* pipeline) {
	if(std::holds_alternative<GpuPipeline::ComputeCache>(pipeline->cache)) {
		auto& cache = std::get<GpuPipeline::ComputeCache>(pipeline->cache);
		if(cache.pipeline) wgpuComputePipelineRelease(cache.pipeline);
	} else {
		auto& cache = std::get<GpuPipeline::RenderCache>(pipeline->cache);
		if(cache.pipeline) wgpuRenderPipelineRelease(cache.pipeline);
	}

	pipeline->~GpuPipeline();
	queue->cpu_allocator(pipeline, 0);
}

GpuDepthStencilState* gpuCreateDepthStencilState(GpuQueue* queue, const GpuDepthStencilDesc& desc) {
	auto out = (GpuDepthStencilState*)queue->cpu_allocator(nullptr, sizeof(GpuDepthStencilState));
	new(out) GpuDepthStencilState{.descriptor = desc};
	return out;
}

GpuBlendState* gpuCreateBlendState(GpuQueue* queue, const GpuBlendDesc& desc) {
	auto out = (GpuBlendState*)queue->cpu_allocator(nullptr, sizeof(GpuBlendState));
	new(out) GpuBlendState{.descriptor = desc};
	return out;
}

void gpuFreeDepthStencilState(GpuQueue* queue, GpuDepthStencilState* state) {
	state->~GpuDepthStencilState();
	queue->cpu_allocator(state, 0);
}

void gpuFreeBlendState(GpuQueue* queue, GpuBlendState* state) {
	state->~GpuBlendState();
	queue->cpu_allocator(state, 0);
}

GpuCommandBuffer* gpuStartCommandRecording(GpuQueue* queue) {
	auto out = (GpuCommandBuffer*)queue->cpu_allocator(nullptr, sizeof(GpuCommandBuffer));
	new(out) GpuCommandBuffer{
		.queue = queue,
		.encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr)
	};

	GPU::process_pending_code(queue);

	return out;
}

void endRenderPass(GpuCommandBuffer* cmd) {
	if(!cmd->render_pass) return;

	wgpuRenderPassEncoderEnd(cmd->render_pass);
	cmd->queue->code_pending_submission_finished.emplace_back([pass = cmd->render_pass] {
		wgpuRenderPassEncoderRelease(pass);
	}, cmd->queue->next_submission_index);
	cmd->render_pass = nullptr;
	cmd->bound_render_pipeline = nullptr; // Nothing set on a pass outlives it
}

void endComputePass(GpuCommandBuffer* cmd) {
	if(!cmd->compute_pass) return;

	wgpuComputePassEncoderEnd(cmd->compute_pass);
	cmd->queue->code_pending_submission_finished.emplace_back([pass = cmd->compute_pass] {
		wgpuComputePassEncoderRelease(pass);
	}, cmd->queue->next_submission_index);
	cmd->compute_pass = nullptr;
}

void endCurrentPass(GpuCommandBuffer* cmd) {
	endRenderPass(cmd);
	endComputePass(cmd);
}

void gpuFreeCommandBuffer(GpuCommandBuffer* cmd) {
	wgpuCommandEncoderRelease(cmd->encoder);

	auto allocator = cmd->queue->cpu_allocator;
	cmd->~GpuCommandBuffer();
	allocator(cmd, 0);
}

uint64_t gpuSubmitNoFreeEXT(GpuQueue* queue, GpuCommandBufferSpan command_buffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signal_value /* = 0 */) {
	std::vector<WGPUCommandBuffer> buffers; buffers.reserve(command_buffers.size() + 1);
	for(auto buffer: command_buffers) {
		endCurrentPass(buffer);
		buffers.emplace_back(wgpuCommandEncoderFinish(buffer->encoder, nullptr));
	}

	GpuCommandBuffer cmd {
		.queue = queue,
		.encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr),
	};
	GPU::semaphore_gpu_increment(&cmd, queue->current_submission_timeline_semaphore);
	if(semaphore)
		GPU::semaphore_gpu_set_max(&cmd, *semaphore, signal_value);

	buffers.emplace_back(wgpuCommandEncoderFinish(cmd.encoder, nullptr));
	wgpuCommandEncoderRelease(cmd.encoder);

	wgpuQueueSubmit(queue->queue, buffers.size(), buffers.data());

	// Publishes this submission's index once the queue is done with it, so that
	// GPU::process_pending_code can tell what is safe to reclaim without reading the timeline
	// semaphore back. The index travels in userdata rather than a capture because the callback is a
	// plain function pointer; a gpu* is 64 bits wide on every target this backend builds for (see the
	// static_assert above gpu_address_max), so a submission index fits in one.
	wgpuQueueOnSubmittedWorkDone(
		queue->queue,
		WGPUQueueWorkDoneCallbackInfo {
			.mode = WGPUCallbackMode_AllowSpontaneous,
#ifdef __EMSCRIPTEN__
			.callback = [](WGPUQueueWorkDoneStatus status, WGPU_NULLABLE void* userdata1, WGPU_NULLABLE void* userdata2) {
#else
			.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView message, WGPU_NULLABLE void* userdata1, WGPU_NULLABLE void* userdata2) {
#endif
				auto queue = static_cast<GpuQueue*>(userdata1);
				auto finished = reinterpret_cast<size_t>(userdata2);
				// Completion callbacks can arrive out of order, and an older one must not walk the
				// mark backwards over deletes a newer one already released
				if(finished > queue->last_finished_submission)
					queue->last_finished_submission = finished;
			},
			.userdata1 = queue,
			.userdata2 = reinterpret_cast<void*>(queue->next_submission_index)
		}
	);

	for(auto buffer: command_buffers)
		for(auto code: buffer->code_pending_submission_finished)
			queue->code_pending_submission_finished.emplace_back(code, queue->next_submission_index);

	return queue->next_submission_index++;
}

uint64_t gpuSubmit(GpuQueue* queue, GpuCommandBufferSpan command_buffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signal_value /* = 0 */) {
	auto submission = gpuSubmitNoFreeEXT(queue, command_buffers, semaphore, signal_value);
	for(auto cmd: command_buffers)
		gpuFreeCommandBuffer(cmd);
	return submission;
}

void gpuWaitIdleEXT(GpuQueue* queue) {
	struct Wait { volatile bool done = false; } wait;

    wgpuQueueOnSubmittedWorkDone(
        queue->queue,
		WGPUQueueWorkDoneCallbackInfo {
			.mode = WGPUCallbackMode_AllowSpontaneous,
#ifdef __EMSCRIPTEN__
			.callback = [](WGPUQueueWorkDoneStatus status, WGPU_NULLABLE void* userdata1, WGPU_NULLABLE void* userdata2) {
				static_cast<Wait*>(userdata1)->done = true;
			},
#else
			.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView message, WGPU_NULLABLE void* userdata1, WGPU_NULLABLE void* userdata2) {
				static_cast<Wait*>(userdata1)->done = true;
			},
#endif
        	.userdata1 = &wait
		}
	);

    while (!wait.done) {
#ifdef __EMSCRIPTEN__
		emscripten_sleep(1); // yields back to the browser event loop
#else
		wgpuDeviceTick(queue->device);
#endif
    }
	GPU::process_pending_code(queue);
}

void gpuSyncMemoryEXT(GpuCommandBuffer* cmd, gpu* mem) {
	endCurrentPass(cmd);
	auto [range, cpu, memory_type] = cmd->queue->allocations[mem];

	switch (memory_type) {
	case MEMORY_DEFAULT:
	case MEMORY_GPU:
	case MEMORY_TEXTURE:
		GPU::push_to_monobuffer(cmd, range, cpu);

	break; case MEMORY_READBACK:
	case MEMORY_TEXTURE_READBACK:
		GPU::pull_from_monobuffer(cmd, range, cpu);
	}
}

void gpuSyncMemoryImmediateEXT(GpuQueue* queue, gpu* mem) {
	auto cmd = gpuStartCommandRecording(queue);
	gpuSyncMemoryEXT(cmd, mem);
	auto submit_index = gpuSubmit(queue, {&cmd, 1});

	GPU::semaphore_wait(queue, queue->current_submission_timeline_semaphore, submit_index);
	GPU::process_pending_code(queue);
}

const GpuSemaphore* gpuGetSubmissionSemaphoreEXT(GpuQueue* queue) {
	return (GpuSemaphore*)&queue->current_submission_timeline_semaphore;
}

GpuSemaphore* gpuCreateSemaphore(GpuQueue* queue, uint64_t initial_value) {
	auto out = (GpuSemaphore*)queue->cpu_allocator(nullptr, sizeof(GpuSemaphore));

	*out = GPU::semaphore_initialize(queue, initial_value);
	return out;
}

uint64_t gpuWaitSemaphore(GpuQueue* queue, const GpuSemaphore* semaphore, uint64_t value, uint64_t timeout /* = UINT64_MAX */) {
	if(value == GPU_GET_VALUE)
		return GPU::semaphore_value(queue, *semaphore);

	GPU::semaphore_wait(queue, *semaphore, value, std::chrono::nanoseconds(timeout));
	GPU::process_pending_code(queue);
	return value;
}

void gpuFreeSemaphore(GpuQueue* queue, GpuSemaphore* semaphore) {
	GPU::semaphore_destroy(*semaphore);
	queue->cpu_allocator(semaphore, 0);
}




namespace GPU::detail {
	// Where a GPU address lives: the allocation holding it, how far into that allocation it sits,
	// and the allocation's base (the key queue->allocations and gpu2textures use). The members are
	// in that order so the existing `auto [range, offset, base]` call sites read unchanged.
	struct BufferLocation {
		GpuQueue::MonobufferRange range = {};
		ptrdiff_t offset = 0;
		gpu* base = nullptr; // Null when the address landed outside every allocation
	};

	// The allocation holding `addr` is the one whose base is the greatest at or below it and whose
	// size actually reaches it — not, as this used to look for, the nearest base at or *above* it,
	// which resolved every interior pointer to the next allocation along and handed back a negative
	// offset. no_offsets promises the address is already a base, which skips the search.
	//
	// An address encodes its monobuffer in its top bits, so ordering addresses as integers orders
	// them by (monobuffer, offset), which is exactly what this walk wants.
	//
	// Nothing here inserts into `allocations`: an address belonging to no allocation comes back
	// null rather than quietly adding an empty entry that the next search would then find.
	inline BufferLocation closest_buffer(GpuQueue* queue, gpu* addr, bool no_offsets) {
		if(no_offsets) {
			auto found = queue->allocations.find(addr);
			assert(found != queue->allocations.end() && "no_offsets promises an address that is an allocation base");
			if(found == queue->allocations.end()) return {};
			return {std::get<GpuQueue::MonobufferRange>(found->second), 0, addr};
		}

		BufferLocation out;
		auto address = (uintptr_t)addr;
		for(const auto& [key, allocation]: queue->allocations) {
			auto base = (uintptr_t)key;
			if(base > address) continue; // Starts past the address, so it can't be holding it

			const auto& range = std::get<GpuQueue::MonobufferRange>(allocation);
			if(address - base >= range.size()) continue; // Ends before it
			if(out.base && base < (uintptr_t)out.base) continue; // Something tighter was found already

			out = {range, ptrdiff_t(address - base), key};
		}
		assert(out.base && "The address doesn't lie inside any allocation");
		return out;
	}

	inline std::array<BufferLocation, 2> closest_buffer(GpuQueue* queue, gpu* addrA, gpu* addrB, bool no_offsets) {
		return {closest_buffer(queue, addrA, no_offsets), closest_buffer(queue, addrB, no_offsets)};
	}
}

namespace GPU::detail {
	// Bytes per texel for the uncompressed, single plane formats the API exposes
	inline uint32_t format_bytes(FORMAT format) {
		switch (format) {
		case FORMAT_R8_UNORM:
		case FORMAT_R8_SNORM:
		case FORMAT_R8_UINT:
		case FORMAT_R8_SINT:
		case FORMAT_S8_UINT: return 1;

		case FORMAT_R16_UNORM:
		case FORMAT_R16_SNORM:
		case FORMAT_R16_UINT:
		case FORMAT_R16_SINT:
		case FORMAT_R16_FLOAT:
		case FORMAT_RG8_UNORM:
		case FORMAT_RG8_SNORM:
		case FORMAT_RG8_UINT:
		case FORMAT_RG8_SINT:
		case FORMAT_D16_UNORM: return 2;

		case FORMAT_R32_UINT:
		case FORMAT_R32_SINT:
		case FORMAT_R32_FLOAT:
		case FORMAT_RG16_UNORM:
		case FORMAT_RG16_SNORM:
		case FORMAT_RG16_UINT:
		case FORMAT_RG16_SINT:
		case FORMAT_RG16_FLOAT:
		case FORMAT_RGBA8_UNORM:
		case FORMAT_RGBA8_SRGB:
		case FORMAT_RGBA8_SNORM:
		case FORMAT_RGBA8_UINT:
		case FORMAT_RGBA8_SINT:
		case FORMAT_BGRA8_UNORM:
		case FORMAT_BGRA8_SRGB:
		case FORMAT_RGB10_A2_UINT:
		case FORMAT_RGB10_A2_UNORM:
		case FORMAT_RG11B10_UFLOAT:
		case FORMAT_RGB9E5_UFLOAT:
		case FORMAT_D24_PLUS_S8_UINT:
		case FORMAT_D32_FLOAT: return 4;

		case FORMAT_RG32_UINT:
		case FORMAT_RG32_SINT:
		case FORMAT_RG32_FLOAT:
		case FORMAT_RGBA16_UNORM:
		case FORMAT_RGBA16_SNORM:
		case FORMAT_RGBA16_UINT:
		case FORMAT_RGBA16_SINT:
		case FORMAT_RGBA16_FLOAT:
		case FORMAT_D32_FLOAT_S8_UINT: return 8;

		case FORMAT_RGBA32_UINT:
		case FORMAT_RGBA32_SINT:
		case FORMAT_RGBA32_FLOAT: return 16;

		// FORMAT_D24_PLUS lands here with FORMAT_NONE: its texels have no size anyone can name,
		// which is also why WebGPU refuses to copy it
		default: return 0;
		}
	}

	// The extent of the monotexture slot a texture sits in, at one mip level. A monotexture is
	// sized by the first texture to land in its bucket and the bucket keys on power of two rounded
	// dimensions, so a later, smaller texture only covers the top left corner of its slot. Both
	// round up to the same power of two, so a slot is never as much as twice the image.
	inline uvec2 monotexture_slot_extent(const GpuTexture* texture, uint32_t mip) {
		return {
			std::max(wgpuTextureGetWidth(texture->texture) >> mip, 1u),
			std::max(wgpuTextureGetHeight(texture->texture) >> mip, 1u)
		};
	}

}

void gpuMemCpy(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, size_t bytes, bool no_offsets /* = false */) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_, no_offsets);
	auto [dest_range, dest_offset, dest_addr] = dest; auto [src_range, src_offset, src_addr] = src;

	if(src_range.buffer == dest_range.buffer) {
		// We can't copy from a buffer to itself for some reason... so we need to create a temporary transfer buffer.
		WGPUBufferDescriptor d {
			.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
			.size = bytes
		};
		auto tmp = wgpuDeviceCreateBuffer(cmd->queue->device, &d);

		wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, cmd->queue->monobuffers[src_range.buffer], src_range.start + src_offset, tmp, 0, bytes);
		wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, tmp, 0, cmd->queue->monobuffers[dest_range.buffer], dest_range.start + dest_offset, bytes);

		cmd->queue->code_pending_submission_finished.emplace_back([tmp](){ wgpuBufferRelease(tmp); }, cmd->queue->next_submission_index);
	} else wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, cmd->queue->monobuffers[src_range.buffer], src_range.start + src_offset, cmd->queue->monobuffers[dest_range.buffer], dest_range.start + dest_offset, bytes);
}

// TODO: Untested!
void gpuCopyToTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, GpuTexture* texture, bool no_offsets /* = false */) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_, no_offsets);
	auto [dest_range, dest_offset, dest_addr] = dest; auto [src_range, src_offset, src_addr] = src;

	assert(cmd->queue->gpu2textures[dest_addr] == texture);

	auto base = static_cast<uint64_t>(src_range.start + src_offset);
	WGPUTexelCopyBufferInfo source {
		.layout = {
			.offset = base,
			.bytesPerRow = texture->descriptor.dimensions.x * GPU::detail::format_bytes(texture->descriptor.format),
			.rowsPerImage = texture->descriptor.dimensions.y,
		},
		.buffer = cmd->queue->monobuffers[src_range.buffer],
	};
	WGPUTexelCopyTextureInfo destination = {
		.texture = texture->texture,
		.mipLevel = 0, // TODO: How do we allow more control over this?
		.origin = {0, 0, 0},
		.aspect = WGPUTextureAspect_All
	};
	if(texture->range) destination.origin = {0, 0, texture->range->start};
	WGPUExtent3D size {
		.width = texture->descriptor.dimensions.x,
		.height = texture->descriptor.dimensions.y,
		.depthOrArrayLayers = texture->range ? texture->range->end - texture->range->start : texture->descriptor.dimensions.z
	};
	wgpuCommandEncoderCopyBufferToTexture(cmd->encoder, &source, &destination, &size);

	// Nothing fills the rest of the monotexture slot this texture was handed. It used to be
	// padded here with the image's edge replicated across it, at a cost of one copy command per
	// column of the gap, because a coordinate running past the image would otherwise sample
	// undefined data. Nothing reads it any more: a sample is held inside the image by the uvMax in
	// the heap entry (see gpuTextureViewDescriptor), and gpuBlitTextureEXT -- which is what writes
	// the smaller mips -- pins its own source read the same way.
}

// TODO: Untested!
void gpuCopyFromTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, const GpuTexture* texture, bool no_offsets /* = false */) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_, no_offsets);
	auto [dest_range, dest_offset, dest_addr] = dest; auto [src_range, src_offset, src_addr] = src;

	assert(cmd->queue->gpu2textures[src_addr] == texture);

	WGPUTexelCopyTextureInfo source = {
		.texture = texture->texture,
		.mipLevel = 0, // TODO: How do we allow more control over this?
		.origin = {0, 0, 0},
		.aspect = WGPUTextureAspect_All
	};
	if(texture->range) source.origin = {0, 0, texture->range->start};
	// The texture is what is being read here, so the buffer half of the copy is the destination
	WGPUTexelCopyBufferInfo destination {
		.layout = {
			.offset = static_cast<uint64_t>(dest_range.start + dest_offset),
			.bytesPerRow = texture->descriptor.dimensions.x * GPU::detail::format_bytes(texture->descriptor.format),
			.rowsPerImage = texture->descriptor.dimensions.y,
		},
		.buffer = cmd->queue->monobuffers[dest_range.buffer],
	};
	WGPUExtent3D size {
		.width = texture->descriptor.dimensions.x,
		.height = texture->descriptor.dimensions.y,
		.depthOrArrayLayers = texture->range ? texture->range->end - texture->range->start : texture->descriptor.dimensions.z
	};
	wgpuCommandEncoderCopyTextureToBuffer(cmd->encoder, &source, &destination, &size);
}



void gpuSetActiveTextureHeapPtr(GpuCommandBuffer* cmd, gpu* texture_heap, bool no_offsets /* = false */) {
	// TODO: Should we add some validation to check that what is provided is a valid texture heap?
	auto [dest_range, dest_offset, dest_addr] = GPU::detail::closest_buffer(cmd->queue, texture_heap, no_offsets);
	dest_range.start += dest_offset;
	// dest_range.end -= dest_offset;

	// The heap lives in a monobuffer, and group 0 binds every monobuffer as writable storage for
	// compute. WebGPU refuses to have one buffer bound writable and read only in the same pass, so
	// binding the heap out of the monobuffer directly would fail validation on the first dispatch.
	//
	// Copying instead of keeping a shadow buffer around: the heap is written through a raw mapped
	// pointer, so there is no way to tell whether it changed since last time and a persistent buffer
	// would have to re-copy on every activation anyway. A copy recorded here is also ordered in the
	// command stream, so it picks up whatever the commands before it wrote into the heap, and two
	// command buffers in flight with different heaps each get their own.
	endCurrentPass(cmd); // A copy can't be recorded with a pass open

	auto size = dest_range.size();
	WGPUBufferDescriptor d {
		.label = {"NoAPI Texture Heap", WGPU_STRLEN},
		.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
		.size = size,
	};
	auto snapshot = wgpuDeviceCreateBuffer(cmd->queue->device, &d);
	wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, cmd->queue->monobuffers[dest_range.buffer], dest_range.start, snapshot, 0, size);
	cmd->queue->code_pending_submission_finished.emplace_back([snapshot](){ wgpuBufferRelease(snapshot); }, cmd->queue->next_submission_index);

	cmd->active_texture_heap = snapshot;
	cmd->active_texture_heap_size = size;
}



void gpuSetEnabledSamplersEXT(GpuCommandBuffer* cmd, GpuSamplerDescSpan enabled_samplers) {
	// Nothing is recorded into the encoder: the set only decides which group 3 (and which copy of the
	// metadata buffer in group 0) the next dispatch or draw binds
	cmd->sampler_set = GPU::detail::ensure_sampler_set(cmd->queue, enabled_samplers);
}

void gpuBarrier(GpuCommandBuffer* cmd, STAGE before, STAGE after, HAZARD_FLAGS hazards /* = (HAZARD_FLAGS)0 */) {
	// TODO: I believe webgpu handles these internally... is there any reason to not make them noops?
}
void gpuSignalAfter(GpuCommandBuffer* cmd, STAGE before, gpu* ptr, uint64_t value, SIGNAL signal) {
	// TODO: I believe webgpu handles these internally... is there any reason to not make them noops?
}
void gpuWaitBefore(GpuCommandBuffer* cmd, STAGE after, gpu* ptr, uint64_t value, OP op, HAZARD_FLAGS hazards /* = (HAZARD_FLAGS)0 */, uint64_t mask /* = ~uint64_t(0) */) {
	// TODO: I believe webgpu handles these internally... is there any reason to not make them noops?
}



void gpuSetPipeline(GpuCommandBuffer* cmd, const GpuPipeline* pipeline) {
	cmd->bound_pipeline = pipeline;

	if(std::holds_alternative<GpuPipeline::ComputeCache>(pipeline->cache)) {
		if(pipeline->reference_layout != cmd->queue->current_compute_pipeline_layout)
			update_compute_pipeline(cmd->queue, pipeline);

		endRenderPass(cmd);
		if(!cmd->compute_pass) cmd->compute_pass = wgpuCommandEncoderBeginComputePass(cmd->encoder, nullptr);

		wgpuComputePassEncoderSetPipeline(cmd->compute_pass, std::get<GpuPipeline::ComputeCache>(pipeline->cache).pipeline);
	} else {
		endComputePass(cmd);

		// A render pipeline can't be built (let alone set) until we know what is being drawn with it,
		// so the variant matching the bound state is compiled and set by the draw commands instead.
	}
}

// Makes sure the cached variant of the bound graphics pipeline matches the state it is about to be
// drawn with, rebuilding it when the bound elements or the pipeline layout have moved on, and then
// sets it on the render pass
void bindRenderPipeline(GpuCommandBuffer* cmd) {
	assert(cmd->render_pass && "A render pass must be started before drawing");
	assert(cmd->bound_pipeline && "A graphics pipeline must be set before drawing");
	assert(std::holds_alternative<GpuPipeline::RenderCache>(cmd->bound_pipeline->cache) && "A compute pipeline can't be drawn with");

	auto pipeline = cmd->bound_pipeline;
	auto& cache = std::get<GpuPipeline::RenderCache>(pipeline->cache);

	if(!cache.pipeline
	|| pipeline->reference_layout != cmd->queue->current_graphics_pipeline_layout
	|| !GPU::same(cache.depth_stencil, cmd->depth_stencil)
	|| !GPU::same(cache.blend, cmd->blend)
	// The index format is only baked into the pipeline when primitives are restarted by the indices
	|| (cache.descriptor.topology == TOPOLOGY_TRIANGLE_STRIP && cache.index_type != cmd->index_type))
		update_render_pipeline(cmd->queue, pipeline, cmd->depth_stencil, cmd->blend, cmd->index_type);

	if(cmd->bound_render_pipeline != cache.pipeline) {
		wgpuRenderPassEncoderSetPipeline(cmd->render_pass, cache.pipeline);
		cmd->bound_render_pipeline = cache.pipeline;
	}

	// The stencil reference is the one piece of depth/stencil state WebGPU leaves dynamic
	if(cmd->depth_stencil)
		wgpuRenderPassEncoderSetStencilReference(cmd->render_pass, cmd->depth_stencil->stencilFront.reference);
}

// The field offsets here are the ones the generated GpuShaderData lands on (0, 8 and 16 --
// Slang lays a uniform block out std140 style, where a uint2 sits on an 8 byte boundary). The
// trailing padding is not: a WGSL uniform block's *size* is rounded up to a multiple of 16, and
// the binding is rejected outright if the buffer behind it is smaller than that rounded size,
// so each of these is padded out to what the shader side actually binds as.
struct ComputeShaderData {
	gpu* compute;
	uint64_t _padding = 0;
};

struct GraphicsShaderData {
	gpu* vertex;
	gpu* fragment;
	gpu* indices;
	uint64_t _padding = 0;
};

// Builds group 0: every monobuffer, the active texture heap, and a throwaway uniform holding the
// root data pointer(s) the shaders were invoked with
WGPUBindGroup createBufferBindGroup(GpuCommandBuffer* cmd, const void* shader_data, size_t shader_data_size, bool compute) {
	uint32_t binding = 0;
	std::vector<WGPUBindGroupEntry> group0;
	for(uint32_t i = 0; i < cmd->queue->monobuffers.size(); ++i)
		group0.push_back({
			.binding = binding++,
			.buffer = cmd->queue->monobuffers[i],
			.offset = 0,
			.size = cmd->queue->monobuffer_sizes[i]
		});

	// If there is no active texture heap bind the empty buffer
	if(cmd->active_texture_heap)
		group0.push_back({
			.binding = binding++,
			.buffer = cmd->active_texture_heap,
			.offset = 0,
			.size = cmd->active_texture_heap_size
		});
	else group0.push_back({
		.binding = binding++,
		.buffer = cmd->queue->empty_buffer,
		.offset = 0,
		.size = 4
	});

	{
		auto& sampler_set = cmd->sampler_set ? *cmd->sampler_set : *cmd->queue->default_sampler_set;
		group0.push_back({
			.binding = binding++,
			.buffer = sampler_set.lookup_buffer,
			.offset = 0,
			.size = GPU::detail::sampler_lookup_size * sizeof(uint32_t)
		});
	}

	{
		WGPUBufferDescriptor d {
			.label = {"NoAPI ShaderData", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Uniform,
			.size = shader_data_size,
			.mappedAtCreation = true,
		};
		auto tmp = wgpuDeviceCreateBuffer(cmd->queue->device, &d);
		cmd->queue->code_pending_submission_finished.emplace_back([tmp] { wgpuBufferRelease(tmp); }, cmd->queue->next_submission_index);
		memcpy(wgpuBufferGetMappedRange(tmp, 0, d.size), shader_data, shader_data_size);
		wgpuBufferUnmap(tmp);

		group0.push_back({
			.binding = binding++,
			.buffer = tmp,
			.offset = 0,
			.size = d.size
		});
	}

	WGPUBindGroupDescriptor d {
		.layout = compute ? cmd->queue->current_compute_bind_group_layout0 : cmd->queue->current_graphics_bind_group_layout0,
		.entryCount = group0.size(),
		.entries = group0.data()
	};
	return wgpuDeviceCreateBindGroup(cmd->queue->device, &d);
}

WGPUBindGroup createBufferBindGroup(GpuCommandBuffer* cmd, gpu* data, bool no_offsets) {
	ComputeShaderData shader_data { .compute = data };
	return createBufferBindGroup(cmd, &shader_data, sizeof(shader_data), true);
}

// Binds every group needed by a dispatch or a draw (the texture groups are shared between the two)
void bindGroups(GpuCommandBuffer* cmd, const void* shader_data, size_t shader_data_size, bool compute) {
	auto group0 = createBufferBindGroup(cmd, shader_data, shader_data_size, compute);
	cmd->queue->code_pending_submission_finished.emplace_back([group0] { wgpuBindGroupRelease(group0); }, cmd->queue->next_submission_index);

	auto group3 = (cmd->sampler_set ? cmd->sampler_set : cmd->queue->default_sampler_set)->bind_group;

	if(compute) {
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 0, group0, 0, nullptr);
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 1, cmd->queue->current_bind_group1, 0, nullptr);
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 2, cmd->queue->current_bind_group2, 0, nullptr);
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 3, group3, 0, nullptr);
	} else {
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 0, group0, 0, nullptr);
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 1, cmd->queue->current_bind_group1, 0, nullptr);
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 2, cmd->queue->current_bind_group2, 0, nullptr);
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 3, group3, 0, nullptr);
	}
}

void bindComputeGroups(GpuCommandBuffer* cmd, gpu* data) {
	ComputeShaderData shader_data { .compute = data };
	bindGroups(cmd, &shader_data, sizeof(shader_data), true);
}

void bindGraphicsGroups(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* index_data) {
	GraphicsShaderData shader_data { .vertex = vertex_data, .fragment = fragment_data, .indices = index_data };
	bindGroups(cmd, &shader_data, sizeof(shader_data), false);
}

void gpuDispatch(GpuCommandBuffer* cmd, gpu* data, uvec3 grid_dimensions, bool no_offsets /* = false */) {
	assert(cmd->compute_pass);

	bindComputeGroups(cmd, data);

	wgpuComputePassEncoderDispatchWorkgroups(cmd->compute_pass, grid_dimensions.x, grid_dimensions.y, grid_dimensions.z);
}
void gpuDispatchIndirect(GpuCommandBuffer* cmd, gpu* data, gpu* grid_dimensions_gpu, bool no_offsets /* = false */) {
	assert(cmd->compute_pass);
	assert(grid_dimensions_gpu);

	auto [grid_range, grid_offset, grid_addr] = GPU::detail::closest_buffer(cmd->queue, grid_dimensions_gpu, no_offsets);

	bindComputeGroups(cmd, data);

	wgpuComputePassEncoderDispatchWorkgroupsIndirect(cmd->compute_pass, cmd->queue->monobuffers[grid_range.buffer], grid_range.start + grid_offset);
}


void gpuSetDepthStencilState(GpuCommandBuffer* cmd, const GpuDepthStencilState* state) {
	// WebGPU bakes everything except the stencil reference into the pipeline, so all we can do here
	// is record the state; bindRenderPipeline rebuilds the bound pipeline when it no longer matches
	cmd->depth_stencil = state ? std::optional{state->descriptor} : std::nullopt;

	if(cmd->render_pass && cmd->depth_stencil)
		wgpuRenderPassEncoderSetStencilReference(cmd->render_pass, cmd->depth_stencil->stencilFront.reference);
}

void gpuSetBlendState(GpuCommandBuffer* cmd, const GpuBlendState* state) {
	// Blending is part of a WebGPU PSO, so this is recorded and applied by the next draw
	cmd->blend = state ? std::optional{state->descriptor} : std::nullopt;
}

void gpuSetViewportEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin /* = {0, 0} */, float depth_min /* = 0 */, float depth_max /* = 1 */) {
	assert(cmd->render_pass && "The viewport can only be set inside a render pass");
	wgpuRenderPassEncoderSetViewport(cmd->render_pass, origin.x, origin.y, extent.x, extent.y, depth_min, depth_max);
}

void gpuSetScissorRectEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin /* = {0, 0} */) {
	assert(cmd->render_pass && "The scissor rectangle can only be set inside a render pass");
	wgpuRenderPassEncoderSetScissorRect(cmd->render_pass, origin.x, origin.y, extent.x, extent.y);
}



namespace GPU::detail {
	// Creates the single subresource view a texture is attached to a render pass through. The view
	// lives until the submission referencing it has finished.
	inline WGPUTextureView attachment_view(GpuCommandBuffer* cmd, const GpuTexture* texture, uint32_t mipLevel, uint32_t slice) {
		bool volume = texture->descriptor.type == TEXTURE_3D;

		WGPUTextureViewDescriptor d {
			.label = {"NoAPI Attachment", WGPU_STRLEN},
			.format = GPU::format2wgpu(texture->descriptor.format),
			.dimension = volume ? WGPUTextureViewDimension_3D : WGPUTextureViewDimension_2D,
			.baseMipLevel = mipLevel,
			.mipLevelCount = 1,
			// Textures backed by a monotexture live at an offset into its array layers
			.baseArrayLayer = volume ? 0 : (texture->range ? texture->range->start : 0) + slice,
			.arrayLayerCount = 1,
			.aspect = WGPUTextureAspect_All,
			.usage = WGPUTextureUsage_RenderAttachment,
		};
		auto view = wgpuTextureCreateView(texture->texture, &d);

		cmd->queue->code_pending_submission_finished.emplace_back([view] {
			wgpuTextureViewRelease(view);
		}, cmd->queue->next_submission_index);
		return view;
	}

	inline uvec2 attachment_extent(const GpuTexture* texture, uint32_t mipLevel) {
		return {
			std::max(texture->descriptor.dimensions.x >> mipLevel, 1u),
			std::max(texture->descriptor.dimensions.y >> mipLevel, 1u)
		};
	}
}

void gpuBeginRenderPass(GpuCommandBuffer* cmd, const GpuRenderPassDesc& desc) {
	endCurrentPass(cmd);

	uvec2 extent = {0, 0};

	std::vector<WGPURenderPassColorAttachment> colors; colors.reserve(desc.colorAttachments.size());
	for(auto& color: desc.colorAttachments) {
		bool volume = color.texture->descriptor.type == TEXTURE_3D;
		colors.push_back({
			.view = GPU::detail::attachment_view(cmd, color.texture, color.mipLevel, color.slice),
			// Only a view into a 3D texture picks the rendered slice here, everything else selects it as an array layer
			.depthSlice = volume ? color.slice : WGPU_DEPTH_SLICE_UNDEFINED,
			.resolveTarget = color.resolveTexture ? GPU::detail::attachment_view(cmd, color.resolveTexture, color.mipLevel, color.slice) : nullptr,
			.loadOp = GPU::load2wgpu(color.loadOp),
			.storeOp = GPU::store2wgpu(color.storeOp),
			.clearValue = {color.clearValue.r, color.clearValue.g, color.clearValue.b, color.clearValue.a},
		});

		if(extent.x == 0) extent = GPU::detail::attachment_extent(color.texture, color.mipLevel);
	}

	// Depth and stencil are separate attachments in our API but WebGPU merges them into one, so when
	// both are provided they have to name the same subresource
	auto& merged = desc.depthAttachment ? desc.depthAttachment : desc.stencilAttachment;
	assert((!(desc.depthAttachment && desc.stencilAttachment)
		|| (desc.depthAttachment->texture == desc.stencilAttachment->texture
			&& desc.depthAttachment->mipLevel == desc.stencilAttachment->mipLevel
			&& desc.depthAttachment->slice == desc.stencilAttachment->slice))
		&& "WebGPU requires the depth and stencil attachments to be the same subresource");

	WGPURenderPassDepthStencilAttachment depth_stencil {};
	if(merged) {
		auto format = merged->texture->descriptor.format;
		depth_stencil.view = GPU::detail::attachment_view(cmd, merged->texture, merged->mipLevel, merged->slice);

		// Both aspects of the format need ops, whether or not the user described that aspect
		if(gpuFormatIsDepthEXT(format)) {
			auto& attachment = desc.depthAttachment ? *desc.depthAttachment : *merged;
			depth_stencil.depthLoadOp = desc.depthAttachment ? GPU::load2wgpu(attachment.loadOp) : WGPULoadOp_Load;
			depth_stencil.depthStoreOp = desc.depthAttachment ? GPU::store2wgpu(attachment.storeOp) : WGPUStoreOp_Store;
			depth_stencil.depthClearValue = static_cast<float>(attachment.clearValue);
		}
		if(gpuFormatIsStencilEXT(format)) {
			auto& attachment = desc.stencilAttachment ? *desc.stencilAttachment : *merged;
			depth_stencil.stencilLoadOp = desc.stencilAttachment ? GPU::load2wgpu(attachment.loadOp) : WGPULoadOp_Load;
			depth_stencil.stencilStoreOp = desc.stencilAttachment ? GPU::store2wgpu(attachment.storeOp) : WGPUStoreOp_Store;
			depth_stencil.stencilClearValue = static_cast<uint32_t>(attachment.clearValue);
		}

		if(extent.x == 0) extent = GPU::detail::attachment_extent(merged->texture, merged->mipLevel);
	}

	WGPURenderPassDescriptor d {
		.label = {"NoAPI Render Pass", WGPU_STRLEN},
		.colorAttachmentCount = colors.size(),
		.colorAttachments = colors.data(),
		.depthStencilAttachment = merged ? &depth_stencil : nullptr,
	};
	cmd->render_pass = wgpuCommandEncoderBeginRenderPass(cmd->encoder, &d);

	gpuSetViewportEXT(cmd, extent);
	gpuSetScissorRectEXT(cmd, extent);
}

void gpuEndRenderPass(GpuCommandBuffer* cmd, GpuOptionalRenderPassDesc desc /* = {} */) {
	// The descriptor is only needed by backends that transition images by hand, WebGPU tracks that itself
	endRenderPass(cmd);
}



namespace GPU::detail {
	// The blit's fullscreen triangle. Both bind group layout flavors compile this same module; they
	// only differ in the sampler and texture types they declare, which the WGSL never names.
	constexpr static std::string_view BLIT_WGSL_CODE = R"WGSL(
@group(0) @binding(0) var blit_source : texture_2d<f32>;
@group(0) @binding(1) var blit_sampler : sampler;
// xy = the viewport's size over the destination image's, times the source image's share of the
// monotexture holding it, so that uv reaches the source image's edge exactly where the destination
// image ends even when the viewport was opened up to cover the rest of a monotexture slot.
// zw = that edge, half a texel in. The fragments past it are pinned there, which is what fills the
// gap around the destination image by replicating the source's edge instead of resampling whatever
// the source's own slot holds beyond it.
@group(0) @binding(2) var<uniform> blit_uv_scale : vec4<f32>;

struct Varyings {
	@builtin(position) position : vec4<f32>,
	@location(0) uv : vec2<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) index : u32) -> Varyings {
	// An oversized triangle covering the whole viewport, its uvs run 0..1 across the covered area
	let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
	return Varyings(
		// WebGPU's clip space puts +Y at the top of the viewport, but v = 0 is the top texel row
		vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0),
		uv * blit_uv_scale.xy
	);
}

@fragment
fn fragment(varyings : Varyings) -> @location(0) vec4<f32> {
	// The view covers a single mip, so there is never another level to pick between
	return textureSampleLevel(blit_source, blit_sampler, min(varyings.uv, blit_uv_scale.zw), 0.0);
}
)WGSL";

	inline void ensure_blit_layouts(GpuQueue* queue, bool filtering) {
		if(queue->blit_bind_group_layouts[filtering]) return;

		std::array<WGPUBindGroupLayoutEntry, 3> entries = {
			WGPUBindGroupLayoutEntry{
				.binding = 0,
				.visibility = WGPUShaderStage_Fragment,
				.texture = {
					.sampleType = filtering ? WGPUTextureSampleType_Float : WGPUTextureSampleType_UnfilterableFloat,
					.viewDimension = WGPUTextureViewDimension_2D,
					.multisampled = false,
				}
			}, WGPUBindGroupLayoutEntry{
				.binding = 1,
				.visibility = WGPUShaderStage_Fragment,
				.sampler = {
					.type = filtering ? WGPUSamplerBindingType_Filtering : WGPUSamplerBindingType_NonFiltering,
				}
			}, WGPUBindGroupLayoutEntry{
				.binding = 2,
				// Both stages: the vertex half reads the scale, the fragment half reads the limit
				// it pins the source coordinate to
				.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
				.buffer = {
					.type = WGPUBufferBindingType_Uniform,
					.minBindingSize = sizeof(float) * 4,
				}
			}
		};

		WGPUBindGroupLayoutDescriptor layout {
			.label = {"NoAPI Blit", WGPU_STRLEN},
			.entryCount = entries.size(),
			.entries = entries.data(),
		};
		queue->blit_bind_group_layouts[filtering] = wgpuDeviceCreateBindGroupLayout(queue->device, &layout);

		WGPUPipelineLayoutDescriptor pipeline_layout {
			.label = {"NoAPI Blit", WGPU_STRLEN},
			.bindGroupLayoutCount = 1,
			.bindGroupLayouts = &queue->blit_bind_group_layouts[filtering],
		};
		queue->blit_pipeline_layouts[filtering] = wgpuDeviceCreatePipelineLayout(queue->device, &pipeline_layout);

		WGPUSamplerDescriptor sampler {
			.label = {"NoAPI Blit", WGPU_STRLEN},
			.addressModeU = WGPUAddressMode_ClampToEdge,
			.addressModeV = WGPUAddressMode_ClampToEdge,
			.addressModeW = WGPUAddressMode_ClampToEdge,
			.magFilter = filtering ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest,
			.minFilter = filtering ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest,
			.mipmapFilter = WGPUMipmapFilterMode_Nearest,
			.lodMinClamp = 0,
			.lodMaxClamp = 1,
			.maxAnisotropy = 1,
		};
		queue->blit_samplers[filtering] = wgpuDeviceCreateSampler(queue->device, &sampler);
	}

	// The destination format is baked into a WebGPU PSO and the sampler type into its layout, so one
	// pipeline is kept per (format, filtering) pair for the lifetime of the queue
	inline WGPURenderPipeline blit_pipeline(GpuQueue* queue, WGPUTextureFormat format, bool filtering) {
		auto key = static_cast<uint64_t>(format) << 1 | filtering;
		if(auto found = queue->blit_pipelines.find(key); found != queue->blit_pipelines.end())
			return found->second;

		ensure_blit_layouts(queue, filtering);

		if(!queue->blit_shader) {
			WGPUShaderSourceWGSL source {
				.chain = {.sType = WGPUSType_ShaderSourceWGSL},
				.code = {BLIT_WGSL_CODE.data(), BLIT_WGSL_CODE.size()}
			};
			WGPUShaderModuleDescriptor module {
				.nextInChain = &source.chain,
				.label = {"NoAPI Blit", WGPU_STRLEN},
			};
			queue->blit_shader = wgpuDeviceCreateShaderModule(queue->device, &module);
		}

		WGPUColorTargetState target {
			.format = format,
			.writeMask = WGPUColorWriteMask_All,
		};
		WGPUFragmentState fragment {
			.module = queue->blit_shader,
			.entryPoint = {"fragment", WGPU_STRLEN},
			.targetCount = 1,
			.targets = &target,
		};
		WGPURenderPipelineDescriptor d {
			.label = {"NoAPI Blit", WGPU_STRLEN},
			.layout = queue->blit_pipeline_layouts[filtering],
			.vertex = {
				.module = queue->blit_shader,
				.entryPoint = {"vertex", WGPU_STRLEN},
			},
			.primitive = {
				.topology = WGPUPrimitiveTopology_TriangleList,
				.stripIndexFormat = WGPUIndexFormat_Undefined,
				.frontFace = WGPUFrontFace_CCW,
				.cullMode = WGPUCullMode_None,
			},
			.multisample = {
				.count = 1,
				.mask = 0xFFFFFFFF,
			},
			.fragment = &fragment,
		};

		auto pipeline = wgpuDeviceCreateRenderPipeline(queue->device, &d);
		queue->blit_pipelines[key] = pipeline;
		return pipeline;
	}

}

void gpuBlitTextureEXT(GpuCommandBuffer* cmd, GpuTexture* destination, const GpuTexture* source,
		bool linear_filter /* = true */,
		uint32_t destination_mip /* = 0 */, uint32_t destination_slice /* = 0 */,
		uint32_t source_mip /* = 0 */, uint32_t source_slice /* = 0 */) {
	assert(!cmd->render_pass && "A blit opens a render pass of its own, so it can't be recorded inside another one");
	assert((source->descriptor.usage & USAGE_SAMPLED) && "The blit source must have been created with USAGE_SAMPLED");
	assert((destination->descriptor.usage & USAGE_COLOR_ATTACHMENT) && "The blit destination must have been created with USAGE_COLOR_ATTACHMENT");
	assert(!gpuFormatIsDepthStencilEXT(source->descriptor.format) && !gpuFormatIsDepthStencilEXT(destination->descriptor.format) && "Depth/stencil textures can't be blitted");
	assert(source->descriptor.type != TEXTURE_3D && "A slice of a 3D texture can't be bound on its own, blit out of a 2D array instead");
	assert(source_mip < source->descriptor.mipCount && destination_mip < destination->descriptor.mipCount);

	endComputePass(cmd);

	// The source is only ever read through a single mip of a single layer, which is what lets the
	// shader stay a plain texture_2d no matter what the texture it came from actually is
	WGPUTextureViewDescriptor view {
		.label = {"NoAPI Blit Source", WGPU_STRLEN},
		.format = GPU::format2wgpu(source->descriptor.format),
		.dimension = WGPUTextureViewDimension_2D,
		.baseMipLevel = source_mip,
		.mipLevelCount = 1,
		// Textures backed by a monotexture live at an offset into its array layers
		.baseArrayLayer = (source->range ? source->range->start : 0) + source_slice,
		.arrayLayerCount = 1,
		.aspect = WGPUTextureAspect_All,
		.usage = WGPUTextureUsage_TextureBinding,
	};
	auto source_view = wgpuTextureCreateView(source->texture, &view);
	cmd->queue->code_pending_submission_finished.emplace_back([source_view] {
		wgpuTextureViewRelease(source_view);
	}, cmd->queue->next_submission_index);

	// Building the pipeline is what brings the matching layout and sampler into existence
	bool filtering = linear_filter && gpuFormatIsFilterableEXT(source->descriptor.format);
	auto pipeline = GPU::detail::blit_pipeline(cmd->queue, GPU::format2wgpu(destination->descriptor.format), filtering);

	// The viewport is opened up to the whole monotexture slot so that fragments exist past the
	// destination image, and the uv scale puts the source image's edge exactly where the
	// destination image ends. Those outer fragments read past it and are pinned to it, which
	// replicates the source's edge texels into the gap, covering it in the same draw.
	//
	// Both sides need their own slot ratio. The source view covers a whole monotexture mip, not
	// just the image inside it, so 1.0 in view coordinates is the slot's edge; scaling by the
	// destination's ratio alone -- which is what this did before -- resampled the source's
	// leftovers into the destination image whenever either texture failed to fill its slot.
	auto extent = GPU::detail::attachment_extent(destination, destination_mip);
	auto slot = GPU::detail::monotexture_slot_extent(destination, destination_mip);
	auto source_extent = GPU::detail::attachment_extent(source, source_mip);
	auto source_slot = GPU::detail::monotexture_slot_extent(source, source_mip);

	float source_fraction[2] = {
		static_cast<float>(source_extent.x) / source_slot.x,
		static_cast<float>(source_extent.y) / source_slot.y,
	};
	float uv_scale[4] = {
		source_fraction[0] * slot.x / extent.x,
		source_fraction[1] * slot.y / extent.y,
		source_fraction[0] - 0.5f / source_slot.x,
		source_fraction[1] - 0.5f / source_slot.y,
	};
	WGPUBufferDescriptor scale_desc {
		.label = {"NoAPI Blit UV Scale", WGPU_STRLEN},
		.usage = WGPUBufferUsage_Uniform,
		.size = sizeof(uv_scale),
		.mappedAtCreation = true,
	};
	auto scale_buffer = wgpuDeviceCreateBuffer(cmd->queue->device, &scale_desc);
	memcpy(wgpuBufferGetMappedRange(scale_buffer, 0, scale_desc.size), uv_scale, sizeof(uv_scale));
	wgpuBufferUnmap(scale_buffer);
	cmd->queue->code_pending_submission_finished.emplace_back([scale_buffer] {
		wgpuBufferRelease(scale_buffer);
	}, cmd->queue->next_submission_index);

	std::array<WGPUBindGroupEntry, 3> entries = {
		WGPUBindGroupEntry{
			.binding = 0,
			.textureView = source_view,
		}, WGPUBindGroupEntry{
			.binding = 1,
			.sampler = cmd->queue->blit_samplers[filtering],
		}, WGPUBindGroupEntry{
			.binding = 2,
			.buffer = scale_buffer,
			.offset = 0,
			.size = sizeof(uv_scale),
		}
	};
	WGPUBindGroupDescriptor group_desc {
		.label = {"NoAPI Blit", WGPU_STRLEN},
		.layout = cmd->queue->blit_bind_group_layouts[filtering],
		.entryCount = entries.size(),
		.entries = entries.data(),
	};
	auto group = wgpuDeviceCreateBindGroup(cmd->queue->device, &group_desc);
	cmd->queue->code_pending_submission_finished.emplace_back([group] {
		wgpuBindGroupRelease(group);
	}, cmd->queue->next_submission_index);

	WGPURenderPassColorAttachment attachment {
		.view = GPU::detail::attachment_view(cmd, destination, destination_mip, destination_slice),
		// Only a view into a 3D texture picks the written slice here, everything else selected it as an array layer
		.depthSlice = destination->descriptor.type == TEXTURE_3D ? destination_slice : WGPU_DEPTH_SLICE_UNDEFINED,
		// The draw covers the whole mip, so there is never anything worth loading back in first
		.loadOp = WGPULoadOp_Clear,
		.storeOp = WGPUStoreOp_Store,
		.clearValue = {0, 0, 0, 0},
	};
	WGPURenderPassDescriptor pass_desc {
		.label = {"NoAPI Blit", WGPU_STRLEN},
		.colorAttachmentCount = 1,
		.colorAttachments = &attachment,
	};

	// A pass of its own rather than cmd->render_pass, so nothing the command buffer had bound for
	// the user's rendering is disturbed (a WebGPU pass owns every piece of state set on it)
	auto pass = wgpuCommandEncoderBeginRenderPass(cmd->encoder, &pass_desc);
	wgpuRenderPassEncoderSetPipeline(pass, pipeline);
	wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
	wgpuRenderPassEncoderSetViewport(pass, 0, 0, slot.x, slot.y, 0, 1);
	wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, slot.x, slot.y);
	wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
	wgpuRenderPassEncoderEnd(pass);
	wgpuRenderPassEncoderRelease(pass);
}



// Matches both VkDrawIndexedIndirectCommand and WebGPU's indexed indirect argument layout
struct GpuDrawIndexedIndirectCommand {
	uint32_t indexCount;
	uint32_t instanceCount;
	uint32_t firstIndex;
	int32_t baseVertex;
	uint32_t firstInstance;
};

namespace GPU::detail {
	// The monobuffers are created with Index usage, so the indices are drawn straight out of the one
	// they were allocated in; no shadow copy (and thus nothing for no_index_buffer_changes to skip)
	inline void bind_index_buffer(GpuCommandBuffer* cmd, gpu* indices, INDEX_TYPE_EXT index_type, bool no_offsets) {
		auto [range, offset, address] = GPU::detail::closest_buffer(cmd->queue, indices, no_offsets);
		assert(range.start + offset < range.end && "The indices point past the end of their allocation");

		wgpuRenderPassEncoderSetIndexBuffer(cmd->render_pass, cmd->queue->monobuffers[range.buffer],
			GPU::index2wgpu(index_type), range.start + offset, range.end - (range.start + offset));
	}
}

void gpuDrawIndexedInstanced(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, uint32_t index_count, uint32_t instance_count, INDEX_TYPE_EXT index_type /* = INDEX_TYPE_UINT32 */, bool no_offsets /* = false */, bool no_index_buffer_changes /* = false */) {
	assert(cmd->render_pass);
	assert(indices);

	cmd->index_type = index_type;
	bindRenderPipeline(cmd);
	bindGraphicsGroups(cmd, vertex_data, fragment_data, indices);
	GPU::detail::bind_index_buffer(cmd, indices, index_type, no_offsets);

	wgpuRenderPassEncoderDrawIndexed(cmd->render_pass, index_count, instance_count, 0, 0, 0);
}

void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, gpu* args, INDEX_TYPE_EXT index_type /* = INDEX_TYPE_UINT32 */, bool no_offsets /* = false */, bool no_index_buffer_changes /* = false */) {
	assert(cmd->render_pass);
	assert(indices);
	assert(args);

	cmd->index_type = index_type;
	bindRenderPipeline(cmd);
	bindGraphicsGroups(cmd, vertex_data, fragment_data, indices);
	GPU::detail::bind_index_buffer(cmd, indices, index_type, no_offsets);

	auto [range, offset, address] = GPU::detail::closest_buffer(cmd->queue, args, no_offsets);
	auto start = range.start + offset;
	auto count = (range.end - start) / sizeof(GpuDrawIndexedIndirectCommand);

	// WebGPU has no multi draw indirect (outside of an extension), so every argument struct between
	// the provided pointer and the end of its allocation becomes its own indirect draw
	for(size_t i = 0; i < count; ++i)
		wgpuRenderPassEncoderDrawIndexedIndirect(cmd->render_pass, cmd->queue->monobuffers[range.buffer], start + i * sizeof(GpuDrawIndexedIndirectCommand));
}

void gpuDrawMeshlets(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, uvec3 dim) {
	throw std::runtime_error("WebGPU doesn't support mesh shaders!");
}

void gpuDrawMeshletsIndirect(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, gpu* dim, bool no_offsets /* = false */) {
	throw std::runtime_error("WebGPU doesn't support mesh shaders!");
}



namespace GPU::detail {
	// Hands the presentable texture back to the surface. Any view recorded from it outlives this call:
	// a WebGPU view keeps its texture alive internally, so the views gpuBeginRenderPass created are
	// still valid until the submission that used them releases them.
	inline void release_surface_texture(GpuSurface* surface) {
		if(!surface->acquired) return;

		wgpuTextureRelease(surface->current.texture);
		surface->current.texture = nullptr;
		surface->acquired = false;
	}

	// The requested present mode when the surface supports it, otherwise the best one it does:
	// mailbox (tear free at the lowest latency), then relaxed fifo, then the fifo every surface has.
	inline WGPUPresentMode pick_present_mode(PRESENT_MODE requested, const WGPUSurfaceCapabilities& caps) {
		auto supported = [&caps](WGPUPresentMode mode) {
			for(size_t i = 0; i < caps.presentModeCount; ++i)
				if(caps.presentModes[i] == mode) return true;
			return false;
		};

		if(requested != PRESENT_MODE_BEST_AVAILABLE)
			if(auto wanted = GPU::present2wgpu(requested); supported(wanted))
				return wanted;

		if(supported(WGPUPresentMode_Mailbox)) return WGPUPresentMode_Mailbox;
		if(supported(WGPUPresentMode_FifoRelaxed)) return WGPUPresentMode_FifoRelaxed;
		return WGPUPresentMode_Fifo; // Guaranteed to be supported
	}

	// The requested format when the surface supports it, otherwise the one it prefers. FORMAT_NONE asks
	// for the preferred one outright, which is the portable thing to do: it is BGRA8 on most platforms
	// and the only format offered on some of them.
	inline WGPUTextureFormat pick_surface_format(FORMAT requested, const WGPUSurfaceCapabilities& caps) {
		if(requested != FORMAT_NONE) {
			auto wanted = GPU::format2wgpu(requested);
			for(size_t i = 0; i < caps.formatCount; ++i)
				if(caps.formats[i] == wanted) return wanted;
		}

		return caps.formatCount ? caps.formats[0] : WGPUTextureFormat_BGRA8Unorm;
	}

	inline WGPUCompositeAlphaMode pick_alpha_mode(bool opaque, const WGPUSurfaceCapabilities& caps) {
		auto wanted = opaque ? WGPUCompositeAlphaMode_Opaque : WGPUCompositeAlphaMode_Premultiplied;
		for(size_t i = 0; i < caps.alphaModeCount; ++i)
			if(caps.alphaModes[i] == wanted) return wanted;

		// Auto aliases the surface's first supported mode and is never listed as one itself
		return WGPUCompositeAlphaMode_Auto;
	}
}

GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, WGPUSurface surface, const GpuSurfaceDescriptor& desc) {
	auto out = (GpuSurface*)queue->cpu_allocator(nullptr, sizeof(GpuSurface));
	new(out) GpuSurface {
		.surface = surface,
	};

	gpuSurfaceReconfigureEXT(queue, out, desc);
	return out;
}

void gpuFreeSurfaceEXT(GpuQueue* queue, GpuSurface* surface) {
	GPU::detail::release_surface_texture(surface);
	wgpuSurfaceUnconfigure(surface->surface);
	// The WGPUSurface itself was handed to us by whoever owns the window, so it is theirs to release
	// (matching the Vulkan backend, which leaves the VkSurfaceKHR alone too)

	auto allocator = queue->cpu_allocator;
	surface->~GpuSurface();
	allocator(surface, 0);
}

void gpuSurfaceReconfigureEXT(GpuQueue* queue, GpuSurface* surface, const GpuSurfaceDescriptor& desc) {
	// Whatever is still acquired belongs to the configuration being replaced
	GPU::detail::release_surface_texture(surface);

	WGPUSurfaceCapabilities caps = {};
	if(wgpuSurfaceGetCapabilities(surface->surface, queue->adapter, &caps) != WGPUStatus_Success) {
		errno = WGPUErrorType_Unknown;
		return;
	}

	surface->descriptor = desc;
	auto& texture = surface->descriptor.texture;

	// A presentable texture is a single 2D image with no mips and no multisampling, whatever was asked
	// for. Render into an MSAA texture of your own and resolve (or blit) into this one instead.
	texture.type = TEXTURE_2D;
	texture.mipCount = 1;
	texture.sampleCount = 1;
	texture.layerCount = 1;
	texture.dimensions.z = 1;
	assert(texture.dimensions.x > 0 && texture.dimensions.y > 0 && "A surface can't be configured for a zero sized window");

	auto format = GPU::detail::pick_surface_format(texture.format, caps);
	auto present_mode = GPU::detail::pick_present_mode(surface->descriptor.presentMode, caps);
	auto alpha_mode = GPU::detail::pick_alpha_mode(surface->descriptor.opaque, caps);

	// RenderAttachment is the one usage every surface offers, and the only one the rest of the API can
	// reach: a presentable texture is allocated by WebGPU itself, so it can't be placed inside a
	// monotexture, and a GpuTextureDescriptor (what a sampled or storage binding goes through) can only
	// name a texture that lives in one. Asking for USAGE_SAMPLED is still worth it where the surface
	// supports it, because gpuBlitTextureEXT binds its source view directly rather than through the
	// heap, which makes a presented frame readable that way.
	// emdawnwebgpu's wgpuSurfaceGetCapabilities never fills usages in, so a zero there means "not
	// reported" rather than "nothing supported" — masking against it would configure the surface with
	// no usage at all, which WebGPU rejects
	auto supported_usages = caps.usages ? caps.usages : WGPUTextureUsage(WGPUTextureUsage_RenderAttachment);
	auto usage = (GPU::usage2wgpu(texture.usage) | WGPUTextureUsage_RenderAttachment) & supported_usages;

	WGPUSurfaceConfiguration config {
		.device = queue->device,
		.format = format,
		.usage = usage,
		.width = texture.dimensions.x,
		.height = texture.dimensions.y,
		.alphaMode = alpha_mode,
		.presentMode = present_mode,
	};
	wgpuSurfaceConfigure(surface->surface, &config);
	wgpuSurfaceCapabilitiesFreeMembers(caps);

	// Report what the surface was actually given rather than what was requested, so that code which
	// has to match the swapchain (a pipeline's color target format above all) can just ask
	texture.format = GPU::wgpu2format(format);
	texture.usage = (TEXTURE_USAGE_FLAGS)(USAGE_COLOR_ATTACHMENT
		| ((usage & WGPUTextureUsage_TextureBinding) ? USAGE_SAMPLED : 0)
		| ((usage & WGPUTextureUsage_StorageBinding) ? USAGE_STORAGE : 0)
		| ((usage & WGPUTextureUsage_CopySrc) ? USAGE_TRANSFER_SRC : 0)
		| ((usage & WGPUTextureUsage_CopyDst) ? USAGE_TRANSFER_DST : 0));
	surface->descriptor.presentMode = GPU::wgpu2present(present_mode);
	surface->descriptor.opaque = alpha_mode == WGPUCompositeAlphaMode_Opaque
		|| (alpha_mode == WGPUCompositeAlphaMode_Auto && surface->descriptor.opaque);
}

GpuSurfaceCapabilities gpuGetSurfaceCapabilitiesEXT(GpuQueue* queue, GpuSurface* surface) {
	GpuSurfaceCapabilities out = {};

	WGPUSurfaceCapabilities caps = {};
	if(wgpuSurfaceGetCapabilities(surface->surface, queue->adapter, &caps) != WGPUStatus_Success) {
		errno = WGPUErrorType_Unknown;
		return out;
	}

	// WebGPU already reports the formats in the order it prefers them, which is the order
	// GPU::detail::pick_surface_format walks, so keeping it makes formats[0] the FORMAT_NONE pick.
	// The same format can be listed more than once (once per configuration the surface would
	// accept it in), so the first sighting is the one that counts.
	for(size_t i = 0; i < caps.formatCount && out.formatCount < GPU_MAX_SURFACE_FORMATS; ++i) {
		auto format = GPU::wgpu2format(caps.formats[i]);
		// A format the rest of the API can't name is one it could never be configured with either
		if(format == FORMAT_NONE) continue;
		auto listed = out.formatList();
		if(std::ranges::find(listed, format) == listed.end())
			out.formats[out.formatCount++] = format;
	}

	// GPU::detail::pick_present_mode's preference, so presentModes[0] is what
	// PRESENT_MODE_BEST_AVAILABLE resolves to. Immediate trails the tear free modes rather than
	// leading on its latency, matching the mode that picker never reaches for on its own.
	for(auto mode: {PRESENT_MODE_MAILBOX, PRESENT_MODE_FIFO_RELAXED, PRESENT_MODE_FIFO, PRESENT_MODE_IMMEDIATE}) {
		if(out.presentModeCount >= GPU_MAX_SURFACE_PRESENT_MODES) break;

		auto wanted = GPU::present2wgpu(mode);
		for(size_t i = 0; i < caps.presentModeCount; ++i)
			if(caps.presentModes[i] == wanted) {
				out.presentModes[out.presentModeCount++] = mode;
				break;
			}
	}

	// Auto is whatever the surface does by default, which is not a promise that it blends
	for(size_t i = 0; i < caps.alphaModeCount; ++i)
		if(caps.alphaModes[i] == WGPUCompositeAlphaMode_Premultiplied
			|| caps.alphaModes[i] == WGPUCompositeAlphaMode_Unpremultiplied)
			out.supportsTransparency = true;

	wgpuSurfaceCapabilitiesFreeMembers(caps);
	return out;
}

GpuSurfaceDescriptor gpuSurfaceGetConfigurationEXT(const GpuSurface* surface) {
	return surface->descriptor;
}

const GpuTexture* gpuSurfaceNextTextureEXT(GpuQueue* queue, GpuSurface* surface) {
	// Acquiring twice without presenting in between is a mistake, but it shouldn't leak the texture
	GPU::detail::release_surface_texture(surface);

	WGPUSurfaceTexture acquired = {};
	wgpuSurfaceGetCurrentTexture(surface->surface, &acquired);

	switch(acquired.status) {
	break; case WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal:
		// Nothing to report
	break; case WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal:
		// The texture no longer matches the window but can still be rendered into and presented. The
		// caller decides whether that is worth a reconfiguration.
		errno = SURFACE_SUBOPTIMAL;
	break; default:
		// Outdated, Lost, Timeout and Error all leave nothing to render into
		errno = acquired.status;
		if(acquired.texture) wgpuTextureRelease(acquired.texture);
		return nullptr;
	}

	// The texture's own size rather than the configured one: they agree, and taking it from the texture
	// keeps the descriptor honest if a platform ever rounds the configuration
	surface->current = GpuTexture {
		.descriptor = surface->descriptor.texture,
		// Deliberately no monotexture range. Everything that consumes a texture through the descriptor
		// heap therefore can't touch this one; gpuBlitTextureEXT is the bridge in both directions.
		.range = {},
		.texture = acquired.texture,
	};
	surface->current.descriptor.dimensions = {
		wgpuTextureGetWidth(acquired.texture),
		wgpuTextureGetHeight(acquired.texture),
		1
	};
	surface->acquired = true;
	return &surface->current;
}

void gpuSurfacePresentEXT(GpuQueue* queue, GpuSurface* surface, uint64_t wait_submission_index /* = NO_SUBMISSION_WAIT */) {
	assert(surface->acquired && "gpuSurfaceNextTextureEXT has to succeed before the frame can be presented");

	// WebGPU has one queue and presents behind everything already submitted to it, so there is nothing
	// to wait on here: any index the caller can name belongs to a submission that has already been
	// handed over. All the parameter can do on this backend is get checked.
	assert((wait_submission_index == NO_SUBMISSION_WAIT || wait_submission_index < queue->next_submission_index)
		&& "Presenting behind a submission that hasn't happened yet would deadlock on a backend that waits");

	GPU::detail::release_surface_texture(surface);
#ifndef __EMSCRIPTEN__
	// In the browser the canvas is presented when control returns to the event loop, and calling this
	// is an error there
	if(wgpuSurfacePresent(surface->surface) != WGPUStatus_Success)
		errno = WGPUErrorType_Unknown;
#endif
}
