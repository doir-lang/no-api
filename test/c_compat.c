#include <string.h>

#ifdef NOAPI_BACKEND_VULKAN
	#include <vulkan/noapi.h>
#else
	#include <webgpu/noapi.h>
#endif

static const char shader_ir[] = "// not a shader";

uint16_t noapi_c_compat_packed_default_sampler(void) {
	GpuSamplerDesc desc = GPU_SAMPLER_DESC_DEFAULT;
	return gpuSamplerDescPackEXT(desc);
}

static void noapi_c_compat_diagnostic(GpuQueue* queue, GPU_DIAGNOSTIC kind, GpuStringView message, void* userdata) {
	(void)queue; (void)kind; (void)message; (void)userdata;
}

void noapi_c_compat_api(GpuQueue* queue, GpuSurface* surface, gpu* data);
void noapi_c_compat_api(GpuQueue* queue, GpuSurface* surface, gpu* data) {
	GpuCapabilities capabilities = gpuGetCapabilitiesEXT(queue);
	(void)capabilities.mesh_shaders;
	(void)capabilities.split_barrier_signals;
	(void)capabilities.split_barriers;
	(void)capabilities.gpu_writable_texture_heap;
	(void)capabilities.descriptor_heap_in_any_memory;
	(void)capabilities.texture_descriptor_stride_ratio;
	gpuSetDiagnosticCallbackEXT(queue, noapi_c_compat_diagnostic, NULL);

	void* host = gpuMalloc(queue, 1024, 16, MEMORY_DEFAULT);
	void* heap_memory = gpuMalloc(queue, 1024, 64, MEMORY_DESCRIPTOR_HEAP);
	(void)heap_memory;
	gpu* device = gpuHostToDevicePointer(queue, host);
	void* mapped = gpuDeviceToHostPointerEXT(queue, device);

	GpuTextureDesc texture_desc = GPU_TEXTURE_DESC_DEFAULT;
	texture_desc.dimensions.x = 256;
	texture_desc.dimensions.y = 256;
	texture_desc.format = FORMAT_RGBA8_UNORM;
	texture_desc.usage = USAGE_SAMPLED | USAGE_COLOR_ATTACHMENT | USAGE_TRANSFER_DST;

	GpuTextureSizeAlign size_align = gpuTextureSizeAlign(queue, &texture_desc);
	gpu* texture_memory = (gpu*)gpuMalloc(queue, size_align.size, size_align.align, MEMORY_TEXTURE);
	GpuTexture* texture = gpuCreateTexture(queue, &texture_desc, texture_memory);

	GpuViewDesc view_desc = GPU_VIEW_DESC_DEFAULT;
	GpuTextureDescriptor sampled = gpuTextureViewDescriptor(queue, texture, &view_desc);
	GpuTextureDescriptor storage = gpuRWTextureViewDescriptor(queue, texture, &view_desc);

	GpuByteSpan ir;
	ir.ptr = shader_ir;
	ir.count = sizeof(shader_ir) - 1;

	GpuPipeline* compute = gpuCreateComputePipeline(queue, ir);

	GpuColorTarget target = GPU_COLOR_TARGET_DEFAULT;
	target.format = FORMAT_BGRA8_UNORM;

	GpuRasterDesc raster = GPU_RASTER_DESC_DEFAULT;
	raster.colorTargets.ptr = &target;
	raster.colorTargets.count = 1;
	raster.blendstate.has_value = true;
	raster.blendstate.value.srcColorFactor = FACTOR_SRC_ALPHA;
	raster.blendstate.value.dstColorFactor = FACTOR_ONE_MINUS_SRC_ALPHA;

	GpuPipeline* graphics = gpuCreateGraphicsPipeline(queue, ir, ir, &raster);

	GpuPipeline* meshlet = gpuCreateGraphicsMeshletPipeline(queue, ir, ir, &raster);

	GpuDepthStencilDesc depth_desc = GPU_DEPTH_STENCIL_DESC_DEFAULT;
	depth_desc.depthMode = DEPTH_READ | DEPTH_WRITE;
	depth_desc.depthTest = OP_LESS_EQUAL;
	GpuDepthStencilState* depth = gpuCreateDepthStencilState(queue, &depth_desc);

	GpuBlendDesc blend_desc = GPU_BLEND_DESC_DEFAULT;
	GpuBlendState* blend = gpuCreateBlendState(queue, &blend_desc);

	GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
	uvec3 grid = {1, 1, 1};
	uvec2 extent = {256, 256};
	ivec2 origin = {0, 0};

	gpuSetActiveTextureHeapPtr(cmd, data);
	gpuBarrier(cmd, STAGE_COMPUTE, STAGE_PIXEL_SHADER, HAZARD_DESCRIPTORS);
	gpuSignalAfter(cmd, STAGE_COMPUTE, data, 1, SIGNAL_ATOMIC_MAX);
	gpuWaitBefore(cmd, STAGE_PIXEL_SHADER, data, 1, OP_GREATER_EQUAL, HAZARD_DRAW_ARGUMENTS, ~(uint64_t)0);

	gpuSetPipeline(cmd, compute);
	gpuDispatch(cmd, data, grid);
	gpuDispatchIndirect(cmd, data, data);

	gpuMemCpy(cmd, data, data, 64);
	gpuCopyToTexture(cmd, data, data, texture);
	gpuCopyFromTexture(cmd, data, data, texture);
	gpuBlitTextureEXT(cmd, texture, texture, true, 0, 0, 0, 0);
	gpuSyncMemoryEXT(cmd, data);

	GpuSamplerDesc samplers[2] = {GPU_SAMPLER_DESC_DEFAULT, GPU_SAMPLER_DESC_DEFAULT};
	GpuSamplerDescSpan enabled_samplers;
	samplers[1].mag_filter = FILTER_NEAREST;
	samplers[1].address_mode_u = ADDRESS_MODE_CLAMP;
	enabled_samplers.ptr = samplers;
	enabled_samplers.count = 2;
	gpuSetEnabledSamplersEXT(cmd, enabled_samplers);

	GpuColorAttachment color = GPU_COLOR_ATTACHMENT_DEFAULT;
	color.texture = texture;
	color.loadOp = LOAD_OP_CLEAR;
	color.clearValue.r = 0.05f;

	GpuRenderPassDesc pass = GPU_RENDER_PASS_DESC_DEFAULT;
	pass.colorAttachments.ptr = &color;
	pass.colorAttachments.count = 1;
	pass.depthAttachment.has_value = false;

	GpuOptionalRenderPassDesc presented;
	presented.has_value = true;
	presented.value = pass;

	gpuBeginRenderPass(cmd, &pass);
	gpuSetViewportEXT(cmd, extent, origin, 0.0f, 1.0f);
	gpuSetScissorRectEXT(cmd, extent, origin);
	gpuSetDepthStencilState(cmd, depth);
	gpuSetBlendState(cmd, blend);
	gpuSetPipeline(cmd, graphics);
	gpuDrawIndexedInstanced(cmd, data, data, data, 3, 1, INDEX_TYPE_UINT32);
	gpuDrawIndexedInstancedIndirect(cmd, data, data, data, data, INDEX_TYPE_UINT32);
	gpuDrawMeshlets(cmd, data, data, grid);
	gpuDrawMeshletsIndirect(cmd, data, data, data);
	gpuEndRenderPass(cmd, presented);

	GpuSemaphore* semaphore = gpuCreateSemaphore(queue, 0);
	GpuCommandBufferSpan buffers;
	buffers.ptr = &cmd;
	buffers.count = 1;

	uint64_t submission = gpuSubmit(queue, buffers, semaphore, 1);
	gpuWaitSemaphore(queue, semaphore, 1, UINT64_MAX);
	gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), GPU_GET_VALUE, UINT64_MAX);
	gpuSyncMemoryImmediateEXT(queue, data);
	gpuWaitIdleEXT(queue);

	GpuSurfaceCapabilities caps = gpuGetSurfaceCapabilitiesEXT(queue, surface);
	GpuSurfaceDescriptor config = gpuSurfaceGetConfigurationEXT(surface);
	if(caps.formatCount) config.texture.format = caps.formats[0];
	if(caps.presentModeCount) config.presentMode = caps.presentModes[0];
	config.opaque = !caps.supportsTransparency;
	gpuSurfaceReconfigureEXT(queue, surface, &config);

	const GpuTexture* next = gpuSurfaceNextTextureEXT(queue, surface);
	gpuSurfacePresentEXT(queue, surface, submission);
	gpuSurfacePresentEXT(queue, surface, NO_SUBMISSION_WAIT);

	// Teardown
	gpuFreeCommandBuffer(gpuStartCommandRecording(queue));
	gpuFreeSemaphore(queue, semaphore);
	gpuFreeBlendState(queue, blend);
	gpuFreeDepthStencilState(queue, depth);
	gpuFreePipeline(queue, graphics);
	if(meshlet) gpuFreePipeline(queue, meshlet);
	gpuFreePipeline(queue, compute);
	gpuFreeSurfaceEXT(queue, surface);
	gpuFree(queue, host);
	gpuFreeDevicePointerEXT(queue, device);
	gpuFreeQueue(queue);

	(void)mapped; (void)sampled; (void)storage; (void)next;
}

#ifdef NOAPI_BACKEND_VULKAN

static VkSurfaceKHR noapi_c_compat_load_surface(VkInstance instance, void* userdata) {
	(void)instance; (void)userdata;
	return VK_NULL_HANDLE;
}

void noapi_c_compat_backend(void);
void noapi_c_compat_backend(void) {
	const char* const instance_extensions[] = {"VK_KHR_surface"};
	GpuCStringSpan extensions;
	GpuVulkanDefault vulkan;
	char error[256];

	extensions.ptr = instance_extensions;
	extensions.count = 1;

	GpuCStringSpan none;
	none.ptr = NULL;
	none.count = 0;

	if(!gpuSetupDefaultVulkanEXT(noapi_c_compat_load_surface, NULL, gpuDefaultErrorCallbackEXT,
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, extensions, none, none, true,
		&vulkan, error, sizeof(error)))
		return;

	VkPhysicalDeviceFeatures features;
	VkPhysicalDeviceVulkan12Features features12;
	memset(&features, 0, sizeof(features));
	memset(&features12, 0, sizeof(features12));
	features = gpuEnableRequiredVulkanFeaturesEXT(features);
	features12 = gpuEnableRequiredVulkan12FeaturesEXT(features12);
	GpuCStringSpan required = gpuRequiredVulkanDeviceExtensionsEXT();
	void* pnext = gpuRequiredVulkanDeviceCreateInfoPnextEXT();

	GpuQueue* queue = gpuCreateQueue(vulkan.instance, vulkan.gpu, vulkan.device,
		vulkan.graphics_queue, vulkan.graphics_queue_family, true, gpuDefaultCpuAllocator, NULL);

	GpuSurfaceDescriptor desc = GPU_SURFACE_DESCRIPTOR_DEFAULT;
	desc.texture.usage = USAGE_COLOR_ATTACHMENT;
	GpuSurface* surface = gpuCreateSurfaceEXT(queue, vulkan.surface, &desc);

	noapi_c_compat_api(queue, surface, NULL);
	(void)required; (void)pnext;
}

#else

static WGPUSurface noapi_c_compat_load_surface(WGPUInstance instance, void* userdata) {
	(void)instance; (void)userdata;
	return NULL;
}

void noapi_c_compat_backend(void);
void noapi_c_compat_backend(void) {
	GpuWebGPUDefault webgpu;
	char error[256];

	if(!gpuSetupDefaultWebGPUEXT(noapi_c_compat_load_surface, NULL, gpuDefaultErrorCallbackEXT, true,
		&webgpu, error, sizeof(error)))
		return;

	GpuQueue* queue = gpuCreateQueue(webgpu.adapter, webgpu.device, webgpu.limits, gpuDefaultCpuAllocator);

	GpuSurfaceDescriptor desc = GPU_SURFACE_DESCRIPTOR_DEFAULT;
	desc.texture.usage = USAGE_COLOR_ATTACHMENT;
	GpuSurface* surface = gpuCreateSurfaceEXT(queue, webgpu.surface, &desc);

	gpu* pointer = gpuEncodeWebGPUAddressEXT(0, 256);
	GpuWebGPUAddressEXT decoded = gpuDecodeWebGPUAddressEXT(pointer);
	gpu* again = gpuEncodeWebGPUAddressEXT(decoded.monobuffer, decoded.address);

	noapi_c_compat_api(queue, surface, again);
	(void)gpu_sampler_slot_count;
}

#endif
