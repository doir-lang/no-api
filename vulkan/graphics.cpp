#include <cassert>
#include <utility>
#include <volk.h>
#include <vk_mem_alloc.h>

#include "compute.hpp"
#include "noapi.hpp"
#include "common.hpp"

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added

struct GraphicsPipelinePushConstants {
	gpu* vertex;
	gpu* fragment;
	gpu* index;
	gpu* sampler_map;
};


VkBlendOp blend2vulkan(BLEND op) {
	switch (op) {
	case BLEND_ADD: return VK_BLEND_OP_ADD;
	case BLEND_SUBTRACT: return VK_BLEND_OP_SUBTRACT;
	case BLEND_MIN: return VK_BLEND_OP_MIN;
	case BLEND_MAX: return VK_BLEND_OP_MAX;
	case BLEND_REV_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
	}
	std::unreachable();
};

VkBlendFactor factor2vulkan(FACTOR f) {
	switch (f) {
		case FACTOR_ZERO: return VK_BLEND_FACTOR_ZERO;
		case FACTOR_ONE: return VK_BLEND_FACTOR_ONE;
		case FACTOR_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR;
		case FACTOR_ONE_MINUS_SRC_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
		case FACTOR_DST_COLOR: return VK_BLEND_FACTOR_DST_COLOR;
		case FACTOR_ONE_MINUS_DST_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
		case FACTOR_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
		case FACTOR_ONE_MINUS_SRC_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		case FACTOR_DST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA;
		case FACTOR_ONE_MINUS_DST_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
		case FACTOR_SRC1_COLOR: return VK_BLEND_FACTOR_SRC1_COLOR;
		case FACTOR_ONE_MINUS_SRC1_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
		case FACTOR_SRC1_ALPHA: return VK_BLEND_FACTOR_SRC1_ALPHA;
		case FACTOR_ONE_MINUS_SRC1_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
	}
	std::unreachable();
};

VkColorComponentFlags mask2vulkan(uint8_t mask) {
	VkColorComponentFlags flags = 0;
	if (mask & 0x1) flags |= VK_COLOR_COMPONENT_R_BIT;
	if (mask & 0x2) flags |= VK_COLOR_COMPONENT_G_BIT;
	if (mask & 0x4) flags |= VK_COLOR_COMPONENT_B_BIT;
	if (mask & 0x8) flags |= VK_COLOR_COMPONENT_A_BIT;
	return flags;
};



GpuPipeline* gpuCreateGraphicsPipeline(GpuQueue* queue, std::span<const std::byte> vertexIR, std::span<const std::byte> fragmentIR, const GpuRasterDesc& desc) {
	constexpr static auto topology2vulkan = [](TOPOLOGY t) -> VkPrimitiveTopology{
		switch (t) {
			// case TOPOLOGY_POINT_LIST: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
			// case TOPOLOGY_LINE_LIST: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
			// case TOPOLOGY_LINE_STRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
			case TOPOLOGY_TRIANGLE_LIST: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			case TOPOLOGY_TRIANGLE_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
		}
		std::unreachable();
	};
	constexpr static auto cull2vulkan = [](CULL c) -> VkCullModeFlags {
		// We are going to define CCW as front
		switch (c) {
			case CULL_NONE: return VK_CULL_MODE_NONE;
			case CULL_CCW: return VK_CULL_MODE_FRONT_BIT;
			case CULL_CW: return VK_CULL_MODE_BACK_BIT;
			case CULL_ALL: return VK_CULL_MODE_FRONT_AND_BACK;
		}
		std::unreachable();
	};

	auto out = (GpuPipeline*)queue->cpu_allocator(nullptr, sizeof(GpuPipeline));
	out->color_target_count = desc.colorTargets.size();

	std::array<VkShaderModule, 2> shader_modules;
	{
		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = vertexIR.size(),
			.pCode = (uint32_t*)vertexIR.data(),
		};
		VK_CHECK(vkCreateShaderModule(queue->device, &info, queue->callbacks, &shader_modules[0]), nullptr);
	}{
		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = fragmentIR.size(),
			.pCode = (uint32_t*)fragmentIR.data(),
		};
		VK_CHECK(vkCreateShaderModule(queue->device, &info, queue->callbacks, &shader_modules[1]), nullptr);
	}

	std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages {
		VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = shader_modules[0],
			.pName = "main",
		}, VkPipelineShaderStageCreateInfo {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = shader_modules[1],
			.pName = "main",
		}
	};

	VkPipelineVertexInputStateCreateInfo vertex_input_state = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

	VkPipelineInputAssemblyStateCreateInfo assembly_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology2vulkan(desc.topology),
	};

	VkPipelineViewportStateCreateInfo viewport_state {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	VkPipelineRasterizationStateCreateInfo rasterization_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = cull2vulkan(desc.cull),
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo multisample_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = GPU::detail::samples2vulkan(desc.sampleCount),
		.alphaToCoverageEnable = desc.alphaToCoverage ? VK_TRUE : VK_FALSE,
	};

	VkPipelineDepthStencilStateCreateInfo depth_stencil_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

	std::vector<VkPipelineColorBlendAttachmentState> attachments;
	{
		attachments.reserve(desc.colorTargets.size());

		const bool blendEnabled = desc.blendstate.has_value();
		const GpuBlendDesc blend = desc.blendstate.value_or(GpuBlendDesc{});

		for (const GpuColorTarget& target : desc.colorTargets) {
			VkPipelineColorBlendAttachmentState state{};
			state.blendEnable = blendEnabled ? VK_TRUE : VK_FALSE;

			if (blendEnabled) {
				state.srcColorBlendFactor = factor2vulkan(blend.srcColorFactor);
				state.dstColorBlendFactor = factor2vulkan(blend.dstColorFactor);
				state.colorBlendOp = blend2vulkan(blend.colorOp);
				state.srcAlphaBlendFactor = factor2vulkan(blend.srcAlphaFactor);
				state.dstAlphaBlendFactor = factor2vulkan(blend.dstAlphaFactor);
				state.alphaBlendOp = blend2vulkan(blend.alphaOp);
			}

			const uint8_t effectiveMask = blendEnabled ? (target.writeMask & blend.colorWriteMask) : target.writeMask;
			state.colorWriteMask = mask2vulkan(effectiveMask);

			attachments.push_back(state);
		}
	}

	VkPipelineColorBlendStateCreateInfo color_blend_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = static_cast<uint32_t>(attachments.size()),
		.pAttachments = attachments.empty() ? nullptr : attachments.data(),
	};

	std::array<VkDynamicState, 14> dynamic = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
		VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
		VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
		VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
		VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
		VK_DYNAMIC_STATE_STENCIL_OP,
		VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
		VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
		VK_DYNAMIC_STATE_STENCIL_REFERENCE,
		VK_DYNAMIC_STATE_DEPTH_BIAS,
		VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
		VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
		VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,
	};

	VkPipelineDynamicStateCreateInfo dynamic_state{};
	dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic_state.dynamicStateCount = 2;
	dynamic_state.pDynamicStates = dynamic.data();

	std::vector<VkFormat> color_formats; color_formats.reserve(desc.colorTargets.size());
	for(auto& target: desc.colorTargets)
		color_formats.emplace_back(GPU::detail::format2vulkan(target.format));

	VkPipelineRenderingCreateInfo dynamic_rendering_info {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = static_cast<uint32_t>(color_formats.size()),
		.pColorAttachmentFormats = color_formats.data(),
		.depthAttachmentFormat = desc.depthFormat == FORMAT_NONE ? VK_FORMAT_UNDEFINED : GPU::detail::format2vulkan(desc.depthFormat),
		.stencilAttachmentFormat = desc.stencilFormat == FORMAT_NONE ? VK_FORMAT_UNDEFINED : GPU::detail::format2vulkan(desc.stencilFormat),
	};

	VkPipelineCreateFlags2CreateInfo create_flags {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
		.pNext = &dynamic_rendering_info,
		.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT
	};
	VkGraphicsPipelineCreateInfo info {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &create_flags,
		.stageCount = shader_stages.size(),
		.pStages = shader_stages.data(),
		.pVertexInputState = &vertex_input_state,
		.pInputAssemblyState = &assembly_state,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization_state,
		.pMultisampleState = &multisample_state,
		.pDepthStencilState = &depth_stencil_state,
		.pColorBlendState = &color_blend_state,
		.pDynamicState = &dynamic_state,
		.renderPass = VK_NULL_HANDLE,
		// .subpass = 0,
		.basePipelineIndex = -1
	};
	vkCreateGraphicsPipelines(queue->device, VK_NULL_HANDLE, 1, &info, queue->callbacks, &out->pipeline);

	for(auto module: shader_modules)
		vkDestroyShaderModule(queue->device, module, queue->callbacks);
	// vkDestroyRenderPass(queue->device, compatible_render_pass, queue->callbacks);

	return out;
}

// TODO: Untested!
GpuDepthStencilState* gpuCreateDepthStencilState(GpuQueue* queue, const GpuDepthStencilDesc& desc) {
	auto out = (GpuDepthStencilState*)queue->cpu_allocator(nullptr, sizeof(GpuDepthStencilState));
	out->descriptor = desc;
	return out;
}

// TODO: Untested!
GpuBlendState* gpuCreateBlendState(GpuQueue* queue, const GpuBlendDesc& desc) {
	auto out = (GpuBlendState*)queue->cpu_allocator(nullptr, sizeof(GpuBlendState));
	out->descriptor = desc;
	return out;
}

// TODO: Untested!
void gpuFreeDepthStencilState(GpuQueue* queue, GpuDepthStencilState* state) {
	queue->cpu_allocator(state, 0);
}

// TODO: Untested!
void gpuFreeBlendState(GpuQueue* queue, GpuBlendState* state) {
	queue->cpu_allocator(state, 0);
}

// TODO: Untested!
void gpuSetDepthStencilState(GpuCommandBuffer* cmd, const GpuDepthStencilState* state) {
	constexpr static auto op2vulkan = [](OP op) -> VkCompareOp {
		switch (op) {
			case OP_ALWAYS: return VK_COMPARE_OP_ALWAYS;
			case OP_NEVER: return VK_COMPARE_OP_NEVER;
			case OP_LESS: return VK_COMPARE_OP_LESS;
			case OP_LESS_EQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
			case OP_GREATER: return VK_COMPARE_OP_GREATER;
			case OP_GREATER_EQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
			case OP_EQUAL: return VK_COMPARE_OP_EQUAL;
			case OP_NOT_EQUAL: return VK_COMPARE_OP_NOT_EQUAL;
		}
		std::unreachable();
	};
	constexpr static auto stencil2vulkan = [](STENCIL_OP op) -> VkStencilOp {
		switch (op) {
			case STENCIL_OP_KEEP: return VK_STENCIL_OP_KEEP;
			case STENCIL_OP_ZERO: return VK_STENCIL_OP_ZERO;
			case STENCIL_OP_REPLACE: return VK_STENCIL_OP_REPLACE;
			case STENCIL_OP_INCR_SAT: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
			case STENCIL_OP_DECR_SAT: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
			case STENCIL_OP_INVERT: return VK_STENCIL_OP_INVERT;
			case STENCIL_OP_INCR_WRAP: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
			case STENCIL_OP_DECR_WRAP: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
		}
		std::unreachable();
	};

	const bool depthRead = (state->descriptor.depthMode & DEPTH_READ) != 0;
	const bool depthWrite = (state->descriptor.depthMode & DEPTH_WRITE) != 0;

	// depthTestEnable gates both the compare op AND whether writes happen
	// in classic GL/D3D semantics; Vulkan separates test-enable from
	// write-enable, so: enable the test if either read or write was
	// requested (write without a passing test wouldn't write anything
	// meaningful), and gate the actual write bit off DEPTH_WRITE.
	vkCmdSetDepthTestEnable(cmd->command_buffer, (depthRead || depthWrite) ? VK_TRUE : VK_FALSE);
	vkCmdSetDepthWriteEnable(cmd->command_buffer, depthWrite ? VK_TRUE : VK_FALSE);
	vkCmdSetDepthCompareOp(cmd->command_buffer, op2vulkan(state->descriptor.depthTest));

	const bool stencilEnabled = state->descriptor.stencilFront.test != OP_ALWAYS || state->descriptor.stencilBack.test != OP_ALWAYS
		|| state->descriptor.stencilFront.failOp != STENCIL_OP_KEEP || state->descriptor.stencilFront.passOp != STENCIL_OP_KEEP
		|| state->descriptor.stencilFront.depthFailOp != STENCIL_OP_KEEP
		|| state->descriptor.stencilBack.failOp != STENCIL_OP_KEEP || state->descriptor.stencilBack.passOp != STENCIL_OP_KEEP
		|| state->descriptor.stencilBack.depthFailOp != STENCIL_OP_KEEP;
	vkCmdSetStencilTestEnable(cmd->command_buffer, stencilEnabled ? VK_TRUE : VK_FALSE);

	if (stencilEnabled) {
		vkCmdSetStencilOp(cmd->command_buffer, VK_STENCIL_FACE_FRONT_BIT, stencil2vulkan(state->descriptor.stencilFront.failOp),
			stencil2vulkan(state->descriptor.stencilFront.passOp), stencil2vulkan(state->descriptor.stencilFront.depthFailOp),
			op2vulkan(state->descriptor.stencilFront.test)
		);
		vkCmdSetStencilOp(cmd->command_buffer, VK_STENCIL_FACE_BACK_BIT, stencil2vulkan(state->descriptor.stencilBack.failOp),
			stencil2vulkan(state->descriptor.stencilBack.passOp), stencil2vulkan(state->descriptor.stencilBack.depthFailOp),
			op2vulkan(state->descriptor.stencilBack.test)
		);

		vkCmdSetStencilCompareMask(cmd->command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, state->descriptor.stencilReadMask);
		vkCmdSetStencilWriteMask(cmd->command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, state->descriptor.stencilWriteMask);

		// Front/back reference values differ in your struct
		// (Stencil::reference is per-face) but vkCmdSetStencilReference
		// takes a face mask too, so two calls if front != back.
		if (state->descriptor.stencilFront.reference == state->descriptor.stencilBack.reference) {
			vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, state->descriptor.stencilFront.reference);
		} else {
			vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_FRONT_BIT, state->descriptor.stencilFront.reference);
			vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_BACK_BIT, state->descriptor.stencilBack.reference);
		}
	}

	vkCmdSetDepthBias(cmd->command_buffer, state->descriptor.depthBias, state->descriptor.depthBiasClamp, state->descriptor.depthBiasSlopeFactor);
}

// TODO: Untested!
void gpuSetBlendState(GpuCommandBuffer* cmd, const GpuBlendState* state) {
	assert(cmd->bound_pipeline);

	const uint32_t count = cmd->bound_pipeline->color_target_count.value_or(0);
	const bool blend_enable = state->descriptor.colorWriteMask > 0;

	std::vector<VkBool32> enables(count, blend_enable ? VK_TRUE : VK_FALSE);
	vkCmdSetColorBlendEnableEXT(cmd->command_buffer, 0, count, enables.data());

	if(blend_enable) {
		std::vector<VkColorBlendEquationEXT> equations(count);
		for (uint32_t i = 0; i < count; ++i) {
			equations[i].srcColorBlendFactor = factor2vulkan(state->descriptor.srcColorFactor);
			equations[i].dstColorBlendFactor = factor2vulkan(state->descriptor.dstColorFactor);
			equations[i].colorBlendOp = blend2vulkan(state->descriptor.colorOp);
			equations[i].srcAlphaBlendFactor = factor2vulkan(state->descriptor.srcAlphaFactor);
			equations[i].dstAlphaBlendFactor = factor2vulkan(state->descriptor.dstAlphaFactor);
			equations[i].alphaBlendOp = blend2vulkan(state->descriptor.alphaOp);
		}
		vkCmdSetColorBlendEquationEXT(cmd->command_buffer, 0, count, equations.data());
	}

	// std::vector<VkColorComponentFlags> masks(count);
	// for (uint32_t i = 0; i < count; ++i)
	// 	masks[i] = mask2vulkan(writeMasks[i] & state->descriptor.colorWriteMask);
	// vkCmdSetColorWriteMaskEXT(cmd, 0, count, masks.data());
}

void gpuSetViewportEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin /*= {0, 0} */, float depth_min /* = 0 */, float depth_max /* = 1 */) {
	VkViewport viewport {
		.x = static_cast<float>(origin.x),
		.y = static_cast<float>(origin.y),
		.width = static_cast<float>(extent.x),
		.height = static_cast<float>(extent.y),
		.minDepth = depth_min,
		.maxDepth = depth_max,
	};
	vkCmdSetViewport(cmd->command_buffer, 0, 1, &viewport);
}

void gpuSetScissorRectEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin /* = {0, 0} */) {
	VkRect2D scissor { // TODO: Should we support additional scissors?
		.offset = {origin.x, origin.y},
		.extent = {extent.x, extent.y},
	};
	vkCmdSetScissor(cmd->command_buffer, 0, 1, &scissor);
}

inline void transition_image_layout(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout, VkAccessFlags source_access_mask, VkAccessFlags destination_access_mask, VkPipelineStageFlags source_stage, VkPipelineStageFlags destination_stage, uint32_t base_mip_level, uint32_t mip_levels, uint32_t base_slice, uint32_t slices) {
	VkImageMemoryBarrier barrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = source_access_mask,
		.dstAccessMask = destination_access_mask,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = base_mip_level == ALL_MIPS ? 0 : base_mip_level,
			.levelCount = base_mip_level == ALL_MIPS ? mip_levels : 1,
			.baseArrayLayer = base_slice == ALL_LAYERS ? 0 : base_slice,
			.layerCount = base_slice == ALL_LAYERS ? slices : 1,
		}
	};
	vkCmdPipelineBarrier(cmd, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void gpuBeginRenderPass(GpuCommandBuffer* cmd, const GpuRenderPassDesc& desc) {
	constexpr static auto view_create_info = [](const GpuTexture* texture) -> VkImageViewCreateInfo {
		return {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = texture->image,
			.viewType = GPU::detail::type2vulkan(texture->descriptor.type),
			.format = GPU::detail::format2vulkan(texture->descriptor.format),
			.subresourceRange = {
				.aspectMask = static_cast<VkImageAspectFlags>(gpuFormatIsDepth(texture->descriptor.format)
					? VK_IMAGE_ASPECT_DEPTH_BIT | (gpuFormatIsStencil(texture->descriptor.format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
					: VK_IMAGE_ASPECT_COLOR_BIT),
				.baseMipLevel = 0,
				.levelCount = texture->descriptor.mipCount,
				.baseArrayLayer = 0,
				.layerCount = texture->descriptor.layerCount
			}
		};
	};

	constexpr static auto load2vulkan = [](LOAD_OP op) {
		switch (op) {
		case LOAD_OP_LOAD: return VK_ATTACHMENT_LOAD_OP_LOAD;
		case LOAD_OP_CLEAR: return VK_ATTACHMENT_LOAD_OP_CLEAR;
		case LOAD_OP_DONT_CARE: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		}
		std::unreachable();
	};
	constexpr static auto store2vulkan = [](STORE_OP op) {
		switch (op) {
		case STORE_OP_STORE: return VK_ATTACHMENT_STORE_OP_STORE;
		case STORE_OP_DONT_CARE: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
		}
		std::unreachable();
	};

	assert(cmd->state == GpuCommandBuffer::Recording && "The command buffer must not be ended or recording another active render pass");
	cmd->state = GpuCommandBuffer::RecordingRenderPass;

	std::vector<VkRenderingAttachmentInfo> color_attachments; color_attachments.reserve(desc.colorAttachments.size());
	for(auto& color: desc.colorAttachments) {
		if(color.texture->available_semaphore)
			cmd->wait_semaphores.push_back(color.texture->available_semaphore);

		transition_image_layout(cmd->command_buffer,
			color.texture->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, color.mipLevel, color.texture->descriptor.mipCount, color.slice, color.texture->descriptor.sampleCount
		);

		if(!color.texture->full_view) {
			auto info = view_create_info(color.texture);
			vkCreateImageView(cmd->queue->device, &info, cmd->queue->callbacks, const_cast<VkImageView*>(&color.texture->full_view));
		}
		if(color.resolveTexture && !color.resolveTexture->full_view) {
			auto info = view_create_info(color.resolveTexture);
			VK_CHECK(vkCreateImageView(cmd->queue->device, &info, cmd->queue->callbacks, const_cast<VkImageView*>(&color.resolveTexture->full_view)), /*NOTHING*/);
		}

		color_attachments.emplace_back(VkRenderingAttachmentInfo{
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = color.texture->full_view,
			.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.resolveMode = color.resolveTexture ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT : VK_RESOLVE_MODE_NONE,
			.resolveImageView = color.resolveTexture ? color.resolveTexture->full_view : VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = load2vulkan(color.loadOp),
			.storeOp = store2vulkan(color.storeOp),
			.clearValue = {
				.color = {
					.float32 = {color.clearValue.r, color.clearValue.g, color.clearValue.b, color.clearValue.a}
				}
			}
		});
	}

	VkRenderingAttachmentInfo depth_attachment;
	if(desc.depthAttachment) {
		if(desc.depthAttachment->texture->available_semaphore)
			cmd->wait_semaphores.push_back(desc.depthAttachment->texture->available_semaphore);

		transition_image_layout(cmd->command_buffer,
			desc.depthAttachment->texture->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, 0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, desc.depthAttachment->mipLevel, desc.depthAttachment->texture->descriptor.mipCount,
			desc.depthAttachment->slice, desc.depthAttachment->texture->descriptor.sampleCount
		);

		if(!desc.depthAttachment->texture->full_view) {
			auto info = view_create_info(desc.depthAttachment->texture);
			VK_CHECK(vkCreateImageView(cmd->queue->device, &info, cmd->queue->callbacks, const_cast<VkImageView*>(&desc.depthAttachment->texture->full_view)), /*nothing*/);
		}

		depth_attachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = desc.depthAttachment->texture->full_view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.resolveMode = VK_RESOLVE_MODE_NONE,
			.resolveImageView = VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = load2vulkan(desc.depthAttachment->loadOp),
			.storeOp = store2vulkan(desc.depthAttachment->storeOp),
			.clearValue = {
				.depthStencil = {
					.depth = static_cast<float>(desc.depthAttachment->clearValue)
				}
			}
		};
	}

	VkRenderingAttachmentInfo stencil_attachment;
	if(desc.stencilAttachment) {
		if(desc.stencilAttachment->texture->available_semaphore)
			cmd->wait_semaphores.push_back(desc.stencilAttachment->texture->available_semaphore);

		transition_image_layout(cmd->command_buffer,
			desc.stencilAttachment->texture->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, desc.stencilAttachment->mipLevel, desc.stencilAttachment->texture->descriptor.mipCount,
			desc.stencilAttachment->slice, desc.stencilAttachment->texture->descriptor.sampleCount
		);

		if(!desc.stencilAttachment->texture->full_view) {
			auto info = view_create_info(desc.stencilAttachment->texture);
			VK_CHECK(vkCreateImageView(cmd->queue->device, &info, cmd->queue->callbacks, const_cast<VkImageView*>(&desc.stencilAttachment->texture->full_view)), /*NOTHING*/);
		}

		stencil_attachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = desc.stencilAttachment->texture->full_view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.resolveMode = VK_RESOLVE_MODE_NONE,
			.resolveImageView = VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = load2vulkan(desc.stencilAttachment->loadOp),
			.storeOp = store2vulkan(desc.stencilAttachment->storeOp),
			.clearValue = {
				.depthStencil = {
					.stencil = static_cast<uint32_t>(desc.stencilAttachment->clearValue)
				}
			}
		};
	}

	VkRenderingInfo info{
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = {
			.offset = {0, 0},
			.extent = {desc.colorAttachments[0].texture->descriptor.dimensions.x, desc.colorAttachments[0].texture->descriptor.dimensions.y},
		},
		.layerCount = 1,
		.colorAttachmentCount = static_cast<uint32_t>(color_attachments.size()),
		.pColorAttachments = color_attachments.data(),
		.pDepthAttachment = desc.depthAttachment ? &depth_attachment : nullptr,
		.pStencilAttachment = desc.stencilAttachment ? &stencil_attachment : nullptr,
	};
	vkCmdBeginRendering(cmd->command_buffer, &info);

	gpuSetViewportEXT(cmd, {info.renderArea.extent.width, info.renderArea.extent.height});
	gpuSetScissorRectEXT(cmd, {info.renderArea.extent.width, info.renderArea.extent.height});
}

void gpuEndRenderPass(GpuCommandBuffer* cmd, std::optional<const GpuRenderPassDesc> desc /*= {}*/) {
	vkCmdEndRendering(cmd->command_buffer);

	if(desc) for(auto& color: desc->colorAttachments)
		transition_image_layout(cmd->command_buffer,
			color.texture->image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, color.mipLevel, color.texture->descriptor.mipCount, color.slice, color.texture->descriptor.sampleCount
		);

	cmd->state = GpuCommandBuffer::Recording;
}

namespace GPU::detail {
	std::pair<VkBuffer, VkDeviceSize> ensureIndexBufferAvailable(GpuQueue* queue, gpu* indicesGpu, bool no_offsets, bool no_index_buffer_changes) {
		constexpr static std::pair<VkBuffer, VkDeviceSize> null_out = {VK_NULL_HANDLE, 0};

		auto [_buffer, offset, address] = GPU::detail::closest_buffer(queue, indicesGpu, no_offsets);
		auto [source_buffer, _alloc, size] = queue->allocations[address];
		if(!queue->gpu2index.contains(address)) {
			VkBufferCreateInfo buffer_info {
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = size,
				.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
			};
			VmaAllocationCreateInfo alloc_info {
				.usage = VMA_MEMORY_USAGE_AUTO,
				.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
			};

			auto& [index_buffer, allocation, index_size] = queue->gpu2index[address];
			index_size = size;
			VK_CHECK(vmaCreateBuffer(queue->gpu_allocator, &buffer_info, &alloc_info, &index_buffer, &allocation, nullptr), null_out);

			no_index_buffer_changes = false;
		}

		auto [index_buffer, _allocation, _size] = queue->gpu2index[address];
		if(!no_index_buffer_changes) {
			VkCommandBuffer tmp = VK_NULL_HANDLE;
			VkCommandBufferAllocateInfo info {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = queue->command_pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1
			};
			VK_CHECK(vkAllocateCommandBuffers(queue->device, &info, &tmp), null_out);

			VkCommandBufferBeginInfo begin {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
			};
			VK_CHECK(vkBeginCommandBuffer(tmp, &begin), null_out);

			VkBufferCopy copy {
				.srcOffset = offset,
				.dstOffset = offset,
				.size = size - offset
			};
			vkCmdCopyBuffer(tmp, source_buffer, index_buffer, 1, &copy);

			VK_CHECK(vkEndCommandBuffer(tmp), null_out);
			VkSubmitInfo submit {
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.commandBufferCount = 1,
				.pCommandBuffers = &tmp,
			};
			VK_CHECK(vkQueueSubmit(queue->queue, 1, &submit, VK_NULL_HANDLE), null_out);
			queue->command_buffers_pending_free.emplace_back(tmp, queue->command_submission_timeline_semaphore_next_value);
		}

		return {index_buffer, offset};
	}
}

VkIndexType index2vulkan(INDEX_TYPE_EXT index_type) {
	switch (index_type) {
	case INDEX_TYPE_UINT8: return VK_INDEX_TYPE_UINT8;
	case INDEX_TYPE_UINT16: return VK_INDEX_TYPE_UINT16;
	case INDEX_TYPE_UINT32: return VK_INDEX_TYPE_UINT32;
	}
	std::unreachable();
}

void gpuDrawIndexedInstanced(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, uint32_t index_count, uint32_t instance_count, INDEX_TYPE_EXT index_type /* = INDEX_TYPE_UINT32 */, bool no_offsets /* = false */, bool no_index_buffer_changes /* = false */) {
	GraphicsPipelinePushConstants data {
		.vertex = vertex_data,
		.fragment = fragment_data,
		.index = indices,
		.sampler_map = (gpu*)cmd->sampler_map
	};
	VkPushDataInfoEXT info {
		.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
		.offset = 0,
		.data = {
			.address = &data,
			.size = sizeof(GraphicsPipelinePushConstants)
		}
	};
	vkCmdPushDataEXT(cmd->command_buffer, &info);

	auto [index_buffer, offset] = GPU::detail::ensureIndexBufferAvailable(cmd->queue, indices, no_offsets, no_index_buffer_changes);
	assert(index_buffer != VK_NULL_HANDLE);

	vkCmdBindIndexBuffer(cmd->command_buffer, index_buffer, offset, index2vulkan(index_type));
	vkCmdDrawIndexed(cmd->command_buffer, index_count, instance_count, 0, 0, 0);
}

// TODO: Untested!
void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, gpu* args, INDEX_TYPE_EXT index_type /* = INDEX_TYPE_UINT32 */, bool no_offsets /* = false */, bool no_index_buffer_changes /* = false */) {
	GraphicsPipelinePushConstants data {
		.vertex = vertex_data,
		.fragment = fragment_data,
		.index = indices,
		.sampler_map = (gpu*)cmd->sampler_map
	};
	VkPushDataInfoEXT info {
		.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
		.offset = 0,
		.data = {
			.address = &data,
			.size = sizeof(GraphicsPipelinePushConstants)
		}
	};
	vkCmdPushDataEXT(cmd->command_buffer, &info);

	{
		auto [index_buffer, offset] = GPU::detail::ensureIndexBufferAvailable(cmd->queue, indices, no_offsets, no_index_buffer_changes);
		assert(index_buffer != VK_NULL_HANDLE);
		vkCmdBindIndexBuffer(cmd->command_buffer, index_buffer, offset, index2vulkan(index_type));
	}

	auto [_buffer, offset, address] = GPU::detail::closest_buffer(cmd->queue, args, no_offsets);
	auto [buffer, _allocation, size] = cmd->queue->allocations[address];
	auto count = (size - offset) / sizeof(VkDrawIndexedIndirectCommand);
	vkCmdDrawIndexedIndirect(cmd->command_buffer, buffer, offset, count, sizeof(VkDrawIndexedIndirectCommand)); // TODO: We should we check the size of the buffer and divide by sizeof(VkDrawIndexedIndirectCommand)?
}

// TODO: Untested!
void gpuDrawMeshlets(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, uvec3 dim) {
	GraphicsPipelinePushConstants data {
		.vertex = meshlet_data,
		.fragment = fragment_data,
		.index = nullptr,
		.sampler_map = (gpu*)cmd->sampler_map
	};
	VkPushDataInfoEXT info {
		.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
		.offset = 0,
		.data = {
			.address = &data,
			.size = sizeof(GraphicsPipelinePushConstants)
		}
	};
	vkCmdPushDataEXT(cmd->command_buffer, &info);

	vkCmdDrawMeshTasksEXT(cmd->command_buffer, dim.x, dim.y, dim.z);
}

// TODO: Untested!
void gpuDrawMeshletsIndirect(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, gpu* dim, bool no_offsets /* = false */) {
	GraphicsPipelinePushConstants data {
		.vertex = meshlet_data,
		.fragment = fragment_data,
		.index = nullptr,
		.sampler_map = (gpu*)cmd->sampler_map
	};
	VkPushDataInfoEXT info {
		.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
		.offset = 0,
		.data = {
			.address = &data,
			.size = sizeof(GraphicsPipelinePushConstants)
		}
	};
	vkCmdPushDataEXT(cmd->command_buffer, &info);

	auto [_buffer, offset, address] = GPU::detail::closest_buffer(cmd->queue, dim, no_offsets);
	auto [buffer, _allocation, size] = cmd->queue->allocations[address];
	auto count = (size - offset) / sizeof(VkDrawMeshTasksIndirectCommandEXT); // VkDrawMeshTasksIndirectCommandEXT == uvec3
	vkCmdDrawMeshTasksIndirectEXT(cmd->command_buffer, buffer, offset, count, sizeof(VkDrawMeshTasksIndirectCommandEXT));
}