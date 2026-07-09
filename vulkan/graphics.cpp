#include <cassert>
#include <utility>
#include <volk.h>
#include <vk_mem_alloc.h>

#include "compute.hpp"
#include "noapi.hpp"
#include "vulkan/common.hpp"

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added


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



inline VkRenderPass create_compatible_render_pass(VkDevice device, const GpuRasterDesc& raster) {
	std::vector<VkAttachmentDescription> attachmentDescs;
	std::vector<VkAttachmentReference> colorRefs;

	attachmentDescs.reserve(raster.colorTargets.size() + 1);
	colorRefs.reserve(raster.colorTargets.size());

	for (const ColorTarget& target : raster.colorTargets) {
		VkAttachmentDescription desc{};
		desc.format = GPU::detail::format2vulkan(target.format);
		desc.samples = GPU::detail::samples2vulkan(raster.sampleCount);
		desc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // adjust to your clear policy
		desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		desc.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		desc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference ref{};
		ref.attachment = static_cast<uint32_t>(attachmentDescs.size());
		ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		attachmentDescs.push_back(desc);
		colorRefs.push_back(ref);
	}

	VkAttachmentReference depthRef{};
	const bool hasDepth = raster.depthFormat != FORMAT_NONE;
	const bool hasStencil = raster.stencilFormat != FORMAT_NONE || gpuFormatIsStencil(raster.depthFormat);
	if (hasDepth) {
		VkAttachmentDescription desc{
			.format = GPU::detail::format2vulkan(raster.depthFormat),
			.samples = GPU::detail::samples2vulkan(raster.sampleCount),
			.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = hasStencil ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = hasStencil ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		depthRef.attachment = static_cast<uint32_t>(attachmentDescs.size());
		depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		attachmentDescs.push_back(desc);
	}

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = static_cast<uint32_t>(colorRefs.size());
	subpass.pColorAttachments = colorRefs.empty() ? nullptr : colorRefs.data();
	subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;

	VkSubpassDependency dep = {
		.srcSubpass = VK_SUBPASS_EXTERNAL,
		.dstSubpass = 0,
		.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	};
	VkRenderPassCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount = static_cast<uint32_t>(attachmentDescs.size()),
		.pAttachments = attachmentDescs.empty() ? nullptr : attachmentDescs.data(),
		.subpassCount = 1,
		.pSubpasses = &subpass,
		.dependencyCount = 1,
		.pDependencies = &dep,
	};
	VkRenderPass renderPass = VK_NULL_HANDLE;
	VK_CHECK(vkCreateRenderPass(device, &info, nullptr, &renderPass), VK_NULL_HANDLE);
	return renderPass;
}

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

		for (const ColorTarget& target : desc.colorTargets) {
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

	auto compatible_render_pass = create_compatible_render_pass(queue->device, desc);

	VkPipelineCreateFlags2CreateInfo create_flags {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
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
		.renderPass = compatible_render_pass,
		.subpass = 0,
		.basePipelineIndex = -1
	};
	vkCreateGraphicsPipelines(queue->device, VK_NULL_HANDLE, 1, &info, queue->callbacks, &out->pipeline);

	for(auto module: shader_modules)
		vkDestroyShaderModule(queue->device, module, queue->callbacks);
	vkDestroyRenderPass(queue->device, compatible_render_pass, queue->callbacks);

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