#include <memory>
#include <volk.h>
#include <vk_mem_alloc.h>

#include "noapi.hpp"
#include "common.hpp"

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added"


GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, VkSurfaceKHR surface, const GpuSurfaceDescriptor& desc) {
	auto out = (GpuSurface*)queue->cpu_allocator(nullptr, sizeof(GpuSurface));
	new(out) GpuSurface {
		.surface = surface,
	};

	gpuSurfaceReconfigureEXT(queue, out, desc);
	return out;
}

void gpuDestroySurfaceNoSemaphores(GpuQueue* queue, GpuSurface* surface) { 
	for(auto view: surface->image_views)
		vkDestroyImageView(queue->device, view, queue->callbacks);
	if(surface->swapchain)
		vkb::destroy_swapchain(*surface->swapchain);
}

void gpuFreeSurfaceEXT(GpuQueue* queue, GpuSurface* surface) {
	for(auto semaphore: surface->image_available_semaphores)
		vkDestroySemaphore(queue->device, semaphore, queue->callbacks);
	for(auto semaphore: surface->render_finished_semaphores)
		vkDestroySemaphore(queue->device, semaphore, queue->callbacks);
	gpuDestroySurfaceNoSemaphores(queue, surface);
}

void gpuSurfaceReconfigureEXT(GpuQueue* queue, GpuSurface* surface, const GpuSurfaceDescriptor& desc) {
	surface->descriptor = desc;
	surface->descriptor.texture.type = TEXTURE_2D;
	surface->descriptor.texture.mipCount = 1;
	surface->descriptor.texture.sampleCount = 1;

	auto builder = vkb::SwapchainBuilder(queue->gpu, queue->device, surface->surface, queue->queue_family);
	if(surface->swapchain)
		builder.set_old_swapchain(*surface->swapchain);
	switch(desc.presentMode){
	break; case PRESENT_MODE_IMMEDIATE:
		builder.set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR);
	break; case PRESENT_MODE_FIFO:
		builder.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR);
	break; case PRESENT_MODE_FIFO_RELAXED:
		builder.set_desired_present_mode(VK_PRESENT_MODE_FIFO_RELAXED_KHR);
	break; case PRESENT_MODE_MAILBOX:
		builder.set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR);
	break; case PRESENT_MODE_BEST_AVAILABLE:
		builder.use_default_present_mode_selection();
	}
	assert(surface->descriptor.texture.dimensions.z == 1);
	auto swap = builder.set_desired_extent(surface->descriptor.texture.dimensions.x, surface->descriptor.texture.dimensions.y)
		.set_allocation_callbacks(queue->callbacks)
		.set_composite_alpha_flags(surface->descriptor.opaque ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
		.set_desired_format({
			GPU::detail::format2vulkan(surface->descriptor.texture.format),
			VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
		})
		.set_image_array_layer_count(surface->descriptor.texture.layerCount)
		.add_image_usage_flags(GPU::detail::usage2vulkan(surface->descriptor.texture.usage))
		.build();
	if(!swap) errno = swap.error().value();

	gpuDestroySurfaceNoSemaphores(queue, surface);
	surface->swapchain = std::make_shared<vkb::Swapchain>(std::move(*swap));
	std::vector<VkImage> images;
	std::tie(images, surface->image_views) = surface->swapchain->get_images_and_image_views().value();

	surface->images.resize(images.size());
	for(size_t i = 0; i < images.size(); ++i)
		surface->images[i] = GpuTexture {
			images[i],
			surface->image_views[i],
			surface->descriptor.texture
		};
}

GpuSurfaceDescriptor gpuSurfaceGetConfigurationEXT(const GpuSurface* surface) {
	return surface->descriptor;
}

const GpuTexture* gpuSurfaceNextTextureEXT(GpuQueue* queue, GpuSurface* surface) {
	if(surface->image_available_semaphores.size() != surface->images.size()) {
		for(auto semaphore: surface->image_available_semaphores)
			vkDestroySemaphore(queue->device, semaphore, queue->callbacks);

		surface->image_available_semaphores.resize(surface->images.size());
		for(auto& semaphore: surface->image_available_semaphores) {
			VkSemaphoreCreateInfo info { info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
			vkCreateSemaphore(queue->device, &info, queue->callbacks, &semaphore);
		}
		surface->semaphore_counter = -1;
	}

	surface->semaphore_counter = (surface->semaphore_counter + 1) % surface->image_available_semaphores.size();
	VK_CHECK(vkAcquireNextImageKHR(queue->device, surface->swapchain->swapchain, UINT64_MAX, surface->image_available_semaphores[surface->semaphore_counter], VK_NULL_HANDLE, &surface->current_image), nullptr);
	
	surface->images[surface->current_image].available_semaphore = surface->image_available_semaphores[surface->semaphore_counter];
	return &surface->images[surface->current_image];
}

void gpuSurfacePresentEXT(GpuQueue* queue, GpuSurface* surface, uint64_t wait_submission_index /*= NO_SUBMISSION_WAIT */) {
	if(wait_submission_index != NO_SUBMISSION_WAIT) {
		if(surface->render_finished_semaphores.size() != surface->images.size()) {
			for(auto semaphore: surface->render_finished_semaphores)
				vkDestroySemaphore(queue->device, semaphore, queue->callbacks);

			surface->render_finished_semaphores.resize(surface->images.size());
			for(auto& semaphore: surface->render_finished_semaphores) {
				VkSemaphoreCreateInfo info { info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
				vkCreateSemaphore(queue->device, &info, queue->callbacks, &semaphore);
			}
		}

		// Launch an empty/no command buffer that waits for the timeline semaphore and then signals the render finished semaphore
		VkSemaphoreSubmitInfo wait {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = queue->command_submission_timeline_semaphore,
			.value = wait_submission_index,
			.stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
		};
		VkSemaphoreSubmitInfo signal {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = surface->render_finished_semaphores[surface->current_image],
			.stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
		};
		VkSubmitInfo2 info = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.waitSemaphoreInfoCount = 1,
			.pWaitSemaphoreInfos = &wait,
			.signalSemaphoreInfoCount = 1,
			.pSignalSemaphoreInfos = &signal
		};
		VK_CHECK(vkQueueSubmit2(queue->queue, 1, &info, VK_NULL_HANDLE), /*nothing*/);
	}

	VkPresentInfoKHR info = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.swapchainCount = 1,
		.pSwapchains = &surface->swapchain->swapchain,
		.pImageIndices = &surface->current_image,
	};
	if(wait_submission_index != NO_SUBMISSION_WAIT) {
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &surface->render_finished_semaphores[surface->current_image];
	}
	VK_CHECK(vkQueuePresentKHR(queue->queue, &info), /*nothing*/);
}