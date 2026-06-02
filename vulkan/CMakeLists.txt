find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)

find_package(vk-bootstrap QUIET)
if (NOT vk-bootstrap_FOUND)
	message(STATUS "vk-bootstrap not found via find_package; fetching from GitHub")
	include(FetchContent)
	FetchContent_Declare(vk-bootstrap
		GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
		GIT_TAG v1.3.295
	)
	FetchContent_MakeAvailable(vk-bootstrap)
endif()
 
find_package(VulkanMemoryAllocator QUIET)
if (NOT VulkanMemoryAllocator_FOUND)
	message(STATUS "VulkanMemoryAllocator not found via find_package; fetching from GitHub")
	include(FetchContent)
	FetchContent_Declare(VulkanMemoryAllocator
		GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
		GIT_TAG v3.1.0
	)
	FetchContent_MakeAvailable(VulkanMemoryAllocator)
endif()

find_package(volk QUIET)
if (NOT volk_FOUND)
    message(STATUS "volk not found via find_package; fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(volk
        GIT_REPOSITORY https://github.com/zeux/volk
        GIT_TAG 1.3.295
    )
    FetchContent_MakeAvailable(volk)
endif()

add_library(noapi-vulkan INTERFACE)
target_link_libraries(noapi-vulkan INTERFACE
	volk::volk
	vk-bootstrap::vk-bootstrap
	GPUOpen::VulkanMemoryAllocator
	glslang::glslang
	glslang::glslang-default-resource-limits
	glslang::SPIRV
)