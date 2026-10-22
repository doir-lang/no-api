find_package(Vulkan REQUIRED)

find_package(vk-bootstrap QUIET)
if(NOT vk-bootstrap_FOUND)
	message(STATUS "vk-bootstrap not found via find_package; fetching from GitHub")
	include(FetchContent)
	FetchContent_Declare(vk-bootstrap
		GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
		GIT_TAG v1.4.357
	)
	FetchContent_MakeAvailable(vk-bootstrap)
endif()

find_package(VulkanMemoryAllocator QUIET)
if(NOT VulkanMemoryAllocator_FOUND)
	message(STATUS "VulkanMemoryAllocator not found via find_package; fetching from GitHub")
	include(FetchContent)
	FetchContent_Declare(VulkanMemoryAllocator
		GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
		GIT_TAG v3.4.0
	)
	FetchContent_MakeAvailable(VulkanMemoryAllocator)
endif()

find_package(volk QUIET)
if(NOT volk_FOUND)
	message(STATUS "volk not found via find_package; fetching from GitHub")
	include(FetchContent)
	FetchContent_Declare(volk
		GIT_REPOSITORY https://github.com/zeux/volk
		GIT_TAG 1.4.350
	)
	FetchContent_MakeAvailable(volk)
endif()

add_library(noapi-vulkan STATIC noapi.cpp)
target_link_libraries(noapi-vulkan PUBLIC
	noapi-slang
	Vulkan::Vulkan
	volk::volk
	vk-bootstrap::vk-bootstrap
	GPUOpen::VulkanMemoryAllocator
)
set_target_properties(noapi-vulkan PROPERTIES CXX_STANDARD 26)
noapi_enable_lto(noapi-vulkan)
target_include_directories(noapi-vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
# Lets anything linking this backend (rather than the other one) pick its shading language and its windowing/instance glue at compile time
target_compile_definitions(noapi-vulkan PUBLIC NOAPI_BACKEND_VULKAN=1)