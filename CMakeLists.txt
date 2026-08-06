cmake_minimum_required(VERSION 3.21)
project(NoAPI CXX C)

set(CMAKE_CXX_STANDARD 26)

# option(WEBGPU_BUILD_FROM_SOURCE "Tell webgpu distribution that we want to build dawn ourselves" ON)
# if(NOT EMSCRIPTEN)
# 	set(WEBGPU_BACKEND "DAWN" CACHE STRING "Set default")
# endif()

# add_subdirectory(WebGPU-distribution)
add_subdirectory(vulkan)

find_package(glfw QUIET)
if (NOT glfw_FOUND AND NOT TARGET glfw)
	message(STATUS "glfw not found via find_package; fetching from GitHub")
	include(FetchContent)
	FetchContent_Declare(glfw
		GIT_REPOSITORY https://github.com/glfw/glfw
		GIT_TAG 3.5.1
	)
	FetchContent_MakeAvailable(glfw)
endif()

if(WIN32)
	add_executable(test WIN32 test/test.cpp)
else()
	add_executable(test test/test.cpp)
endif()
target_link_libraries(test PUBLIC glfw noapi-vulkan)

if (GLFW_BUILD_COCOA)
	target_compile_definitions(test PRIVATE _GLFW_COCOA)
	target_compile_options(test PRIVATE -x objective-c)
	target_link_libraries(test PRIVATE "-framework Cocoa" "-framework CoreVideo" "-framework IOKit" "-framework QuartzCore")
endif()
if (GLFW_BUILD_WIN32)
	target_compile_definitions(test PRIVATE _GLFW_WIN32)
endif()
if (GLFW_BUILD_X11)
	target_compile_definitions(test PRIVATE _GLFW_X11)
endif()
if (GLFW_BUILD_WAYLAND)
	target_compile_definitions(test PRIVATE _GLFW_WAYLAND)
endif()

if(EMSCRIPTEN)
	target_compile_options(test PUBLIC "--use-port=contrib.glfw3")
	target_link_options(test PUBLIC "--use-port=contrib.glfw3" -sASYNCIFY)
	set_target_properties(test PROPERTIES SUFFIX ".html")

	add_custom_target(serve
		COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR} python3 -m http.server 8080
		COMMENT "Web server has started on port 8080, press ctrl+c when you are finished"
		DEPENDS test
		BYPRODUCTS ${CMAKE_BINARY_DIR}/server_started
	) 
endif()
