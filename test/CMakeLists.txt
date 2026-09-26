
if(NOT EMSCRIPTEN)
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
endif()

if(WIN32)
	add_executable(test WIN32 test.cpp)
else()
	add_executable(test test.cpp)
endif()
# noapi::noapi is whichever backend NOAPI_DEFAULT_BACKEND selected, and carries the
# NOAPI_BACKEND_VULKAN/NOAPI_BACKEND_WEBGPU define that test.cpp branches on
target_link_libraries(test PUBLIC glfw noapi::noapi)
set_target_properties(test PROPERTIES CXX_STANDARD 26)

# The API headers are meant to be usable from C too, so one translation unit of the test is
# compiled as C and linked in: a header that stops parsing as C, or an entry point whose C
# declaration no longer matches the exported symbol, then fails the build here
add_library(noapi-c-compat OBJECT c_compat.c)
target_link_libraries(noapi-c-compat PRIVATE noapi::noapi)
set_target_properties(noapi-c-compat PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_link_libraries(test PUBLIC noapi-c-compat)

# Vulkan gets its surface from glfw itself; only WebGPU needs a helper to bridge the two
if(NOAPI_DEFAULT_BACKEND STREQUAL "webgpu")
	add_library(glfw3webgpu STATIC glfw3webgpu.c)
	target_link_libraries(glfw3webgpu PUBLIC glfw noapi::noapi)
	target_link_libraries(test PUBLIC glfw3webgpu)

	if (GLFW_BUILD_COCOA)
		target_compile_definitions(glfw3webgpu PRIVATE _GLFW_COCOA)
		target_compile_options(glfw3webgpu PRIVATE -x objective-c)
		target_link_libraries(glfw3webgpu PRIVATE "-framework Cocoa" "-framework CoreVideo" "-framework IOKit" "-framework QuartzCore")
	endif()
	if (GLFW_BUILD_WIN32)
		target_compile_definitions(glfw3webgpu PRIVATE _GLFW_WIN32)
	endif()
	if (GLFW_BUILD_X11)
		target_compile_definitions(glfw3webgpu PRIVATE _GLFW_X11)
	endif()
	if (GLFW_BUILD_WAYLAND)
		target_compile_definitions(glfw3webgpu PRIVATE _GLFW_WAYLAND)
	endif()
endif()

if(EMSCRIPTEN)
	foreach(target test glfw3webgpu)
		if(TARGET ${target})
			target_compile_options(${target} PUBLIC "--use-port=contrib.glfw3")
			target_link_options(${target} PUBLIC "--use-port=contrib.glfw3" -sASYNCIFY)
		endif()
	endforeach()
	set_target_properties(test PROPERTIES SUFFIX ".html")

	add_custom_target(serve
		COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR} python3 -m http.server 8080
		COMMENT "Web server has started on port 8080, press ctrl+c when you are finished"
		DEPENDS test
		BYPRODUCTS ${CMAKE_BINARY_DIR}/server_started
	)
endif()
