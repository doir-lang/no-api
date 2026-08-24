
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

add_library(glfw3webgpu STATIC glfw3webgpu.c)
if(WIN32)
	add_executable(test WIN32 test.cpp)
else()
	add_executable(test test.cpp)
endif()
target_link_libraries(glfw3webgpu PUBLIC glfw noapi::noapi)
target_link_libraries(test PUBLIC glfw3webgpu)
set_target_properties(test PROPERTIES CXX_STANDARD 26)

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

if(EMSCRIPTEN)
	target_compile_options(glfw3webgpu PUBLIC "--use-port=contrib.glfw3")
	target_link_options(glfw3webgpu PUBLIC "--use-port=contrib.glfw3" -sASYNCIFY)
	set_target_properties(test PROPERTIES SUFFIX ".html")

	add_custom_target(serve
		COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR} python3 -m http.server 8080
		COMMENT "Web server has started on port 8080, press ctrl+c when you are finished"
		DEPENDS test
		BYPRODUCTS ${CMAKE_BINARY_DIR}/server_started
	)
endif()
