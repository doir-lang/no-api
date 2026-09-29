
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

# Vulkan gets its surface from glfw itself; only WebGPU needs a helper to bridge the two
if(NOAPI_DEFAULT_BACKEND STREQUAL "webgpu")
	add_library(glfw3webgpu STATIC glfw3webgpu.c)
	target_link_libraries(glfw3webgpu PUBLIC glfw noapi::noapi)

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
		target_link_options(glfw3webgpu PUBLIC "--use-port=contrib.glfw3")
	endif()
endif()

# The API headers are meant to be usable from C too, so one translation unit of the test suite is
# compiled as C and linked in: a header that stops parsing as C, or an entry point whose C
# declaration no longer matches the exported symbol, then fails the build here
add_library(noapi-c-compat OBJECT c_compat.c)
target_link_libraries(noapi-c-compat PRIVATE noapi::noapi)
set_target_properties(noapi-c-compat PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

# stb_image and tiny_obj_loader are header only, with their implementation halves behind a macro.
# One object library turns each on exactly once for whichever tests load assets.
add_library(noapi-test-loaders OBJECT loaders.cpp)
target_include_directories(noapi-test-loaders PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(noapi-test-loaders PROPERTIES CXX_STANDARD 26)

# The files the asset loading tests open at runtime. A native build reads them out of the source
# tree, so TEST_ASSET_DIR points there; Emscripten has no filesystem to read them from, so they are
# packed into the module's virtual one under /assets and TEST_ASSET_DIR points at that instead.
set(NOAPI_TEST_ASSETS
	pexels-70588695-35341664.jpg
	sunflowers_puresky_2k.hdr
	teapot.obj
)

# noapi_add_test(<name> [ASSETS] [C_COMPAT])
#
# Declares <name> as an executable built from <name>.cpp against the selected backend.
#
# ASSETS gives it the loader library and a TEST_ASSET_DIR to find the files above in.
# C_COMPAT links the C translation unit, which only one test needs to carry for the build to be
# checked.
function(noapi_add_test name)
	cmake_parse_arguments(ARG "ASSETS;C_COMPAT" "" "" ${ARGN})

	if(WIN32)
		add_executable(${name} WIN32 ${name}.cpp)
	else()
		add_executable(${name} ${name}.cpp)
	endif()

	# noapi::noapi is whichever backend NOAPI_DEFAULT_BACKEND selected, and carries the
	# NOAPI_BACKEND_VULKAN/NOAPI_BACKEND_WEBGPU define that the tests branch on
	target_link_libraries(${name} PUBLIC glfw noapi::noapi)
	target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	set_target_properties(${name} PROPERTIES CXX_STANDARD 26)

	if(TARGET glfw3webgpu)
		target_link_libraries(${name} PUBLIC glfw3webgpu)
	endif()

	if(ARG_C_COMPAT)
		target_link_libraries(${name} PUBLIC noapi-c-compat)
	endif()

	if(ARG_ASSETS)
		target_link_libraries(${name} PUBLIC noapi-test-loaders)
		if(EMSCRIPTEN)
			target_compile_definitions(${name} PRIVATE TEST_ASSET_DIR="/assets")
			foreach(asset ${NOAPI_TEST_ASSETS})
				target_link_options(${name} PRIVATE
					"SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/${asset}@/assets/${asset}")
			endforeach()
		else()
			target_compile_definitions(${name} PRIVATE TEST_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
		endif()
	endif()

	if(EMSCRIPTEN)
		target_compile_options(${name} PUBLIC "--use-port=contrib.glfw3")
		target_link_options(${name} PUBLIC "--use-port=contrib.glfw3" -sASYNCIFY)
		set_target_properties(${name} PROPERTIES SUFFIX ".html")
	endif()
endfunction()

noapi_add_test(0_basic C_COMPAT)
noapi_add_test(1_texture ASSETS)
noapi_add_test(2_pbr ASSETS)

if(EMSCRIPTEN)
	add_custom_target(serve
		COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR} python3 -m http.server 8080
		COMMENT "Web server has started on port 8080, press ctrl+c when you are finished"
		DEPENDS 0_basic 1_texture 2_pbr
		BYPRODUCTS ${CMAKE_BINARY_DIR}/server_started
	)
endif()
