#pragma once

// Included by the examples tools/doctest.py generates, from C and C++ alike.
#ifdef NOAPI_BACKEND_VULKAN
	#include <volk.h>
#endif

#ifdef __cplusplus
	#ifdef NOAPI_BACKEND_VULKAN
		#include <vulkan/noapi.hpp>
	#else
		#include <webgpu/noapi.hpp>
	#endif
#else
	#ifdef NOAPI_BACKEND_VULKAN
		#include <vulkan/noapi.h>
	#else
		#include <webgpu/noapi.h>
	#endif
#endif

#include <assert.h>
#include <string.h>
#ifdef __cplusplus
	#include <filesystem>
	#include <fstream>
	#include <string>
#endif

// The backend objects the headless queue was created from, handed to every example as `backend`.
#ifdef NOAPI_BACKEND_VULKAN
	typedef GpuVulkanDefault GpuDoctestBackend;
#else
	typedef GpuWebGPUDefault GpuDoctestBackend;
#endif

NOAPI_EXTERN_C_BEGIN

GpuQueue* noapi_doctest_queue(void);
GpuDoctestBackend noapi_doctest_backend(void);
void noapi_doctest_begin(void);
void noapi_doctest_end(void);
void noapi_doctest_fail(const char* file, int line, const char* expression);

NOAPI_EXTERN_C_END

// A failed assert ends its example by throwing, so the other examples still run; the C examples are built to let that unwind through them.
#undef assert
#define assert(expression) ((expression) ? (void)0 : noapi_doctest_fail(__FILE__, __LINE__, #expression))
