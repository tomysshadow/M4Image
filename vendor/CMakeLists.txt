add_subdirectory(scope_guard)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(ENABLE_SSE2 ON CACHE BOOL "")
set(ENABLE_SSE4 ON CACHE BOOL "")
add_subdirectory(mango/build)

set(X86_SSE2 ON CACHE BOOL "")
set(X86_SSSE3 ON CACHE BOOL "")

if("${CMAKE_GENERATOR_PLATFORM}" MATCHES "Win32")
	set(X86_MMX ON CACHE BOOL "")
else()
	set(X86_MMX OFF CACHE BOOL "")
endif()

add_subdirectory(pixman-1)
