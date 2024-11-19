project(pixman C ASM)

add_library(${PROJECT_NAME} STATIC "")

target_sources(${PROJECT_NAME}
	PRIVATE
		pixman.c
		pixman-access.c
		pixman-access-accessors.c
		pixman-bits-image.c
		pixman-combine32.c
		pixman-combine-float.c
		pixman-conical-gradient.c
		pixman-filter.c
		pixman-x86.c
		pixman-mips.c
		pixman-arm.c
		pixman-ppc.c
		pixman-edge.c
		pixman-edge-accessors.c
		pixman-fast-path.c
		pixman-glyph.c
		pixman-general.c
		pixman-gradient-walker.c
		pixman-image.c
		pixman-implementation.c
		pixman-linear-gradient.c
		pixman-matrix.c
		pixman-noop.c
		pixman-radial-gradient.c
		pixman-region16.c
		pixman-region32.c
		pixman-solid-fill.c
		pixman-timer.c
		pixman-trap.c
		pixman-utils.c
		
		pixman-mmx.c
		pixman-sse2.c
		pixman-ssse3.c

		pixman-private.h
	PUBLIC
		config.h
		pixman.h
		pixman-arma64-neon-asm.h
		pixman-arm-asm.h
		pixman-arm-common.h
		pixman-arm-neon-asm.h
		pixman-arm-simd-asm.h
		pixman-combine32.h
		pixman-compiler.h
		pixman-edge-imp.h
		pixman-inlines.h
		pixman-mips-dspr2.h
		pixman-mips-dspr2-asm.h
		pixman-version.h
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE -DHAVE_CONFIG_H)
