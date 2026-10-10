cmake_minimum_required(VERSION 3.20)
project(numkit_ops VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NUMKIT_WITH_SIMD "Enable Google Highway SIMD acceleration" ON)
if(DEFINED NK_WITH_SIMD)
    set(NUMKIT_WITH_SIMD ${NK_WITH_SIMD})
endif()

option(NUMKIT_WITH_THREADS "Enable multi-threaded parallel execution" ON)
if(DEFINED NK_WITH_THREADS)
    set(NUMKIT_WITH_THREADS ${NK_WITH_THREADS})
endif()

include(FetchContent)

# Dependency: numkit/value
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../value/CMakeLists.txt")
    message(STATUS "Using local numkit/value from ${CMAKE_CURRENT_SOURCE_DIR}/../value")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../value" _deps/numkit_value)
else()
    FetchContent_Declare(
        numkit_value
        GIT_REPOSITORY https://github.com/numkit/value.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(numkit_value)
endif()

# Google Highway (SIMD & cache control)
FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.2.0
)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)

set(NUMKIT_OPS_SOURCES
    src/matlab_mt19937.cpp
    src/rng.cpp
    src/la_solve.cpp
    src/compression/deflate.cpp
    src/blas/blas1_highway.cpp
    src/blas/blas2_highway.cpp
    src/blas/blas3_highway.cpp
    src/root_solve.cpp
    src/conv.cpp
    src/iir_filter.cpp
    src/quantile_select.cpp
    src/fused/fused_complex.cpp
    src/binary_scalar.cpp
    src/fused/fused_scalar.cpp
    src/kernels.cpp
)

if(NUMKIT_WITH_THREADS)
    list(APPEND NUMKIT_OPS_SOURCES src/thread_pool.cpp)
endif()

if(NUMKIT_WITH_SIMD)
    list(APPEND NUMKIT_OPS_SOURCES
        src/binary_ops_highway.cpp
        src/compare_highway.cpp
        src/fused/fused_affine_clamp_highway.cpp
        src/fused/fused_affine_highway.cpp
        src/fused/fused_axpby_highway.cpp
        src/fused/fused_shift_scale_highway.cpp
        src/fused/fused_abs_highway.cpp
        src/fused/fused_unary_affine_highway.cpp
        src/fused/fused_sq_highway.cpp
        src/fused/fused_soft_threshold_highway.cpp
        src/fused/fused_trans_affine_highway.cpp
        src/fft/fft_highway.cpp
        src/fft/fft_r2_highway.cpp
        src/fft/fft_r4_highway.cpp
        src/fft/fft_stockham_highway.cpp
        src/fft/fft_r2_soa_highway.cpp
        src/fft/fft_r4_soa_highway.cpp
        src/nan_reductions_highway.cpp
    )
else()
    list(APPEND NUMKIT_OPS_SOURCES
        src/binary_ops_portable.cpp
        src/compare_portable.cpp
        src/fused/fused_kernels_portable.cpp
        src/fft/fft_portable.cpp
        src/nan_reductions_portable.cpp
    )
endif()

add_library(numkit_ops STATIC ${NUMKIT_OPS_SOURCES})

add_library(numkit::ops ALIAS numkit_ops)
add_library(nk_ops ALIAS numkit_ops)
add_library(nk::ops ALIAS numkit_ops)

target_include_directories(numkit_ops
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(numkit_ops PUBLIC numkit_value PRIVATE hwy hwy_contrib)

if(NUMKIT_WITH_SIMD)
    target_compile_definitions(numkit_ops PUBLIC NUMKIT_WITH_SIMD=1 NK_WITH_SIMD=1)
endif()

if(NUMKIT_WITH_THREADS)
    find_package(Threads REQUIRED)
    target_compile_definitions(numkit_ops PUBLIC NUMKIT_WITH_THREADS=1 NK_WITH_THREADS=1)
    target_link_libraries(numkit_ops PUBLIC Threads::Threads)
endif()

if(EMSCRIPTEN)
    target_compile_options(numkit_ops PRIVATE -fexceptions)
endif()

if(MSVC)
    target_compile_options(numkit_ops PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_ops PRIVATE -Wall -Wextra -fPIC)
endif()

include(GNUInstallDirs)
install(TARGETS numkit_ops
    EXPORT numkit_ops_targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
