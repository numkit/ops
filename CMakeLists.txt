cmake_minimum_required(VERSION 3.20)
project(numkit_ops VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NUMKIT_WITH_SIMD "Enable Google Highway SIMD acceleration" ON)
option(NUMKIT_WITH_THREADS "Enable multi-threaded parallel execution" ON)

include(FetchContent)

# Dependency: numkit/value
if(NOT TARGET numkit_value)
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
endif()

# Google Highway (SIMD & cache control)
FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.2.0
    GIT_SHALLOW TRUE
)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(_numkit_saved_warn_deprecated "${CMAKE_WARN_DEPRECATED}")
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)
if(DEFINED _numkit_saved_warn_deprecated AND NOT "${_numkit_saved_warn_deprecated}" STREQUAL "")
    set(CMAKE_WARN_DEPRECATED "${_numkit_saved_warn_deprecated}" CACHE BOOL "" FORCE)
endif()

# Standalone keelcpp libraries (compress, image, fft, random, linalg)
foreach(_keel_pair
    "KEEL:keel"
    "KEEL_COMPRESS:compress"
    "KEEL_IMAGE:image"
    "KEEL_FFT:fft"
    "KEEL_RANDOM:random"
    "KEEL_LINALG:linalg"
)
    string(REPLACE ":" ";" _keel_parts "${_keel_pair}")
    list(GET _keel_parts 0 _keel_var)
    list(GET _keel_parts 1 _keel_dir)
    if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_${_keel_var})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/${_keel_dir}/CMakeLists.txt")
            set(FETCHCONTENT_SOURCE_DIR_${_keel_var}
                "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/${_keel_dir}"
                CACHE PATH "")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../keelcpp/${_keel_dir}/CMakeLists.txt")
            set(FETCHCONTENT_SOURCE_DIR_${_keel_var}
                "${CMAKE_CURRENT_SOURCE_DIR}/../../keelcpp/${_keel_dir}"
                CACHE PATH "")
        endif()
    endif()
endforeach()

set(KEEL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KEEL_COMPRESS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KEEL_IMAGE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KEEL_FFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KEEL_RANDOM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KEEL_LINALG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KEEL_FFT_WITH_HIGHWAY ${NUMKIT_WITH_SIMD} CACHE BOOL "" FORCE)
set(KEEL_LINALG_WITH_HIGHWAY ${NUMKIT_WITH_SIMD} CACHE BOOL "" FORCE)
set(KEEL_LINALG_WITH_THREADS ${NUMKIT_WITH_THREADS} CACHE BOOL "" FORCE)

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        f9fcebfba18d98d2c14c8e62a1913412f1299220
    )
    FetchContent_MakeAvailable(keel)
endif()

if(NOT TARGET keel::compress)
    find_package(keel_compress QUIET CONFIG)
    if(TARGET keel::keel_compress AND NOT TARGET keel::compress)
        add_library(keel::compress ALIAS keel::keel_compress)
    endif()
endif()
if(NOT TARGET keel::compress)
    FetchContent_Declare(
        keel_compress
        GIT_REPOSITORY https://github.com/keelcpp/compress.git
        GIT_TAG        e426c71ed8d267e488119272c0e066e1da2f720e
    )
    FetchContent_MakeAvailable(keel_compress)
endif()

if(NOT TARGET keel::image)
    find_package(keel_image QUIET CONFIG)
    if(TARGET keel::keel_image AND NOT TARGET keel::image)
        add_library(keel::image ALIAS keel::keel_image)
    endif()
endif()
if(NOT TARGET keel::image)
    FetchContent_Declare(
        keel_image
        GIT_REPOSITORY https://github.com/keelcpp/image.git
        GIT_TAG        bfbbf3bd9daa8890880e28381932d6307aefde2c
    )
    FetchContent_MakeAvailable(keel_image)
endif()

if(NOT TARGET keel::fft)
    find_package(keel_fft QUIET CONFIG)
    if(TARGET keel::keel_fft AND NOT TARGET keel::fft)
        add_library(keel::fft ALIAS keel::keel_fft)
    endif()
endif()
if(NOT TARGET keel::fft)
    FetchContent_Declare(
        keel_fft
        GIT_REPOSITORY https://github.com/keelcpp/fft.git
        GIT_TAG        91357427ef2492fda4d1906add6bcd3fe2d3c260
    )
    FetchContent_MakeAvailable(keel_fft)
endif()

if(NOT TARGET keel::random)
    find_package(keel_random QUIET CONFIG)
    if(TARGET keel::keel_random AND NOT TARGET keel::random)
        add_library(keel::random ALIAS keel::keel_random)
    endif()
endif()
if(NOT TARGET keel::random)
    FetchContent_Declare(
        keel_random
        GIT_REPOSITORY https://github.com/keelcpp/random.git
        GIT_TAG        fce59d71cea1c409c04dd0c62b3d2cf62e5d28aa
    )
    FetchContent_MakeAvailable(keel_random)
endif()

if(NOT TARGET keel::linalg)
    find_package(keel_linalg QUIET CONFIG)
    if(TARGET keel::keel_linalg AND NOT TARGET keel::linalg)
        add_library(keel::linalg ALIAS keel::keel_linalg)
    endif()
endif()
if(NOT TARGET keel::linalg)
    FetchContent_Declare(
        keel_linalg
        GIT_REPOSITORY https://github.com/keelcpp/linalg.git
        GIT_TAG        e6e29ad227ff97d01e837e281ab6a7d9c440f075
    )
    FetchContent_MakeAvailable(keel_linalg)
endif()

set(NUMKIT_OPS_SOURCES
    src/rng/matlab_mt19937.cpp
    src/rng/rng.cpp
    src/compression/deflate.cpp
    src/dsp/conv.cpp
    src/dsp/iir_filter.cpp
    src/optim/root_solve.cpp
    src/kernels/quantile_select.cpp
    src/kernels/binary_scalar.cpp
    src/kernels/kernels.cpp
    src/fused/fused_complex.cpp
    src/fused/fused_scalar.cpp
)

if(NUMKIT_WITH_SIMD)
    list(APPEND NUMKIT_OPS_SOURCES
        src/kernels/binary_ops_highway.cpp
        src/kernels/compare_highway.cpp
        src/kernels/nan_reductions_highway.cpp
        src/fused/fused_affine_clamp_highway.cpp
        src/fused/fused_affine_highway.cpp
        src/fused/fused_axpby_highway.cpp
        src/fused/fused_shift_scale_highway.cpp
        src/fused/fused_abs_highway.cpp
        src/fused/fused_unary_affine_highway.cpp
        src/fused/fused_sq_highway.cpp
        src/fused/fused_soft_threshold_highway.cpp
        src/fused/fused_trans_affine_highway.cpp
    )
else()
    list(APPEND NUMKIT_OPS_SOURCES
        src/kernels/binary_ops_portable.cpp
        src/kernels/compare_portable.cpp
        src/kernels/nan_reductions_portable.cpp
        src/fused/fused_kernels_portable.cpp
    )
endif()

file(GLOB_RECURSE NUMKIT_COMPUTE_DOMAIN_SOURCES
    "src/builtin/*.cpp"
    "src/linalg/*.cpp"
    "src/signal/*.cpp"
    "src/ode/*.cpp"
    "src/optim/local/*.cpp"
    "src/image_io/*.cpp"
)
if(NOT NUMKIT_WITH_SIMD)
    list(FILTER NUMKIT_COMPUTE_DOMAIN_SOURCES EXCLUDE REGEX ".*_highway\\.cpp$")
else()
    list(FILTER NUMKIT_COMPUTE_DOMAIN_SOURCES EXCLUDE REGEX ".*_portable\\.cpp$")
endif()
list(APPEND NUMKIT_OPS_SOURCES ${NUMKIT_COMPUTE_DOMAIN_SOURCES})

add_library(numkit_ops STATIC ${NUMKIT_OPS_SOURCES})

add_library(numkit::ops ALIAS numkit_ops)

target_include_directories(numkit_ops
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/include/numkit/builtin/detail
        ${CMAKE_CURRENT_SOURCE_DIR}/include/numkit/builtin/detail/datafun
        ${CMAKE_CURRENT_SOURCE_DIR}/include/numkit/builtin/detail/datatypes
        ${CMAKE_CURRENT_SOURCE_DIR}/include/numkit/builtin/detail/elfun
        ${CMAKE_CURRENT_SOURCE_DIR}/include/numkit/builtin/detail/elmat
        ${CMAKE_CURRENT_SOURCE_DIR}/include/numkit/builtin/detail/ops
        ${CMAKE_CURRENT_SOURCE_DIR}/include/numkit/builtin/detail/polyfun
        ${CMAKE_CURRENT_SOURCE_DIR}/include/numkit/builtin/detail/specfun
        ${CMAKE_CURRENT_SOURCE_DIR}/include/numkit/builtin/detail/strfun
        ${CMAKE_CURRENT_SOURCE_DIR}/include/numkit/linalg/detail
)

target_link_libraries(numkit_ops
    PUBLIC numkit_value keel::random keel::linalg
    PRIVATE hwy hwy_contrib keel::compress keel::image keel::fft
)

if(NUMKIT_WITH_SIMD)
    target_compile_definitions(numkit_ops PUBLIC NUMKIT_WITH_SIMD=1)
endif()

if(NUMKIT_WITH_THREADS)
    find_package(Threads REQUIRED)
    target_compile_definitions(numkit_ops PUBLIC NUMKIT_WITH_THREADS=1)
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
