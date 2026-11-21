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

# Standalone keelcpp libraries (compress, imagefile, fft, random, linalg)
foreach(_keel_pair
    "KEEL:keel"
    "KEEL_COMPRESS:compress"
    "KEEL_IMAGEFILE:imagefile"
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
set(KEEL_IMAGEFILE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
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
        GIT_TAG        17f70b799a9627d06038de3d6bc14efd2400a171
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
        GIT_TAG        1b0dc6aec5f64773de5c54662ff32b4229fa6980
    )
    FetchContent_MakeAvailable(keel_compress)
endif()

if(NOT TARGET keel::imagefile)
    find_package(keel_imagefile QUIET CONFIG)
    if(TARGET keel::keel_imagefile AND NOT TARGET keel::imagefile)
        add_library(keel::imagefile ALIAS keel::keel_imagefile)
    endif()
endif()
if(NOT TARGET keel::imagefile)
    FetchContent_Declare(
        keel_imagefile
        GIT_REPOSITORY https://github.com/keelcpp/imagefile.git
        GIT_TAG        81751baeb333937d8269a75df0fbe330791d94e2
    )
    FetchContent_MakeAvailable(keel_imagefile)
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
        GIT_TAG        740bed99088ffb27d58fa2aa8e70b9d66f59100f
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
        GIT_TAG        4d2b805928c8cb2cdf36aaf7411f8b08a816c2ab
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
        GIT_TAG        bfbb5449d06dae36a3bafe8f80700082d8201b01
    )
    FetchContent_MakeAvailable(keel_linalg)
endif()

if(NOT TARGET keel::specfun)
    find_package(keel_specfun QUIET CONFIG)
    if(TARGET keel::keel_specfun AND NOT TARGET keel::specfun)
        add_library(keel::specfun ALIAS keel::keel_specfun)
    endif()
endif()
if(NOT TARGET keel::specfun)
    set(KEEL_SPECFUN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/specfun/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_SPECFUN)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_SPECFUN "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/specfun" CACHE PATH "Local keel_specfun checkout")
    endif()
    FetchContent_Declare(
        keel_specfun
        GIT_REPOSITORY https://github.com/keelcpp/specfun.git
        GIT_TAG        243bfb4da65004dc0a8bc76e20eed6f16f08d470
    )
    FetchContent_MakeAvailable(keel_specfun)
endif()

if(NOT TARGET keel::geom)
    find_package(keel_geom QUIET CONFIG)
    if(TARGET keel::keel_geom AND NOT TARGET keel::geom)
        add_library(keel::geom ALIAS keel::keel_geom)
    endif()
endif()
if(NOT TARGET keel::geom)
    set(KEEL_GEOM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/geom/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_GEOM)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_GEOM "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/geom" CACHE PATH "Local keel_geom checkout")
    endif()
    FetchContent_Declare(
        keel_geom
        GIT_REPOSITORY https://github.com/keelcpp/geom.git
        GIT_TAG        f7baf480221c40328637c025d1d1ff5fdcae7c00
    )
    FetchContent_MakeAvailable(keel_geom)
endif()

if(NOT TARGET keel::interp)
    find_package(keel_interp QUIET CONFIG)
    if(TARGET keel::keel_interp AND NOT TARGET keel::interp)
        add_library(keel::interp ALIAS keel::keel_interp)
    endif()
endif()
if(NOT TARGET keel::interp)
    set(KEEL_INTERP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/interp/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_INTERP)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_INTERP "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/interp" CACHE PATH "Local keel_interp checkout")
    endif()
    FetchContent_Declare(
        keel_interp
        GIT_REPOSITORY https://github.com/keelcpp/interp.git
        GIT_TAG        6d56cfb3e6c5d133593583d3b07a141ee0b545b8
    )
    FetchContent_MakeAvailable(keel_interp)
endif()

if(NOT TARGET keel::ode)
    find_package(keel_ode QUIET CONFIG)
    if(TARGET keel::keel_ode AND NOT TARGET keel::ode)
        add_library(keel::ode ALIAS keel::keel_ode)
    endif()
endif()
if(NOT TARGET keel::ode)
    set(KEEL_ODE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/ode/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_ODE)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_ODE "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/ode" CACHE PATH "Local keel_ode checkout")
    endif()
    FetchContent_Declare(
        keel_ode
        GIT_REPOSITORY https://github.com/keelcpp/ode.git
        GIT_TAG        9eb2ff49d5737863873bb49929087b7fbcf21dd6
    )
    FetchContent_MakeAvailable(keel_ode)
endif()

if(NOT TARGET keel::quad)
    find_package(keel_quad QUIET CONFIG)
    if(TARGET keel::keel_quad AND NOT TARGET keel::quad)
        add_library(keel::quad ALIAS keel::keel_quad)
    endif()
endif()
if(NOT TARGET keel::quad)
    set(KEEL_QUAD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/quad/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_QUAD)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_QUAD "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/quad" CACHE PATH "Local keel_quad checkout")
    endif()
    FetchContent_Declare(
        keel_quad
        GIT_REPOSITORY https://github.com/keelcpp/quad.git
        GIT_TAG        d17118196cebc40e63b23b876ad8268abf877098
    )
    FetchContent_MakeAvailable(keel_quad)
endif()

if(NOT TARGET keel::optim)
    find_package(keel_optim QUIET CONFIG)
    if(TARGET keel::keel_optim AND NOT TARGET keel::optim)
        add_library(keel::optim ALIAS keel::keel_optim)
    endif()
endif()
if(NOT TARGET keel::optim)
    set(KEEL_OPTIM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/optim/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_OPTIM)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_OPTIM "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/optim" CACHE PATH "Local keel_optim checkout")
    endif()
    FetchContent_Declare(
        keel_optim
        GIT_REPOSITORY https://github.com/keelcpp/optim.git
        GIT_TAG        d76704f92313b5f7132de937710b2214a088eb48
    )
    FetchContent_MakeAvailable(keel_optim)
endif()

if(NOT TARGET keel::poly)
    find_package(keel_poly QUIET CONFIG)
    if(TARGET keel::keel_poly AND NOT TARGET keel::poly)
        add_library(keel::poly ALIAS keel::keel_poly)
    endif()
endif()
if(NOT TARGET keel::poly)
    set(KEEL_POLY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/poly/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_POLY)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_POLY "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/poly" CACHE PATH "Local keel_poly checkout")
    endif()
    FetchContent_Declare(
        keel_poly
        GIT_REPOSITORY https://github.com/keelcpp/poly.git
        GIT_TAG        87429cfe8d8a3353b24ffdfa7385baba42c049ee
    )
    FetchContent_MakeAvailable(keel_poly)
endif()

if(NOT TARGET keel::signal)
    find_package(keel_signal QUIET CONFIG)
    if(TARGET keel::keel_signal AND NOT TARGET keel::signal)
        add_library(keel::signal ALIAS keel::keel_signal)
    endif()
endif()
if(NOT TARGET keel::signal)
    set(KEEL_SIGNAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/signal/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_SIGNAL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_SIGNAL "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/signal" CACHE PATH "Local keel_signal checkout")
    endif()
    FetchContent_Declare(
        keel_signal
        GIT_REPOSITORY https://github.com/keelcpp/signal.git
        GIT_TAG        c888a92ee4d381d9bb0178d4f11cf9435c8f1c82
    )
    FetchContent_MakeAvailable(keel_signal)
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
    PUBLIC numkit_value keel::random keel::linalg keel::specfun keel::geom keel::interp keel::ode keel::quad keel::optim keel::poly keel::signal
    PRIVATE hwy hwy_contrib keel::compress keel::imagefile keel::fft
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
