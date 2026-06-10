cmake_minimum_required(VERSION 3.20)
project(vextor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options ---
option(VEXTOR_BUILD_TESTS "Build unit tests" ON)
option(VEXTOR_BUILD_BENCHMARKS "Build benchmarks" ON)
option(VEXTOR_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(VEXTOR_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

# --- Compiler warnings ---
add_compile_options(-Wall -Wextra -Wpedantic)

# --- Sanitizers ---
if(VEXTOR_ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(VEXTOR_ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()

# --- AVX2 detection ---
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

# --- Library target ---
add_library(vextor STATIC
    src/core/distance.cpp
    src/core/sq8.cpp
    src/store/in_memory_store.cpp
    src/store/mmap_store.cpp
    src/segment/id_mapping.cpp
    src/segment/active_segment.cpp
    src/segment/sealed_segment.cpp
    src/segment/segment_manager.cpp
    src/persistence/serializer.cpp
    src/persistence/loader.cpp
)

target_include_directories(vextor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(vextor PUBLIC cxx_std_20)
set_target_properties(vextor PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    target_compile_options(vextor PRIVATE -mavx2 -mfma)
    target_compile_definitions(vextor PUBLIC VEXTOR_AVX2=1)
endif()

# --- Tests ---
if(VEXTOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Benchmarks ---
if(VEXTOR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# --- Python bindings ---
option(VEXTOR_BUILD_PYTHON "Build Python bindings" OFF)
if(VEXTOR_BUILD_PYTHON)
    add_subdirectory(python)
endif()