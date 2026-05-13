cmake_minimum_required(VERSION 3.20)
project(vexdb VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options ---
option(VEXDB_BUILD_TESTS "Build unit tests" ON)
option(VEXDB_BUILD_BENCHMARKS "Build benchmarks" ON)
option(VEXDB_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(VEXDB_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

# --- Compiler warnings ---
add_compile_options(-Wall -Wextra -Wpedantic)

# --- Sanitizers ---
if(VEXDB_ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(VEXDB_ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()

# --- AVX2 detection ---
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

# --- Library target ---
add_library(vexdb STATIC
    src/core/distance.cpp
    src/core/sq8.cpp
    src/core/vexdb.cpp
    src/store/in_memory_store.cpp
    src/store/mmap_store.cpp
    src/segment/id_mapping.cpp
    src/segment/active_segment.cpp
    src/segment/sealed_segment.cpp
)

target_include_directories(vexdb PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(vexdb PUBLIC cxx_std_20)

if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    target_compile_options(vexdb PRIVATE -mavx2 -mfma)
    target_compile_definitions(vexdb PUBLIC VEXDB_AVX2=1)
endif()

# --- Tests ---
if(VEXDB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Benchmarks ---
if(VEXDB_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()