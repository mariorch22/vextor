include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(vextor_bench
    core_distance_bench.cpp
    core_sq8_bench.cpp
    index_flat_index_bench.cpp
    index_hnsw_index_bench.cpp
)

target_link_libraries(vextor_bench PRIVATE vextor benchmark::benchmark_main)

add_executable(perf_test perf_test.cpp)
target_link_libraries(perf_test PRIVATE vextor)

# --- SIFT1M benchmark (optional, needs dataset) ---
option(VEXTOR_BUILD_SIFT1M "Build SIFT1M benchmark" OFF)
if(VEXTOR_BUILD_SIFT1M)
    add_subdirectory(sift1m)
endif()
