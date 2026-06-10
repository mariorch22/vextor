include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(vextor_tests
    core_types_test.cpp
    core_distance_test.cpp
    core_sq8_test.cpp
    store_in_memory_store_test.cpp
    store_mmap_store_test.cpp
    index_flat_index_test.cpp
    index_hnsw_index_test.cpp
    segment_id_mapping_test.cpp
    segment_active_segment_test.cpp
    segment_sealed_segment_test.cpp
    segment_segment_manager_test.cpp
    persistence_test.cpp
    vextor_public_api_test.cpp
)

target_link_libraries(vextor_tests PRIVATE vextor GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(vextor_tests)