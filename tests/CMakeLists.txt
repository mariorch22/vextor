include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(vexdb_tests
    core_types_test.cpp
)

target_link_libraries(vexdb_tests PRIVATE vexdb GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(vexdb_tests)