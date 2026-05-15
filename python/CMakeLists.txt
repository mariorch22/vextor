find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.4.0
)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(vexdb_python
    bindings.cpp
)
target_link_libraries(vexdb_python PRIVATE vexdb)
set_target_properties(vexdb_python PROPERTIES OUTPUT_NAME "vexdb")

# nanobind uses non-standard extensions that trigger -Wpedantic; suppress on third-party targets.
# nanobind-static is created by nanobind_add_module, so these must come after.
target_compile_options(nanobind-static PRIVATE -w)
target_compile_options(vexdb_python PRIVATE -Wno-nested-anon-types -Wno-zero-length-array)