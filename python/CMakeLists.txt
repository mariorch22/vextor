# Development.SABIModule is required for STABLE_ABI builds (CPython >= 3.12);
# optional so older interpreters still build a regular module.
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)

# Prefer an installed nanobind (provided by the pip build environment via
# pyproject.toml); fall back to FetchContent for standalone CMake builds.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    ERROR_QUIET
    RESULT_VARIABLE nanobind_lookup_result
)
if(nanobind_lookup_result EQUAL 0)
    find_package(nanobind CONFIG REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.4.0
    )
    FetchContent_MakeAvailable(nanobind)
endif()

# STABLE_ABI builds an abi3 module against CPython >= 3.12 and is ignored
# on older interpreters (regular version-specific module).
nanobind_add_module(vextor_python STABLE_ABI
    bindings.cpp
)
target_link_libraries(vextor_python PRIVATE vextor)
set_target_properties(vextor_python PROPERTIES OUTPUT_NAME "vextor")

# nanobind uses non-standard extensions that trigger -Wpedantic; suppress on
# third-party targets. The nanobind library target is created lazily by
# nanobind_add_module and is named per ABI variant, so cover both.
foreach(nb_target nanobind-static nanobind-static-abi3)
    if(TARGET ${nb_target})
        target_compile_options(${nb_target} PRIVATE -w)
    endif()
endforeach()
target_compile_options(vextor_python PRIVATE -Wno-nested-anon-types -Wno-zero-length-array)

# Wheel layout: scikit-build-core packages whatever install() produces.
if(SKBUILD)
    install(TARGETS vextor_python LIBRARY DESTINATION .)
endif()
