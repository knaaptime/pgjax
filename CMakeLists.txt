cmake_minimum_required(VERSION 3.18)
project(pgjax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# XLA FFI headers ship with jaxlib (header-only, nothing to link).
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import jax.ffi; print(jax.ffi.include_dir())"
    OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)

# Locate nanobind via its Python package.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
find_package(nanobind REQUIRED CONFIG)

nanobind_add_module(pgjax_cpp src/cpp/pgjax.cpp)
target_include_directories(pgjax_cpp PRIVATE ${XLA_FFI_INCLUDE_DIR})

install(TARGETS pgjax_cpp LIBRARY DESTINATION .)
