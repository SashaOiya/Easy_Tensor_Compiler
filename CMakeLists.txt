cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project(etc LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

enable_testing()

add_executable(etc src/main.cpp)
target_compile_features(etc PUBLIC cxx_std_23)
target_include_directories(etc PRIVATE ${CMAKE_SOURCE_DIR}/includes)

add_subdirectory(tests/unit_test)
#add_subdirectory(tests/e2e_tests)
#add_subdirectory(benchmark)
