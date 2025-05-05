add_executable(matmul_benchmark
			   matmul.cpp)

target_compile_options(matmul_benchmark PRIVATE -O3)

target_link_libraries(matmul_benchmark
PRIVATE
    benchmark::benchmark
)

target_include_directories(matmul_benchmark PUBLIC ${PROJECT_SOURCE_DIR}/includes)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/run_benchmark.sh
    ${CMAKE_CURRENT_BINARY_DIR}/run_benchmark.sh
    COPYONLY
)