add_executable(etc_ut matrix.cpp tensor.cpp network.cpp)

target_link_libraries(etc_ut
PRIVATE
	GTest::GTest
	GTest::gtest_main
)

target_include_directories(etc_ut PUBLIC ${PROJECT_SOURCE_DIR}/includes)

include(GoogleTest)
gtest_discover_tests(etc_ut)