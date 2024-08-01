cmake_minimum_required(VERSION 3.25)
project(shaderpulse-tests)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(gtest PRIVATE -Wno-error)
  target_compile_options(gtest_main PRIVATE -Wno-error)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_options(gtest PRIVATE /wd4996)
  target_compile_options(gtest_main PRIVATE /wd4996)
endif()

enable_testing()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/lib)
include_directories(${CMAKE_BINARY_DIR}/include)

add_executable(LexerTest ${CMAKE_SOURCE_DIR}/unittests/Lexer/LexerTest.cpp)
target_link_libraries(LexerTest
  gtest
  gtest_main
  Lexer
)

add_executable(ParserTest ${CMAKE_SOURCE_DIR}/unittests/Parser/ParserTest.cpp)
target_link_libraries(ParserTest
  gtest
  gtest_main
  Parser
  Lexer
)

add_test(NAME LexerTest COMMAND LexerTest)
add_test(NAME ParserTest COMMAND ParserTest)
