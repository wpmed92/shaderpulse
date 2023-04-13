cmake_minimum_required(VERSION 3.25)

project(
  jasp
  VERSION 0.1
  LANGUAGES CXX
)

set (CMAKE_CXX_STANDARD 17)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

add_library(jasp-lib 
  lib/Lexer/Lexer.cpp 
  lib/Lexer/Token.cpp
)

target_include_directories(jasp-lib PUBLIC 
  include/
  utils/)

enable_testing()

# Lexer tests
add_executable(
  lexer_tests
  unittests/Lexer/LexerTest.cpp
)
target_link_libraries(
  lexer_tests
  PRIVATE jasp-lib
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(lexer_tests)
