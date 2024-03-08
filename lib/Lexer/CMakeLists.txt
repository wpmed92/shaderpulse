include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/utils/include)
add_library(Lexer
  Lexer.cpp
  Token.cpp
)
