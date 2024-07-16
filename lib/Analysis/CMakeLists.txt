include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(Analysis
  Scope.cpp
  SymbolTable.cpp
  TypeChecker.cpp
)
