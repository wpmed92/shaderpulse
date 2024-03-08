include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/utils/include)
add_library(CodeGen
  TypeConversion.cpp
  MLIRCodeGen.cpp
)
