cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(
  shaderpulse
  VERSION 0.1
  LANGUAGES CXX C
)

# Set the MLIR and LLVM installation prefixes
set(CMAKE_PREFIX_PATH "./llvm-project/build" ${CMAKE_PREFIX_PATH})

# MLIR specific
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}/cmake/modules")

add_subdirectory(lib/Preprocessor)
add_subdirectory(lib/Lexer)
add_subdirectory(lib/Parser)
add_subdirectory(lib/AST)
add_subdirectory(lib/CodeGen)
add_subdirectory(lib/Analysis)

add_executable(shaderpulse-standalone standalone/shaderpulse.cpp)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/utils/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

target_link_libraries(shaderpulse-standalone
  PRIVATE
    Preprocessor
    Lexer
    Parser
    AST
    CodeGen
    Analysis
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRSPIRVDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms
)

add_subdirectory(unittests)
