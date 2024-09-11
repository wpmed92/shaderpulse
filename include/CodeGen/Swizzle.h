#include "AST/AST.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include <unordered_map>

namespace shaderpulse {

namespace codegen {

extern std::unordered_map<char, int> swizzleMap;
mlir::Value swizzle(mlir::OpBuilder &builder, mlir::Value composite, ast::MemberAccessExpression* memberAccess);

};

};