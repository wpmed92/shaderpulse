#pragma once
#include "AST/Types.h"
#include "AST/AST.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace shaderpulse {

namespace codegen {

mlir::Type convertShaderPulseType(mlir::MLIRContext *, shaderpulse::Type *, llvm::StringMap<ast::StructDeclaration*> &);
std::optional<mlir::spirv::StorageClass> getSpirvStorageClass(shaderpulse::TypeQualifier *);
std::optional<mlir::IntegerAttr> getIntegerAttrFromLayoutQualifier(mlir::MLIRContext *ctx, const std::string& id, TypeQualifier *typeQualifier);
mlir::Type getElementType(mlir::Type type);
bool isBoolLike(mlir::Type type);
bool isIntLike(mlir::Type type);
bool isSIntLike(mlir::Type type);
bool isUIntLike(mlir::Type type);
bool isFloatLike(mlir::Type type);
bool isF32Like(mlir::Type type);
bool isF64Like(mlir::Type type);

}; // namespace codegen
}; // namespace shaderpulse
