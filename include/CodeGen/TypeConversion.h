#pragma once
#include "AST/Types.h"
#include "AST/AST.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace shaderpulse {

namespace codegen {

mlir::Type convertShaderPulseType(mlir::MLIRContext *, Type *, llvm::StringMap<ast::StructDeclaration*> &);
std::optional<mlir::spirv::StorageClass> getSpirvStorageClass(TypeQualifier *);
std::optional<mlir::IntegerAttr> getLocationFromTypeQualifier(mlir::MLIRContext *ctx, TypeQualifier *);

}; // namespace codegen
}; // namespace shaderpulse
