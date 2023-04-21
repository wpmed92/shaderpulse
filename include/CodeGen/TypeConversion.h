#pragma once
#include "AST/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

namespace shaderpulse {

namespace codegen {

mlir::Type convertShaderPulseType(mlir::MLIRContext*, Type*);
std::optional<mlir::spirv::StorageClass> getSpirvStorageClass(TypeQualifier*);

}; // namespace codegen
}; // namespace shaderpulse