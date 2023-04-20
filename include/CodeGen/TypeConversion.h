#pragma once
#include "AST/Types.h"
#include "mlir/IR/BuiltinTypes.h"


namespace shaderpulse {

namespace codegen {

mlir::Type convertShaderPulseType(mlir::MLIRContext*, Type*);

}; // namespace codegen
}; // namespace shaderpulse