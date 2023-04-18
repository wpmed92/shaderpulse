#include "CodeGen/TypeConversion.h"

namespace shaderpulse {

namespace codegen {

mlir::Type convertShaderPulseType(mlir::MLIRContext* ctx, Type* shaderPulseType) {
    switch (shaderPulseType->getKind()) {
        case TypeKind::Void:
            // Conversion not available
            break;
        case TypeKind::Bool:
            return mlir::IntegerType::get(ctx, 1);
        case TypeKind::Integer:
            return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
        case TypeKind::UnsignedInteger:
            return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
        case TypeKind::Float:
            return mlir::FloatType::getF32(ctx);
        case TypeKind::Double:
            return mlir::FloatType::getF64(ctx);
        case TypeKind::Vector:
            // TODO:
            break;
        case TypeKind::Matrix:
            // TODO: 
            break;
    }
}

};

};
