#include "CodeGen/TypeConversion.h"

namespace shaderpulse {

namespace codegen {

mlir::Type convertShaderPulseType(mlir::MLIRContext* ctx, Type* shaderPulseType) {
    switch (shaderPulseType->getKind()) {
        case TypeKind::Void:
            return  mlir::NoneType::get(ctx);
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
        case TypeKind::Vector: {
            auto vecType = dynamic_cast<shaderpulse::VectorType*>(shaderPulseType);
            llvm::SmallVector<int64_t, 1> shape;
            shape.push_back(vecType->getLength());
            return mlir::VectorType::get(shape, convertShaderPulseType(ctx, vecType->getElementType()));
        }
        case TypeKind::Matrix: {
            auto matrixType = dynamic_cast<shaderpulse::MatrixType*>(shaderPulseType);
            llvm::SmallVector<int64_t, 2> shape;
            shape.push_back(matrixType->getRows());
            shape.push_back(matrixType->getCols());
            return mlir::VectorType::get(shape, convertShaderPulseType(ctx, matrixType->getElementType()));
        }
    }
}


std::optional<mlir::spirv::StorageClass> getSpirvStorageClass(TypeQualifier* typeQualifier) {
    if (!typeQualifier) {
        return std::nullopt;
    }

    if (typeQualifier->getKind() == TypeQualifierKind::Storage) {
        auto storageQualifier = dynamic_cast<StorageQualifier*>(typeQualifier);

        switch (storageQualifier->getKind()) {
            case StorageQualifierKind::Uniform:
                return mlir::spirv::StorageClass::Uniform;
            case StorageQualifierKind::In:
                return mlir::spirv::StorageClass::Input;
            case StorageQualifierKind::Out:
                return mlir::spirv::StorageClass::Output;
            default:
                std::nullopt;
        }
    }
    
    return std::nullopt;
}

};

};
