#include "CodeGen/TypeConversion.h"
#include <iostream>

namespace shaderpulse {

namespace codegen {

mlir::Type convertShaderPulseType(mlir::MLIRContext *ctx, Type *shaderPulseType, llvm::StringMap<ast::StructDeclaration*> &structDeclarations) {
  switch (shaderPulseType->getKind()) {
  case TypeKind::Void:
    return mlir::NoneType::get(ctx);
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
  case TypeKind::Array: {
    auto arrType = dynamic_cast<shaderpulse::ArrayType *>(shaderPulseType);
    auto elementType = convertShaderPulseType(ctx, arrType->getElementType(), structDeclarations);
    auto shape = arrType->getShape();
    mlir::spirv::ArrayType spirvArrType = mlir::spirv::ArrayType::get(elementType, shape[shape.size()-1]);

    for (int i = shape.size()-2; i >= 0; i--) {
      spirvArrType = mlir::spirv::ArrayType::get(spirvArrType, shape[i]);
    }

    return spirvArrType;
  }
  case TypeKind::Vector: {
    auto vecType = dynamic_cast<shaderpulse::VectorType *>(shaderPulseType);
    llvm::SmallVector<int64_t, 1> shape;
    shape.push_back(vecType->getLength());
    return mlir::VectorType::get(
        shape, convertShaderPulseType(ctx, vecType->getElementType(), structDeclarations));
  }
  case TypeKind::Struct: {
    auto structType = dynamic_cast<shaderpulse::StructType *>(shaderPulseType);
    std::vector<mlir::Type> memberTypes;
    ast::StructDeclaration *structDecl = structDeclarations[structType->getName()];

    for (auto &member : structDecl->getMembers()) {
      auto varMember = dynamic_cast<ast::VariableDeclaration*>(member.get());
      std::cout << "Converting member type: " << varMember->getIdentifierName() << std::endl;
      memberTypes.push_back(convertShaderPulseType(ctx, varMember->getType(), structDeclarations));
    }

    return mlir::spirv::StructType::get(memberTypes);
  }
  case TypeKind::Matrix: {
    // Need to create a spirv::MatrixType here, defined as columns of vectors
    auto matrixType = dynamic_cast<shaderpulse::MatrixType *>(shaderPulseType);
    llvm::SmallVector<int64_t, 1> shape;
    shape.push_back(matrixType->getRows());
    auto colVectorType = mlir::VectorType::get(shape, convertShaderPulseType(ctx, matrixType->getElementType(), structDeclarations));
    return mlir::spirv::MatrixType::get(colVectorType, matrixType->getCols());
  }
  default:
    return mlir::Type();
  }
}

std::optional<mlir::spirv::StorageClass>
getSpirvStorageClass(TypeQualifier *typeQualifier) {
  if (!typeQualifier) {
    return std::nullopt;
  }

  if (typeQualifier->getKind() == TypeQualifierKind::Storage) {
    auto storageQualifier = dynamic_cast<StorageQualifier *>(typeQualifier);

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


std::optional<mlir::IntegerAttr>
getLocationFromTypeQualifier(mlir::MLIRContext *ctx, TypeQualifier *typeQualifier) {
  if (!typeQualifier) {
    std::cout << "Null typequalifier" << std::endl;
    return std::nullopt;
  }

  if (auto layoutQualifier = dynamic_cast<LayoutQualifier *>(typeQualifier)) {
    if (auto location = layoutQualifier->getQualifierId("location")) {
      if (!location->getExpression()) {
        return std::nullopt;
      } 
      
      if (auto integerConstant = dynamic_cast<ast::IntegerConstantExpression *>(location->getExpression())) {
        return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signless), integerConstant->getVal());
      }
    }
  }
  
  return std::nullopt;
}

}; // namespace codegen

}; // namespace shaderpulse
