#include "CodeGen/MLIRCodeGen.h"
#include "AST/AST.h"
#include "CodeGen/Swizzle.h"
#include "CodeGen/TypeConversion.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include "mlir/Target/SPIRV/Serialization.h"
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/MemoryBuffer.h>
#include <utility>

namespace shaderpulse {

using namespace ast;

namespace codegen {

std::vector<std::pair<std::string, std::string>> builtinComputeVars = {
    {"gl_GlobalInvocationID", "GlobalInvocationId"},
    {"gl_WorkGroupID", "WorkgroupId"},
    {"gl_WorkGroupSize", "WorkgroupSize"},
    {"gl_LocalInvocationID", "LocalInvocationId"}
};

MLIRCodeGen::MLIRCodeGen() : builder(&context) {
  context.getOrLoadDialect<spirv::SPIRVDialect>();
  initModuleOp();
  initBuiltinFuncMap();
}

void MLIRCodeGen::initBuiltinFuncMap() {
  builtInFuncMap = {
    {"abs", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      mlir::Value result {};

      if (isFloatLike(operands[0].getType())) {
        result = builder.create<spirv::GLFAbsOp>(builder.getUnknownLoc(), operands[0]);
      } else {
        result = builder.create<spirv::GLSAbsOp>(builder.getUnknownLoc(), operands[0]);
      }

      return result;
    }},
    {"acos", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLAcosOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"asin", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLAsinOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"atan", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLAtanOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"ceil", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLCeilOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"clamp", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      mlir::Type type = operands[0].getType();
      mlir::Value result {};

      if (isFloatLike(type)) {
        result = builder.create<spirv::GLFClampOp>(builder.getUnknownLoc(), type, operands[0], operands[1], operands[2]);
      } else if (isUIntLike(type)) {
        result = builder.create<spirv::GLUClampOp>(builder.getUnknownLoc(), type, operands[0], operands[1], operands[2]);
      } else {
        result = builder.create<spirv::GLSClampOp>(builder.getUnknownLoc(), type, operands[0], operands[1], operands[2]);
      }

      return result;
    }},
    {"cos", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLCosOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"cosh", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLCoshOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"dot", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      auto op0ElementType = operands[0].getType().dyn_cast<mlir::VectorType>().getElementType();
      return builder.create<spirv::DotOp>(builder.getUnknownLoc(), op0ElementType, operands[0], operands[1]);
    }},
    {"exp", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLExpOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"exp2", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      auto two = builder.create<spirv::ConstantOp>(builder.getUnknownLoc(), mlir::FloatType::getF32(&context), builder.getF32FloatAttr(2.0f));
      return builder.create<spirv::GLPowOp>(builder.getUnknownLoc(), operands[0].getType(), two, operands[0]);
    }},
    {"findumsb", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLFindUMsbOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"floor", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLFloorOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"fma", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLFmaOp>(builder.getUnknownLoc(), operands[0].getType(), operands[0], operands[1], operands[2]);
    }},
    {"frexpstruct", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      // TODO: implement me
      return mlir::Value();
    }},
    {"inversesqrt", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLInverseSqrtOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"ldexp", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLLdexpOp>(builder.getUnknownLoc(), operands[0].getType(), operands[0], operands[1]);
    }},
    {"log", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLLogOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"max", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      mlir::Type type = operands[0].getType();
      mlir::Value result {};

      if (isFloatLike(type)) {
        result = builder.create<spirv::GLFMaxOp>(builder.getUnknownLoc(), type, operands[0], operands[1]);
      } else if (isUIntLike(type)) {
        result = builder.create<spirv::GLUMaxOp>(builder.getUnknownLoc(), type, operands[0], operands[1]);
      } else {
        result = builder.create<spirv::GLSMaxOp>(builder.getUnknownLoc(), type, operands[0], operands[1]);
      }

      return result;
    }},
    {"min", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      mlir::Type type = operands[0].getType();
      mlir::Value result {};

      if (isFloatLike(type)) {
        result = builder.create<spirv::GLFMinOp>(builder.getUnknownLoc(), type, operands[0], operands[1]);
      } else if (isUIntLike(type)) {
        result = builder.create<spirv::GLUMinOp>(builder.getUnknownLoc(), type, operands[0], operands[1]);
      } else {
        result = builder.create<spirv::GLSMinOp>(builder.getUnknownLoc(), type, operands[0], operands[1]);
      }

      return result;
    }},
    {"mix", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLFMixOp>(builder.getUnknownLoc(), operands[0].getType(), operands[0], operands[1], operands[2]);
    }},
    {"pow", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLPowOp>(builder.getUnknownLoc(), operands[0].getType(), operands[0], operands[1]);
    }},
    {"roundeven", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLRoundEvenOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"round", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLRoundOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"sign", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      mlir::Value result {};

      if (isFloatLike(operands[0].getType())) {
        result = builder.create<spirv::GLFSignOp>(builder.getUnknownLoc(), operands[0]);
      } else {
        result = builder.create<spirv::GLSSignOp>(builder.getUnknownLoc(), operands[0]);
      }

      return result;
    }},
    {"sin", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLSinOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"sinh", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLSinhOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"sqrt", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLSqrtOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"tan", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLTanOp>(builder.getUnknownLoc(), operands[0]);
    }},
    {"tanh", [](mlir::MLIRContext &context, mlir::OpBuilder &builder, mlir::ValueRange operands) {
      return builder.create<spirv::GLTanhOp>(builder.getUnknownLoc(), operands[0]);
    }}
  };
}

void MLIRCodeGen::initModuleOp() {
  OperationState state(UnknownLoc::get(&context),
                        spirv::ModuleOp::getOperationName());
  state.addAttribute("addressing_model",
                      builder.getAttr<spirv::AddressingModelAttr>(
                          spirv::AddressingModel::Logical));
  state.addAttribute("memory_model", builder.getAttr<spirv::MemoryModelAttr>(
                                          spirv::MemoryModel::GLSL450));
  state.addAttribute("vce_triple",
                      spirv::VerCapExtAttr::get(
                          spirv::Version::V_1_0,
                          { spirv::Capability::Shader },
                          llvm::ArrayRef<spirv::Extension>(), &context));
  spirv::ModuleOp::build(builder, state);
  spirvModule = cast<spirv::ModuleOp>(Operation::create(state));
}

void MLIRCodeGen::print() {
  spirvModule.print(llvm::outs());
}

bool MLIRCodeGen::saveToFile(const std::filesystem::path& outputPath) {
  std::string buffer;
  llvm::raw_string_ostream outputStream(buffer);
  spirvModule.print(outputStream);
  outputStream.flush();

  std::ofstream outputFile(outputPath);
  if (!outputFile) {
      llvm::errs() << "Failed to open output file: " << outputPath << "\n";
      return false;
  }

  outputFile << buffer;

  return true;
}

bool MLIRCodeGen::emitSpirv(const std::filesystem::path& outputPath) {
  llvm::SmallVector<uint32_t, 128> spirvBinary;
  mlir::LogicalResult result = mlir::spirv::serialize(spirvModule, spirvBinary);
  if (failed(result)) {
      std::cerr << "Failed to serialize SPIR-V module." << std::endl;
      return false;
  }

  std::ofstream outFile(outputPath, std::ios::binary);
  outFile.write(reinterpret_cast<const char*>(spirvBinary.data()), spirvBinary.size() * sizeof(uint32_t));
  return true;
}

bool MLIRCodeGen::verify() { return !failed(mlir::verify(spirvModule)); }

void MLIRCodeGen::insertEntryPoint() {
  std::vector<mlir::Attribute> interfaceArr;

  for (const auto& pair : interface) {
    interfaceArr.push_back(pair.second);
  }

  builder.setInsertionPointToEnd(spirvModule.getBody());
  builder.create<spirv::EntryPointOp>(builder.getUnknownLoc(), spirv::ExecutionModelAttr::get(&context, spirv::ExecutionModel::GLCompute),
                                        SymbolRefAttr::get(&context, "main"), ArrayAttr::get(&context, interfaceArr));
}

void MLIRCodeGen::visit(TranslationUnit *unit) {
  SymbolTableScopeT globalScope(symbolTable);
  builder.setInsertionPointToEnd(spirvModule.getBody());
  
  for (auto &glslIdSPIRVIdPair : builtinComputeVars) {
    createBuiltinComputeVar(glslIdSPIRVIdPair.first, glslIdSPIRVIdPair.second);
  }

  for (auto &extDecl : unit->getExternalDeclarations()) {
    extDecl->accept(this);
  }

  insertEntryPoint();
}

mlir::Value MLIRCodeGen::popExpressionStack() {
  assert(expressionStack.size() > 0 && "Expression stack is empty");
  auto val = expressionStack.back();
  expressionStack.pop_back();
  return val;
}

void MLIRCodeGen::visit(BinaryExpression *binExp) {
  binExp->getLhs()->accept(this);
  binExp->getRhs()->accept(this);

  mlir::Value rhs = load(popExpressionStack());
  mlir::Value lhs = load(popExpressionStack());
  mlir::Type typeContext = rhs.getType();

  // TODO: implement source location
  auto loc = builder.getUnknownLoc();
  mlir::Value val;

  switch (binExp->getOp()) {
  case BinaryOperator::Add:
    if (isIntLike(typeContext)) {
      val = builder.create<spirv::IAddOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::FAddOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Sub:
    if (isIntLike(typeContext)) {
      val = builder.create<spirv::ISubOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::FSubOp>(loc, lhs, rhs);
    }

    expressionStack.push_back(val);
    break;
  case BinaryOperator::Mul:
  if (isIntLike(typeContext)) {
      val = builder.create<spirv::IMulOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::FMulOp>(loc, lhs, rhs);
    }

    expressionStack.push_back(val);
    break;
  case BinaryOperator::Div:
    if (isUIntLike(typeContext)) {
      val = builder.create<spirv::UDivOp>(loc, lhs, rhs);
    } else if (isIntLike(typeContext)) {
      val = builder.create<spirv::SDivOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::FDivOp>(loc, lhs, rhs);
    }

    expressionStack.push_back(val);
    break;
  case BinaryOperator::Mod:
    if (isIntLike(typeContext)) {
      val = builder.create<spirv::SRemOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::FRemOp>(loc, lhs, rhs);
    }

    expressionStack.push_back(val);
    break;
  case BinaryOperator::ShiftLeft:
    val = builder.create<spirv::ShiftLeftLogicalOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::ShiftRight:
    val = builder.create<spirv::ShiftRightLogicalOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Lt:
    if (isFloatLike(typeContext)) {
      val = builder.create<spirv::FOrdLessThanOp>(loc, lhs, rhs);
    } else if (isUIntLike(typeContext)) {
      val = builder.create<spirv::ULessThanOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::SLessThanOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Gt:
    if (isFloatLike(typeContext)) {
      val = builder.create<spirv::FOrdGreaterThanOp>(loc, lhs, rhs);
    } else if (isUIntLike(typeContext)) {
      val = builder.create<spirv::UGreaterThanOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::SGreaterThanOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(val);
    break;
  case BinaryOperator::LtEq:
    if (isFloatLike(typeContext)) {
      val = builder.create<spirv::FOrdLessThanEqualOp>(loc, lhs, rhs);
    } else if (isUIntLike(typeContext)) {
      val = builder.create<spirv::ULessThanEqualOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::SLessThanEqualOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(val);
    break;
  case BinaryOperator::GtEq:
    if (isFloatLike(typeContext)) {
      val = builder.create<spirv::FOrdGreaterThanEqualOp>(loc, lhs, rhs);
    } else if (isUIntLike(typeContext)) {
      val = builder.create<spirv::UGreaterThanEqualOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::SGreaterThanEqualOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Eq:
    if (isFloatLike(typeContext)) {
      val = builder.create<spirv::FOrdEqualOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::IEqualOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Neq:
    if (isFloatLike(typeContext)) {
      val = builder.create<spirv::FOrdNotEqualOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::INotEqualOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(val);
    break;
  case BinaryOperator::BitAnd:
    val = builder.create<spirv::BitwiseAndOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::BitXor:
    val = builder.create<spirv::BitwiseXorOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::BitIor:
    val = builder.create<spirv::BitwiseOrOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::LogAnd:
    val = builder.create<spirv::LogicalAndOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::LogXor:
    // TODO: not implemented in current spirv dialect
    break;
  case BinaryOperator::LogOr:
    val = builder.create<spirv::LogicalOrOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  }
}

void MLIRCodeGen::visit(ConditionalExpression *condExp) {
  condExp->getFalsePart()->accept(this);
  condExp->getTruePart()->accept(this);
  condExp->getCondition()->accept(this);

  mlir::Value condition = load(popExpressionStack());
  mlir::Value truePart = load(popExpressionStack());
  mlir::Value falsePart = load(popExpressionStack());

  mlir::Value res = builder.create<spirv::SelectOp>(
    builder.getUnknownLoc(),
    truePart.getType(),
    condition,
    truePart,
    falsePart);

  expressionStack.push_back(res);
}

void MLIRCodeGen::visit(ForStatement *forStmt) {
  generateLoop(forStmt->getInitStatement(), forStmt->getConditionExpression(), forStmt->getInductionExpression(), forStmt->getBody());
}

void MLIRCodeGen::visit(InitializerExpression *initExp) {
  /* TODO: This is a placeholder to have something on the expression stack,
   * will be replaced with actual InitializerExpression code gen once type inference
   * is implemented.
   */
  auto type = builder.getIntegerType(32, true);
  mlir::Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      IntegerAttr::get(type, APInt(32, 0, true)));

  expressionStack.push_back(val);
}

void MLIRCodeGen::visit(UnaryExpression *unExp) {
  unExp->getExpression()->accept(this);
  mlir::Value ptrRhs = popExpressionStack();
  mlir::Value rhs = load(ptrRhs);
  mlir::Value result {};
  mlir::Type rhsType = rhs.getType();

  auto loc = builder.getUnknownLoc();
  auto op = unExp->getOp();

  switch (op) {
  case UnaryOperator::Inc:
  case UnaryOperator::Dec: {
    if (isIntLike(rhsType)) {
      auto one = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(),
        mlir::IntegerType::get(&context, 32, isUIntLike(rhsType) ? mlir::IntegerType::Unsigned : mlir::IntegerType::Signed),
        isUIntLike(rhsType) ? builder.getUI32IntegerAttr(1) :builder.getSI32IntegerAttr(1)
      );

      if (op == UnaryOperator::Inc) {
        result = builder.create<spirv::IAddOp>(loc, rhs, one);
      } else {
        result = builder.create<spirv::ISubOp>(loc, rhs, one);
      }
    } else {
      auto one = builder.create<spirv::ConstantOp>(builder.getUnknownLoc(), mlir::FloatType::getF32(&context), builder.getF32FloatAttr(1.0f));

      if (op == UnaryOperator::Inc) {
        result = builder.create<spirv::FAddOp>(loc, rhs, one);
      } else {
        result = builder.create<spirv::FSubOp>(loc, rhs, one);
      }
    }

    builder.create<spirv::StoreOp>(builder.getUnknownLoc(), ptrRhs, result);
    expressionStack.push_back(result);
    break;
  }
  case UnaryOperator::Plus:
    expressionStack.push_back(rhs);
    break;
  case UnaryOperator::Dash:
    if (isFloatLike(rhsType)) {
      result = builder.create<spirv::FNegateOp>(loc, rhs);
    } else {
      result = builder.create<spirv::SNegateOp>(loc, rhs);
    }

    expressionStack.push_back(result);
    break;
  case UnaryOperator::Bang:
    result = builder.create<spirv::LogicalNotOp>(loc, rhs);
    expressionStack.push_back(result);
    break;
  case UnaryOperator::Tilde:
    result = builder.create<spirv::NotOp>(loc, rhs);
    expressionStack.push_back(result);
    break;
  }
}

void MLIRCodeGen::declare(llvm::StringRef name, SymbolTableEntry entry) {
  symbolTable.insert(name, entry);
}

void MLIRCodeGen::visit(VariableDeclarationList *varDeclList) {
  for (auto &var : varDeclList->getDeclarations()) {
    createVariable(varDeclList->getType()->getQualifiers(), varDeclList->getType(), var.get());
  }
}

void MLIRCodeGen::createBuiltinComputeVar(const std::string& varName, const std::string& mlirName) {
  mlir::Type vec3I32Type = mlir::VectorType::get({3}, mlir::IntegerType::get(&context, 32, mlir::IntegerType::Unsigned));
  spirv::PointerType ptrType = spirv::PointerType::get(vec3I32Type, mlir::spirv::StorageClass::Input);
  auto globalInvocationId = builder.create<spirv::GlobalVariableOp>(
        UnknownLoc::get(&context), TypeAttr::get(ptrType),
        builder.getStringAttr(varName),
        nullptr);

  globalInvocationId->setAttr(
    StringAttr::get(&context, "built_in"),
    StringAttr::get(&context, mlirName)
  );

  declare(varName, { mlir::Value(), nullptr, ptrType, /*isGlobal*/ true});
}

void MLIRCodeGen::createVariable(shaderpulse::TypeQualifierList* qualifiers, shaderpulse::Type *varType,
                                 VariableDeclaration *varDecl) {
  if (inGlobalScope) {
    spirv::StorageClass storageClass;

    if (auto st = getSpirvStorageClass(qualifiers->find(shaderpulse::TypeQualifierKind::Storage))) {
      storageClass = *st;
    } else {
      storageClass = spirv::StorageClass::Private;
    }

    builder.setInsertionPointToEnd(spirvModule.getBody());
  
    spirv::PointerType ptrType = spirv::PointerType::get(
      convertShaderPulseType(&context, varType, structDeclarations), storageClass);

    // TODO: check global initialization
    mlir::Operation *initializerOp {};

    /*
     * mlir::Value val = popExpressionStack().second;
     * initializerOp = val.getDefiningOp();
    */

    auto varOp = builder.create<spirv::GlobalVariableOp>(
        UnknownLoc::get(&context), TypeAttr::get(ptrType),
        builder.getStringAttr(varDecl->getIdentifierName()),
        (initializerOp) ? FlatSymbolRefAttr::get(initializerOp) : nullptr);

    auto layoutQualifier = qualifiers->find(shaderpulse::TypeQualifierKind::Layout);

    if (auto locationOpt = getIntegerAttrFromLayoutQualifier(&context, "location", layoutQualifier)) {
      varOp->setAttr("location", *locationOpt);
    } else if (auto bindingOpt = getIntegerAttrFromLayoutQualifier(&context, "binding", layoutQualifier)) {
      varOp->setAttr("binding", *bindingOpt);
    }

    declare(varDecl->getIdentifierName(), { mlir::Value(), varDecl, ptrType, /*isGlobal*/ true});
    // Set OpDecorate through attributes
    // example:
    // varOp->setAttr(spirv::stringifyDecoration(spirv::Decoration::Invariant),
    // builder.getUnitAttr());
  } else {
    mlir::Value val {};

    if (varDecl->getInitialzerExpression()) {
      varDecl->getInitialzerExpression()->accept(this);
      val = load(popExpressionStack());
    }

    spirv::PointerType ptrType = spirv::PointerType::get(
        convertShaderPulseType(&context, varType, structDeclarations), spirv::StorageClass::Function);

    auto var = builder.create<spirv::VariableOp>(
        builder.getUnknownLoc(), ptrType, spirv::StorageClass::Function, nullptr);

    if (varDecl->getInitialzerExpression()) {
      builder.create<spirv::StoreOp>(builder.getUnknownLoc(), var, val);
    }

    declare(varDecl->getIdentifierName(), { var, varDecl, nullptr, /*isGlobal*/ false});
  }
}

void MLIRCodeGen::visit(VariableDeclaration *varDecl) {
  if (!varDecl->getIdentifierName().empty()) {
    createVariable(varDecl->getType()->getQualifiers(), varDecl->getType(), varDecl);
  }
}

void MLIRCodeGen::visit(SwitchStatement *switchStmt) {
  // TODO: implement me
}

void MLIRCodeGen::visit(WhileStatement *whileStmt) {
  generateLoop(nullptr, whileStmt->getCondition(), nullptr, whileStmt->getBody());
}

void MLIRCodeGen::visit(ConstructorExpression *constructorExp) {
  auto constructorType = constructorExp->getType();

  std::vector<mlir::Value> operands;

  if (constructorExp->getArguments().size() > 0) {
    for (auto &arg : constructorExp->getArguments()) {
      arg->accept(this);
      operands.push_back(load(popExpressionStack()));
    }
  }

  auto constructorTypeKind = constructorType->getKind();

  switch (constructorTypeKind) {
    case shaderpulse::TypeKind::Struct: {
      auto structName = dynamic_cast<shaderpulse::StructType*>(constructorType)->getName();

      if (structDeclarations.find(structName) != structDeclarations.end()) {
        mlir::Value val = builder.create<spirv::CompositeConstructOp>(
              builder.getUnknownLoc(), convertShaderPulseType(&context, constructorType, structDeclarations), operands);
        expressionStack.push_back(val);
      }

      break;
    }

    case shaderpulse::TypeKind::Vector:
    case shaderpulse::TypeKind::Array: {
      // If the vector constructor has a single argument, and it is the same length as the current vector,
      // but the element type is different, than it is a type conversion and not a composite construction.
      if (constructorTypeKind == shaderpulse::TypeKind::Vector && (operands.size() == 1) && (operands[0].getType().isa<mlir::VectorType>())) {
        auto argVecType = operands[0].getType().dyn_cast<mlir::VectorType>();
        auto constrVecType = convertShaderPulseType(&context, constructorType, structDeclarations).dyn_cast<mlir::VectorType>();

        if ((argVecType.getShape()[0] == constrVecType.getShape()[0]) && (argVecType.getElementType() != constrVecType.getElementType())) {
          convertOp(constructorExp, operands[0]);
        } else {
          mlir::Value val = builder.create<spirv::CompositeConstructOp>(builder.getUnknownLoc(), constrVecType, operands);
          expressionStack.push_back(val);
        }
      } else {
        mlir::Value val = builder.create<spirv::CompositeConstructOp>(
              builder.getUnknownLoc(), convertShaderPulseType(&context, constructorType, structDeclarations), operands);
        expressionStack.push_back(val);
      }
      break;
    }

    case shaderpulse::TypeKind::Matrix: {
      auto matrixType = dynamic_cast<shaderpulse::MatrixType *>(constructorType);
      std::vector<mlir::Value> columnVectors;

      for (int i = 0; i < matrixType->getCols(); i++) {
        std::vector<mlir::Value> col;

        for (int j = 0; j < matrixType->getRows(); j++) {
          col.push_back(operands[j*matrixType->getCols() + i]);
        }

        auto elementType = std::make_unique<shaderpulse::Type>(matrixType->getElementType()->getKind());
        auto vecType = std::make_unique<shaderpulse::VectorType>(std::move(elementType), col.size());
        mlir::Value val = builder.create<spirv::CompositeConstructOp>(
          builder.getUnknownLoc(), convertShaderPulseType(&context, vecType.get(), structDeclarations), col);
        columnVectors.push_back(val);
      }

      mlir::Value val = builder.create<spirv::CompositeConstructOp>(
        builder.getUnknownLoc(), convertShaderPulseType(&context, constructorType, structDeclarations), columnVectors);

      expressionStack.push_back(val);
      break;
    }

    // Scalar type conversions
    default:
      convertOp(constructorExp, operands[0]);
      break;
  }
}

mlir::Value MLIRCodeGen::convertOp(ConstructorExpression* constructorExp, mlir::Value val) {
  mlir::Type toType = convertShaderPulseType(&context, constructorExp->getType(), structDeclarations);
  mlir::Type fromType = val.getType();

  if (isUIntLike(fromType) && isFloatLike(toType)) {
    expressionStack.push_back(builder.create<spirv::ConvertUToFOp>(builder.getUnknownLoc(), toType, val));
  } else if (isIntLike(fromType) && isFloatLike(toType)) {
    expressionStack.push_back(builder.create<spirv::ConvertSToFOp>(builder.getUnknownLoc(), toType, val));
  } else if (isFloatLike(fromType) && isUIntLike(toType)) {
    expressionStack.push_back(builder.create<spirv::ConvertFToUOp>(builder.getUnknownLoc(), toType, val));
  } else if (isFloatLike(fromType) && isIntLike(toType)) {
    expressionStack.push_back(builder.create<spirv::ConvertFToSOp>(builder.getUnknownLoc(), toType, val));
  } else if ((isSIntLike(fromType) && isUIntLike(toType)) || (isUIntLike(fromType) && isSIntLike(toType))) {
    expressionStack.push_back(builder.create<spirv::BitcastOp>(builder.getUnknownLoc(), toType, val));
  } else if (isBoolLike(fromType) && isIntLike(toType)) {
    mlir::Value one {};
    mlir::Value zero {};
    auto constOne = buildIntConst(1, isSIntLike(toType));
    auto constZero = buildIntConst(0, isSIntLike(toType));

    if (fromType.isa<mlir::VectorType>()) {
      zero = buildVecConst(constZero, toType);
      one = buildVecConst(constOne, toType);
    } else {
      one = constOne;
      zero = constZero;
    }

    mlir::Value res = builder.create<spirv::SelectOp>(
      builder.getUnknownLoc(),
      toType,
      val,
      one,
      zero
    );
    expressionStack.push_back(res);
  } else if (isBoolLike(fromType) && isFloatLike(toType)) {
    mlir::Value one;
    mlir::Value zero;
    auto constOne = buildFloatConst(1.0, isF64Like(toType));
    auto constZero = buildFloatConst(0.0, isF64Like(toType));

     if (fromType.isa<mlir::VectorType>()) {
      zero = buildVecConst(constZero, toType);
      one = buildVecConst(constOne, toType);
    } else {
      one = constOne;
      zero = constZero;
    }

    mlir::Value res = builder.create<spirv::SelectOp>(
      builder.getUnknownLoc(),
      toType,
      val,
      one,
      zero
    );

    expressionStack.push_back(res);
  } else if ((isF32Like(fromType) && isF64Like(toType)) || (isF64Like(fromType) && isF32Like(toType))) {
    expressionStack.push_back(builder.create<spirv::FConvertOp>(builder.getUnknownLoc(), toType, val));
  } else if (isBoolLike(toType)) {
    mlir::Value zero;

    if (isIntLike(fromType)) {
      zero = buildIntConst(0, isSIntLike(fromType));

      if (fromType.isa<mlir::VectorType>()) {
        zero = buildVecConst(zero, fromType);
      }

      expressionStack.push_back(builder.create<spirv::INotEqualOp>(builder.getUnknownLoc(), val, zero));
    } else if (isFloatLike(fromType)) {
      zero = buildFloatConst(0.0, isF64Like(fromType));

      if (fromType.isa<mlir::VectorType>()) {
        zero = buildVecConst(zero, fromType);
      }

      expressionStack.push_back(builder.create<spirv::FOrdNotEqualOp>(builder.getUnknownLoc(), val, zero));
    }
  } else {
    expressionStack.push_back(val);
  }
}

void MLIRCodeGen::visit(ArrayAccessExpression *arrayAccess) {
  auto array = arrayAccess->getArray();
  array->accept(this);
  mlir::Value mlirArray = popExpressionStack();
  std::vector<mlir::Value> indices;

  for (auto &access : arrayAccess->getAccessChain()) {
    access->accept(this);
    indices.push_back(load(popExpressionStack()));
  }

  mlir::Value accessChain = builder.create<spirv::AccessChainOp>(builder.getUnknownLoc(), mlirArray, indices);
  expressionStack.push_back(accessChain);
}

void MLIRCodeGen::visit(MemberAccessExpression *memberAccess) {
  auto baseComposite = memberAccess->getBaseComposite();
  baseComposite->accept(this);
  mlir::Value baseCompositeValue = popExpressionStack();
  std::vector<mlir::Value> memberIndicesAcc;

  if (currentBaseComposite) {
    std::pair<int, VariableDeclaration*> prevMemberIndexPair;
    for (int i = 0; i < memberAccess->getMembers().size(); i++) {
      auto &member  = memberAccess->getMembers()[i];
      if (auto var = dynamic_cast<VariableExpression*>(member.get())) {
        // Swizzle detected
        if (prevMemberIndexPair.second && prevMemberIndexPair.second->getType()->getKind() == shaderpulse::TypeKind::Vector) {
          mlir::Value accessChain = builder.create<spirv::AccessChainOp>(builder.getUnknownLoc(), baseCompositeValue, memberIndicesAcc);
          mlir::Value swizzled = swizzle(builder, load(accessChain), memberAccess, i);
          expressionStack.push_back(swizzled);
          return;
        }

        auto memberIndexPair = currentBaseComposite->getMemberWithIndex(var->getName());
        memberIndicesAcc.push_back(builder.create<spirv::ConstantOp>(builder.getUnknownLoc(), mlir::IntegerType::get(&context, 32, mlir::IntegerType::Signless), builder.getI32IntegerAttr(memberIndexPair.first)));

        if (memberIndexPair.second->getType()->getKind() == shaderpulse::TypeKind::Struct) {
          auto structName = dynamic_cast<shaderpulse::StructType*>(memberIndexPair.second->getType())->getName();

          if (structDeclarations.find(structName) != structDeclarations.end()) {
            currentBaseComposite = structDeclarations[structName];
          }
        }

        prevMemberIndexPair = memberIndexPair;
      // This is a duplicate of ArrayAccessExpression, idially we want to reuse that.
      } else if (auto arrayAccess = dynamic_cast<ArrayAccessExpression*>(member.get())) {
        auto varName = dynamic_cast<VariableExpression*>(arrayAccess->getArray())->getName();
        auto memberIndexPair = currentBaseComposite->getMemberWithIndex(varName);
        memberIndicesAcc.push_back(builder.create<spirv::ConstantOp>(builder.getUnknownLoc(), mlir::IntegerType::get(&context, 32, mlir::IntegerType::Signless), builder.getI32IntegerAttr(memberIndexPair.first)));

        for (auto &access : arrayAccess->getAccessChain()) {
          access->accept(this);
          memberIndicesAcc.push_back(load(popExpressionStack()));
        }
      }
    }

    mlir::Value accessChain = builder.create<spirv::AccessChainOp>(builder.getUnknownLoc(), baseCompositeValue, memberIndicesAcc);
    expressionStack.push_back(accessChain);
  } else {
    mlir::Value swizzled = swizzle(builder, load(baseCompositeValue), memberAccess);
    expressionStack.push_back(swizzled);
  }
}

void MLIRCodeGen::visit(StructDeclaration *structDecl) {
  if (structDeclarations.find(structDecl->getName()) == structDeclarations.end()) {
    structDeclarations.insert({structDecl->getName(), structDecl});
  }
}

void MLIRCodeGen::visit(InterfaceBlock *interfaceBlock) {
  auto qualifiers = interfaceBlock->getQualifiers();
  auto &members = interfaceBlock->getMembers();

  if (members.empty()) {
    if (auto layout = dynamic_cast<LayoutQualifier *>(qualifiers->find(shaderpulse::TypeQualifierKind::Layout))) {
      int localSizeX = 1, localSizeY = 1, localSizeZ = 1;

      if (auto layoutLocalX = layout->getQualifierId("local_size_x")) {
        localSizeX = dynamic_cast<IntegerConstantExpression*>(layoutLocalX->getExpression())->getVal();
      }

      if (auto layoutLocalY = layout->getQualifierId("local_size_y")) {
        localSizeY = dynamic_cast<IntegerConstantExpression*>(layoutLocalY->getExpression())->getVal();
      }

      if (auto layoutLocalZ = layout->getQualifierId("local_size_z")) {
        localSizeZ = dynamic_cast<IntegerConstantExpression*>(layoutLocalZ->getExpression())->getVal();
      }

      mlir::OperationState state(builder.getUnknownLoc(), spirv::ExecutionModeOp::getOperationName());
      state.addAttribute("fn", SymbolRefAttr::get(&context, "main"));
      auto execModeAttr = spirv::ExecutionModeAttr::get(&context, spirv::ExecutionMode::LocalSize);
      state.addAttribute("execution_mode", execModeAttr);
      state.addAttribute("values", builder.getI32ArrayAttr({localSizeX, localSizeY, localSizeZ}));
      execModeOp = mlir::Operation::create(state);
    }

    return;
  }

  for (auto &member : members) {
    if (auto memberVar = dynamic_cast<VariableDeclaration*>(member.get())) {
      createVariable(qualifiers, memberVar->getType(), memberVar);
    }
  }
}

void MLIRCodeGen::visit(DoStatement *doStmt) {
  generateLoop(nullptr, doStmt->getCondition(), nullptr, doStmt->getBody(), /*isDoWhile*/ true);
}

void MLIRCodeGen::visit(IfStatement *ifStmt) {
  Block *restoreInsertionBlock = builder.getInsertionBlock();
  auto loc = builder.getUnknownLoc();
  spirv::SelectionOp selectionOp;
  mlir::Value condition;
  Block* selectionHeaderBlock;
  Block* thenBlock;
  Block* mergeBlock;

  // Scope for true part
  {
    SymbolTableScopeT varScope(symbolTable);
    ifStmt->getCondition()->accept(this);
    condition = load(popExpressionStack());
    selectionOp = builder.create<spirv::SelectionOp>(loc, spirv::SelectionControl::None);
    selectionOp.addMergeBlock(builder);

    // Merge
    mergeBlock = selectionOp.getMergeBlock();

    // Selection header
    selectionHeaderBlock = new Block();
    selectionOp.getBody().getBlocks().push_front(selectionHeaderBlock);

    // True part
    thenBlock = new Block();
    selectionOp.getBody().getBlocks().insert(std::next(selectionOp.getBody().begin(), 1), thenBlock);
    builder.setInsertionPointToStart(thenBlock);

    ifStmt->getTruePart()->accept(this);
    builder.create<spirv::BranchOp>(loc, mergeBlock);

    // If scope destroyed here
  }

  // Scope for else part
  SymbolTableScopeT varScope(symbolTable);

  // False part
  auto *elseBlock = new Block();
  selectionOp.getBody().getBlocks().insert(std::next(selectionOp.getBody().begin(), 2), elseBlock);

  builder.setInsertionPointToStart(elseBlock);

  if (ifStmt->getFalsePart()) {
    ifStmt->getFalsePart()->accept(this);
  }

  builder.create<spirv::BranchOp>(loc, mergeBlock);

  // Selection header
  builder.setInsertionPointToEnd(selectionHeaderBlock);

  builder.create<spirv::BranchConditionalOp>(
      loc, condition, thenBlock, ArrayRef<mlir::Value>(), elseBlock, ArrayRef<mlir::Value>());
  
  builder.setInsertionPointToEnd(restoreInsertionBlock);

  // Else scope destroyed here
}

void MLIRCodeGen::visit(AssignmentExpression *assignmentExp) {
  assignmentExp->getUnaryExpression()->accept(this);
  assignmentExp->getExpression()->accept(this);

  mlir::Value val = load(popExpressionStack());
  mlir::Value ptr = popExpressionStack();

  builder.create<spirv::StoreOp>(builder.getUnknownLoc(), ptr, val);
}

void MLIRCodeGen::visit(StatementList *stmtList) {
  for (auto &stmt : stmtList->getStatements()) {
    stmt->accept(this);
  }
}

void MLIRCodeGen::visit(CallExpression *callExp) {
  auto calledFuncIt = functionMap.find(callExp->getFunctionName());

  if (calledFuncIt != functionMap.end()) {
    std::vector<mlir::Value> operands;

    if (callExp->getArguments().size() > 0) {
      for (auto &arg : callExp->getArguments()) {
        arg->accept(this);
        operands.push_back(load(popExpressionStack()));
      }
    }

    spirv::FuncOp calledFunc = calledFuncIt->second;

    spirv::FunctionCallOp funcCall = builder.create<spirv::FunctionCallOp>(
        builder.getUnknownLoc(), calledFunc.getFunctionType().getResults(),
        SymbolRefAttr::get(&context, calledFunc.getSymName()), operands);

    expressionStack.push_back(funcCall.getResult(0));
  } else {
    assert(callBuiltIn(callExp) && "Function not found");
  }
}

void MLIRCodeGen::visit(VariableExpression *varExp) {
  auto entry = symbolTable.lookup(varExp->getName());

  if (entry.isFunctionParam) {
    expressionStack.push_back(entry.value);
  } else if (entry.ptrType || entry.variable) {
    mlir::Value val;

    if (entry.isGlobal) {
      auto addressOfGlobal = builder.create<mlir::spirv::AddressOfOp>(builder.getUnknownLoc(), entry.ptrType, varExp->getName());
      val = addressOfGlobal->getResult(0);

      // If we're inside the entry point function, collect the used global variables
      if (insideEntryPoint) {
        interface.insert({varExp->getName(), SymbolRefAttr::get(&context, varExp->getName())});
      }
    } else {
      val = entry.value;
    }

    if (entry.variable && (entry.variable->getType()->getKind() == shaderpulse::TypeKind::Struct)) {
      auto structName = dynamic_cast<shaderpulse::StructType*>(entry.variable->getType())->getName();

      if (structDeclarations.find(structName) != structDeclarations.end()) {
        currentBaseComposite = structDeclarations[structName];
      }
    }

    expressionStack.push_back(val);
  } else {
    std::cout << "Unable to find variable: " << varExp->getName() << std::endl;
  }
}

void MLIRCodeGen::visit(IntegerConstantExpression *intConstExp) {
  auto type = builder.getIntegerType(32, true);
  mlir::Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      IntegerAttr::get(type, APInt(32, intConstExp->getVal(), true)));

  expressionStack.push_back(val);
}

void MLIRCodeGen::visit(UnsignedIntegerConstantExpression *uintConstExp) {
  auto type = builder.getIntegerType(32, false);
  mlir::Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      IntegerAttr::get(type, APInt(32, uintConstExp->getVal(), false)));

  expressionStack.push_back(val);
}

void MLIRCodeGen::visit(FloatConstantExpression *floatConstExp) {
  auto type = builder.getF32Type();
  mlir::Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      FloatAttr::get(type, APFloat(floatConstExp->getVal())));

  expressionStack.push_back(val);
}

void MLIRCodeGen::visit(DoubleConstantExpression *doubleConstExp) {
  auto type = builder.getF64Type();
  mlir::Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      FloatAttr::get(type, APFloat(doubleConstExp->getVal())));

  expressionStack.push_back(val);
}

void MLIRCodeGen::visit(BoolConstantExpression *boolConstExp) {
  auto type = builder.getIntegerType(1);
  mlir::Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      IntegerAttr::get(type, APInt(1, boolConstExp->getVal())));

  expressionStack.push_back(val);
}

void MLIRCodeGen::visit(ReturnStatement *returnStmt) {
  if (auto retExp = returnStmt->getExpression())
    returnStmt->getExpression()->accept(this);

  if (expressionStack.empty()) {
    builder.create<spirv::ReturnOp>(builder.getUnknownLoc());
  } else {
    mlir::Value val = popExpressionStack();
    builder.create<spirv::ReturnValueOp>(builder.getUnknownLoc(), val);
  }
}

void MLIRCodeGen::visit(BreakStatement *breakStmt) {
  setBoolVar(breakStack.back(), true);
  breakDetected = true;
}

void MLIRCodeGen::visit(ContinueStatement *continueStmt) {
  setBoolVar(continueStack.back(), true);
  continueDetected = true;
}

void MLIRCodeGen::setBoolVar(mlir::spirv::VariableOp var, bool val) {
  auto type = builder.getIntegerType(1);
  mlir::Value constTrue = builder.create<spirv::ConstantOp>(builder.getUnknownLoc(), type, IntegerAttr::get(type, APInt(1, val)));
  builder.create<spirv::StoreOp>(builder.getUnknownLoc(), var, constTrue);
}

void MLIRCodeGen::visit(DiscardStatement *discardStmt) {

}

void MLIRCodeGen::visit(FunctionDeclaration *funcDecl) {
  insideEntryPoint = funcDecl->getName() == "main";

  SymbolTableScopeT varScope(symbolTable);
  std::vector<mlir::Type> paramTypes;

  for (auto &param : funcDecl->getParams()) {
    paramTypes.push_back(convertShaderPulseType(&context, param->getType(), structDeclarations));
  }

  mlir::TypeRange resultTypeRange;
  
  if (auto resultType = convertShaderPulseType(&context, funcDecl->getReturnType(), structDeclarations)) {
    if (!resultType.isa<mlir::NoneType>()) {
      resultTypeRange = { resultType };
    }
  }

  spirv::FuncOp funcOp = builder.create<spirv::FuncOp>(
      builder.getUnknownLoc(), funcDecl->getName(),
      builder.getFunctionType(
          paramTypes,
          resultTypeRange
      )
  );

  inGlobalScope = false;

  auto entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Declare params as variables in the current scope
  for (int i = 0; i < funcDecl->getParams().size(); i++) {
    auto &param = funcDecl->getParams()[i];
    declare(param->getName(), {funcOp.getArgument(i), nullptr, nullptr, false, true, param->getType()});
  }

  funcDecl->getBody()->accept(this);

  // Return insertion for void functions
  if (funcDecl->getReturnType()->getKind() == shaderpulse::TypeKind::Void) {
    if (auto stmts = dynamic_cast<StatementList*>(funcDecl->getBody())) {
      if (stmts->getStatements().size() == 0) {
        builder.create<spirv::ReturnOp>(builder.getUnknownLoc());
      } else {
        auto &lastStmt = stmts->getStatements().back();

        if (!dynamic_cast<ReturnStatement*>(lastStmt.get())) {
          builder.create<spirv::ReturnOp>(builder.getUnknownLoc());
        }
      }
    } else if (dynamic_cast<Statement*>(funcDecl->getBody()) && !dynamic_cast<ReturnStatement*>(funcDecl->getBody())) {
        builder.create<spirv::ReturnOp>(builder.getUnknownLoc());
    }
  }

  inGlobalScope = true;

  functionMap.insert({funcDecl->getName(), funcOp});

  builder.setInsertionPointToEnd(spirvModule.getBody());

  if (execModeOp) {
    builder.insert(execModeOp);
  }
}

void MLIRCodeGen::visit(DefaultLabel *defaultLabel) {}

void MLIRCodeGen::visit(CaseLabel *defaultLabel) {}

void MLIRCodeGen::generateLoop(Statement* initStmt, Expression* conditionExpr, Expression* inductionExpr, Statement* bodyStmt, bool isDoWhile) {
  Block *restoreInsertionBlock = builder.getInsertionBlock();
  SymbolTableScopeT varScope(symbolTable);

  mlir::Type boolType = mlir::IntegerType::get(&context, 1, mlir::IntegerType::Signless);
  spirv::PointerType ptrType = spirv::PointerType::get(boolType, mlir::spirv::StorageClass::Function);
  breakStack.push_back(
    builder.create<spirv::VariableOp>(
      builder.getUnknownLoc(), ptrType, spirv::StorageClass::Function, nullptr)
  );
  continueStack.push_back(
    builder.create<spirv::VariableOp>(
      builder.getUnknownLoc(), ptrType, spirv::StorageClass::Function, nullptr)
  );

  setBoolVar(continueStack.back(), false);
  setBoolVar(breakStack.back(), false);

  if (initStmt) {
    initStmt->accept(this);
  }

  auto loc = builder.getUnknownLoc();
  auto loopOp = builder.create<spirv::LoopOp>(loc, spirv::LoopControl::None);
  loopOp.addEntryAndMergeBlock(builder);
  auto header = new Block();

  // Insert the header.
  loopOp.getBody().getBlocks().insert(std::next(loopOp.getBody().begin(), 1), header);

  // Insert the body.
  Block *body = new Block();
  loopOp.getBody().getBlocks().insert(std::next(loopOp.getBody().begin(), 2), body);

  // Insert the continue block.
  Block *continueBlock = new Block();
  loopOp.getBody().getBlocks().insert(std::next(loopOp.getBody().begin(), 3), continueBlock);

  // Emit the entry code.
  Block *entry = loopOp.getEntryBlock();
  builder.setInsertionPointToEnd(entry);
  builder.create<spirv::BranchOp>(loc, header);

  // Emit the header code.
  builder.setInsertionPointToEnd(header);
  Block *merge = loopOp.getMergeBlock();

  if (isDoWhile) {
    builder.create<spirv::BranchOp>(loc, &*std::next(loopOp.getBody().begin(), 2));
  } else {
    conditionExpr->accept(this);
    auto conditionOp = load(popExpressionStack());
    builder.create<spirv::BranchConditionalOp>(loc, conditionOp, body, ArrayRef<mlir::Value>(), merge, ArrayRef<mlir::Value>());
  }


  builder.setInsertionPointToStart(body);

  // Detect break/continue flag
  int postGateBlockInsertionPoint = 2;

  if (auto body = dynamic_cast<StatementList*>(bodyStmt)) {
    for (auto &stmt : body->getStatements()) {
      stmt->accept(this);

      if (breakDetected || continueDetected) {
        if (breakDetected && continueDetected) {
          auto continueGate = continueStack.back();
          auto breakGate = breakStack.back();
          Block *breakCheckBlock = new Block();
          loopOp.getBody().getBlocks().insert(std::next(loopOp.getBody().begin(), ++postGateBlockInsertionPoint), breakCheckBlock);
          builder.create<spirv::BranchConditionalOp>(loc, load(continueGate), loopOp.getContinueBlock(), ArrayRef<mlir::Value>(), breakCheckBlock, ArrayRef<mlir::Value>());
          Block *postGateBlock = new Block();
          loopOp.getBody().getBlocks().insert(std::next(loopOp.getBody().begin(), ++postGateBlockInsertionPoint), postGateBlock);
          builder.setInsertionPointToStart(breakCheckBlock);
          builder.create<spirv::BranchConditionalOp>(loc, load(breakGate), merge, ArrayRef<mlir::Value>(), postGateBlock, ArrayRef<mlir::Value>());
          builder.setInsertionPointToStart(postGateBlock);
        } else if (continueDetected) {
          auto continueGate = continueStack.back();
          Block *postGateBlock = new Block();
          loopOp.getBody().getBlocks().insert(std::next(loopOp.getBody().begin(), ++postGateBlockInsertionPoint), postGateBlock);
          builder.create<spirv::BranchConditionalOp>(loc, load(continueGate), loopOp.getContinueBlock(), ArrayRef<mlir::Value>(), postGateBlock, ArrayRef<mlir::Value>());
          builder.setInsertionPointToStart(postGateBlock);
        } else if (breakDetected) {
          auto breakGate = breakStack.back();
          Block *postGateBlock = new Block();
          loopOp.getBody().getBlocks().insert(std::next(loopOp.getBody().begin(), ++postGateBlockInsertionPoint), postGateBlock);
          builder.create<spirv::BranchConditionalOp>(loc, load(breakGate), merge, ArrayRef<mlir::Value>(), postGateBlock, ArrayRef<mlir::Value>());
          builder.setInsertionPointToStart(postGateBlock);
        }

        breakDetected = false;
        continueDetected = false;
      }
    }
  } else {
    bodyStmt->accept(this);
  }

  builder.create<spirv::BranchOp>(loc, loopOp.getContinueBlock());
  builder.setInsertionPointToEnd(loopOp.getContinueBlock());
  setBoolVar(continueStack.back(), false);

  if (inductionExpr) {
    inductionExpr->accept(this);
  }

  if (isDoWhile) {
    conditionExpr->accept(this);
    auto conditionOp = load(popExpressionStack());
    builder.create<spirv::BranchConditionalOp>(loc, conditionOp, header, ArrayRef<mlir::Value>(), merge, ArrayRef<mlir::Value>());
  } else {
    builder.create<spirv::BranchOp>(loc, header);
  }

  builder.setInsertionPointToEnd(restoreInsertionBlock);
  breakStack.pop_back();
  continueStack.pop_back();
}

mlir::Value MLIRCodeGen::load(mlir::Value val) {
  if (val.getType().isa<spirv::PointerType>()) {
    return builder.create<spirv::LoadOp>(builder.getUnknownLoc(), val);
  }

  return val;
}

mlir::Value MLIRCodeGen::buildBoolConst(bool val) {
  auto type = builder.getIntegerType(1);
  return builder.create<spirv::ConstantOp>(builder.getUnknownLoc(), type, IntegerAttr::get(type, APInt(1, val)));
}

mlir::Value MLIRCodeGen::buildIntConst(uint32_t val, bool isSigned) {
  return builder.create<spirv::ConstantOp>(
    builder.getUnknownLoc(),
    mlir::IntegerType::get(&context, 32, isSigned ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned),
    isSigned ? builder.getSI32IntegerAttr(static_cast<int32_t>(val)) : builder.getUI32IntegerAttr(val)
  );
}

mlir::Value MLIRCodeGen::buildFloatConst(double val, bool isDouble) {
  return builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(),
      isDouble ? mlir::FloatType::getF64(&context) : mlir::FloatType::getF32(&context),
      isDouble ? builder.getF64FloatAttr(val) : builder.getF32FloatAttr(static_cast<float>(val))
  );
}

mlir::Value MLIRCodeGen::buildVecConst(mlir::Value constant, mlir::Type type) {
  std::vector<mlir::Value> operands;

  for (int i = 0; i < type.dyn_cast<mlir::VectorType>().getShape()[0]; i++) {
    operands.push_back(constant);
  }

  return builder.create<spirv::CompositeConstructOp>(builder.getUnknownLoc(), type, operands);
}

bool MLIRCodeGen::callBuiltIn(CallExpression* exp) {
  auto builtinFuncIt = builtInFuncMap.find(exp->getFunctionName());

  if (builtinFuncIt != builtInFuncMap.end()) {
    std::vector<mlir::Value> operands;

    for (auto &arg : exp->getArguments()) {
      arg->accept(this);
      operands.push_back(load(popExpressionStack()));
    }
    expressionStack.push_back(builtinFuncIt->second(context, builder, operands));
    return true;
  } else {
    return false;
  }
}

}; // namespace codegen

}; // namespace shaderpulse
