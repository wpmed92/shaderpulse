#include "CodeGen/MLIRCodeGen.h"
#include "AST/AST.h"
#include "CodeGen/TypeConversion.h"
#include <iostream>

namespace shaderpulse {

using namespace ast;

namespace codegen {

MLIRCodeGen::MLIRCodeGen() : builder(&context), globalScope(symbolTable) {
  context.getOrLoadDialect<spirv::SPIRVDialect>();
  initModuleOp();
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

bool MLIRCodeGen::verify() { return !failed(mlir::verify(spirvModule)); }

void MLIRCodeGen::insertEntryPoint() {
  builder.setInsertionPointToEnd(spirvModule.getBody());
  builder.create<spirv::EntryPointOp>(builder.getUnknownLoc(), spirv::ExecutionModelAttr::get(&context, spirv::ExecutionModel::Fragment),
                                        SymbolRefAttr::get(&context, "main"),
                                        ArrayAttr::get(&context, interface));
}

void MLIRCodeGen::visit(TranslationUnit *unit) {
  builder.setInsertionPointToEnd(spirvModule.getBody());
  
  for (auto &extDecl : unit->getExternalDeclarations()) {
    extDecl->accept(this);
  }

  insertEntryPoint();
}

std::pair<Type*, mlir::Value> MLIRCodeGen::popExpressionStack() {
  auto val = expressionStack.back();
  expressionStack.pop_back();
  return val;
}

void MLIRCodeGen::visit(BinaryExpression *binExp) {
  binExp->getLhs()->accept(this);
  binExp->getRhs()->accept(this);

  std::pair<Type*, mlir::Value> rhsPair = popExpressionStack();
  std::pair<Type*, mlir::Value> lhsPair = popExpressionStack();

  mlir::Value rhs = load(rhsPair.second);
  mlir::Value lhs = load(lhsPair.second);
  Type* typeContext = lhsPair.first;

  // TODO: implement source location
  auto loc = builder.getUnknownLoc();
  mlir::Value val;

  switch (binExp->getOp()) {
  case BinaryOperator::Add:
    if (typeContext->isIntLike()) {
      val = builder.create<spirv::IAddOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::FAddOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::Sub:
    if (typeContext->isIntLike()) {
      val = builder.create<spirv::ISubOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::FSubOp>(loc, lhs, rhs);
    }

    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::Mul:
  if (typeContext->isIntLike()) {
      val = builder.create<spirv::IMulOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::FMulOp>(loc, lhs, rhs);
    }

    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::Div:
    if (typeContext->isUintLike()) {
      val = builder.create<spirv::UDivOp>(loc, lhs, rhs);
    } else if (typeContext->isIntLike()) {
      val = builder.create<spirv::SDivOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::FDivOp>(loc, lhs, rhs);
    }

    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::Mod:
    if (typeContext->isIntLike()) {
      val = builder.create<spirv::SRemOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::FRemOp>(loc, lhs, rhs);
    }

    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::ShiftLeft:
    val = builder.create<spirv::ShiftLeftLogicalOp>(loc, lhs, rhs);
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::ShiftRight:
    val = builder.create<spirv::ShiftRightLogicalOp>(loc, lhs, rhs);
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::Lt:
    if (typeContext->isFloatLike()) {
      val = builder.create<spirv::FOrdLessThanOp>(loc, lhs, rhs);
    } else if (typeContext->isUintLike()) {
      val = builder.create<spirv::ULessThanOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::SLessThanOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::Gt:
    if (typeContext->isFloatLike()) {
      val = builder.create<spirv::FOrdGreaterThanOp>(loc, lhs, rhs);
    } else if (typeContext->isUintLike()) {
      val = builder.create<spirv::UGreaterThanOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::SGreaterThanOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::LtEq:
    if (typeContext->isFloatLike()) {
      val = builder.create<spirv::FOrdLessThanEqualOp>(loc, lhs, rhs);
    } else if (typeContext->isUintLike()) {
      val = builder.create<spirv::ULessThanEqualOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::SLessThanEqualOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::GtEq:
    if (typeContext->isFloatLike()) {
      val = builder.create<spirv::FOrdGreaterThanEqualOp>(loc, lhs, rhs);
    } else if (typeContext->isUintLike()) {
      val = builder.create<spirv::UGreaterThanEqualOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::SGreaterThanEqualOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::Eq:
    if (typeContext->isFloatLike()) {
      val = builder.create<spirv::FOrdEqualOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::IEqualOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::Neq:
    if (typeContext->isFloatLike()) {
      val = builder.create<spirv::FOrdNotEqualOp>(loc, lhs, rhs);
    } else {
      val = builder.create<spirv::INotEqualOp>(loc, lhs, rhs);
    }
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::BitAnd:
    val = builder.create<spirv::BitwiseAndOp>(loc, lhs, rhs);
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::BitXor:
    val = builder.create<spirv::BitwiseXorOp>(loc, lhs, rhs);
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::BitIor:
    val = builder.create<spirv::BitwiseOrOp>(loc, lhs, rhs);
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::LogAnd:
    val = builder.create<spirv::LogicalAndOp>(loc, lhs, rhs);
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  case BinaryOperator::LogXor:
    // TODO: not implemented in current spirv dialect
    break;
  case BinaryOperator::LogOr:
    val = builder.create<spirv::LogicalOrOp>(loc, lhs, rhs);
    expressionStack.push_back(std::make_pair(typeContext, val));
    break;
  }
}

void MLIRCodeGen::visit(ConditionalExpression *condExp) {
  condExp->getFalsePart()->accept(this);
  condExp->getTruePart()->accept(this);
  condExp->getCondition()->accept(this);

  std::pair<Type*, mlir::Value> condition = popExpressionStack();
  std::pair<Type*, mlir::Value> truePart = popExpressionStack();
  std::pair<Type*, mlir::Value> falsePart = popExpressionStack();

  mlir::Value res = builder.create<spirv::SelectOp>(
    builder.getUnknownLoc(),
    convertShaderPulseType(&context, truePart.first, structDeclarations),
    condition.second,
    truePart.second,
    falsePart.second);

  expressionStack.push_back(std::make_pair(truePart.first, res));
}

void MLIRCodeGen::visit(ForStatement *forStmt) {
  // TODO: implement me
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

  expressionStack.push_back(std::make_pair(nullptr, val));
}

void MLIRCodeGen::visit(UnaryExpression *unExp) {
  unExp->getExpression()->accept(this);
  std::pair<Type*, mlir::Value> rhsPair = popExpressionStack();

  auto loc = builder.getUnknownLoc();
  mlir::Value rhs = load(rhsPair.second);
  mlir::Value result;
  Type* rhsType = rhsPair.first;

  auto op = unExp->getOp();

  switch (op) {
  case UnaryOperator::Inc:
  case UnaryOperator::Dec: {
    mlir::Value ptrRhs = rhsPair.second;

    if (rhsType->isIntLike()) {
      auto one = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(),
        mlir::IntegerType::get(&context, 32, rhsType->isUintLike() ? mlir::IntegerType::Unsigned : mlir::IntegerType::Signed),
        rhsType->isUintLike() ? builder.getUI32IntegerAttr(1) :builder.getSI32IntegerAttr(1)
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
    expressionStack.push_back(std::make_pair(rhsType, result));
    break;
  }
  case UnaryOperator::Plus:
    expressionStack.push_back(std::make_pair(rhsPair.first, rhs));
    break;
  case UnaryOperator::Dash:
    if (rhsType->isFloatLike()) {
      result = builder.create<spirv::FNegateOp>(loc, rhs);
    } else {
      result = builder.create<spirv::SNegateOp>(loc, rhs);
    }

    expressionStack.push_back(std::make_pair(rhsType, result));
    break;
  case UnaryOperator::Bang:
    result = builder.create<spirv::LogicalNotOp>(loc, rhs);
    expressionStack.push_back(std::make_pair(rhsType, result));
    break;
  case UnaryOperator::Tilde:
    result = builder.create<spirv::NotOp>(loc, rhs);
    expressionStack.push_back(std::make_pair(rhsType, result));
    break;
  }
}

void MLIRCodeGen::declare(StringRef name, SymbolTableEntry entry) {
  if (symbolTable.count(name)) {
    return;
  }

  symbolTable.insert(name, entry);
}

void MLIRCodeGen::visit(VariableDeclarationList *varDeclList) {
  for (auto &var : varDeclList->getDeclarations()) {
    createVariable(varDeclList->getType(), var.get());
  }
}

void MLIRCodeGen::createVariable(shaderpulse::Type *type,
                                 VariableDeclaration *varDecl) {
  shaderpulse::Type *varType = (type) ? type : varDecl->getType();

  if (inGlobalScope) {
    spirv::StorageClass storageClass;

    if (auto st = getSpirvStorageClass(varType->getQualifier(TypeQualifierKind::Storage))) {
      storageClass = *st;
    } else {
      storageClass = spirv::StorageClass::Private;
    }

    builder.setInsertionPointToEnd(spirvModule.getBody());
  
    spirv::PointerType ptrType = spirv::PointerType::get(
      convertShaderPulseType(&context, varType, structDeclarations), storageClass);

    Operation *initializerOp = nullptr;

    if (expressionStack.size() > 0) {
      mlir::Value val = popExpressionStack().second;
      initializerOp = val.getDefiningOp();
    }

    auto varOp = builder.create<spirv::GlobalVariableOp>(
        UnknownLoc::get(&context), TypeAttr::get(ptrType),
        builder.getStringAttr(varDecl->getIdentifierName()),
        (initializerOp) ? FlatSymbolRefAttr::get(initializerOp) : nullptr);


    auto locationOpt = getLocationFromTypeQualifier(&context, varType->getQualifier(TypeQualifierKind::Layout));

    if (locationOpt) {
      varOp->setAttr("location", *locationOpt);
    }

    declare(varDecl->getIdentifierName(), { mlir::Value(), varDecl, ptrType, /*isGlobal*/ true});
    // Set OpDecorate through attributes
    // example:
    // varOp->setAttr(spirv::stringifyDecoration(spirv::Decoration::Invariant),
    // builder.getUnitAttr());
  } else {
    if (varDecl->getInitialzerExpression()) {
      varDecl->getInitialzerExpression()->accept(this);
    }

    mlir::Value val;

    if (expressionStack.size() > 0) {
      val = load(popExpressionStack().second);
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
  createVariable(nullptr, varDecl);
}

void MLIRCodeGen::visit(SwitchStatement *switchStmt) {

}

void MLIRCodeGen::visit(WhileStatement *whileStmt) {
  Block *restoreInsertionBlock = builder.getInsertionBlock();

  whileStmt->getCondition()->accept(this);

  auto conditionOp = load(popExpressionStack().second);

  auto loc = builder.getUnknownLoc();

  auto loopOp = builder.create<spirv::LoopOp>(loc, spirv::LoopControl::None);
  loopOp.addEntryAndMergeBlock();
  auto header = new Block();

  // Insert the header.
  loopOp.getBody().getBlocks().insert(std::next(loopOp.getBody().begin(), 1), header);

  // Insert the body.
  Block *body = new Block();
  loopOp.getBody().getBlocks().insert(std::next(loopOp.getBody().begin(), 2), body);

  // Emit the entry code.
  Block *entry = loopOp.getEntryBlock();
  builder.setInsertionPointToEnd(entry);
  builder.create<spirv::BranchOp>(loc, header);

  // Emit the header code.
  builder.setInsertionPointToEnd(header);

  Block *merge = loopOp.getMergeBlock();
  builder.create<spirv::BranchConditionalOp>(
      loc, conditionOp, body, ArrayRef<mlir::Value>(), merge, ArrayRef<mlir::Value>());

  // Emit the continue/latch block.
  Block *continueBlock = loopOp.getContinueBlock();
  builder.setInsertionPointToEnd(continueBlock);
  builder.create<spirv::BranchOp>(loc, header);
  builder.setInsertionPointToStart(body);

  whileStmt->getBody()->accept(this);

  builder.setInsertionPointToEnd(restoreInsertionBlock);
}

void MLIRCodeGen::visit(ConstructorExpression *constructorExp) {
  auto constructorType = constructorExp->getType();

  std::vector<mlir::Value> operands;

  if (constructorExp->getArguments().size() > 0) {
    for (auto &arg : constructorExp->getArguments()) {
      arg->accept(this);
      operands.push_back(popExpressionStack().second);
    }
  }

  switch (constructorType->getKind()) {
    case TypeKind::Struct: {
      auto structName = dynamic_cast<StructType*>(constructorType)->getName();

      if (structDeclarations.find(structName) != structDeclarations.end()) {
        mlir::Value val = builder.create<spirv::CompositeConstructOp>(
              builder.getUnknownLoc(), convertShaderPulseType(&context, constructorType, structDeclarations), operands);
        expressionStack.push_back(std::make_pair(constructorType, val));
      }

      break;
    }

    case TypeKind::Vector:
    case TypeKind::Array: {
      mlir::Value val = builder.create<spirv::CompositeConstructOp>(
            builder.getUnknownLoc(), convertShaderPulseType(&context, constructorType, structDeclarations), operands);
      expressionStack.push_back(std::make_pair(constructorType, val));
      break;
    }

    case TypeKind::Matrix: {
      auto matrixType = dynamic_cast<shaderpulse::MatrixType *>(constructorType);
      std::vector<mlir::Value> columnVectors;

      for (int i = 0; i < matrixType->getCols(); i++) {
        std::vector<mlir::Value> col;

        for (int j = 0; j < matrixType->getRows(); j++) {
          col.push_back(operands[j*matrixType->getCols() + i]);
        }

        auto elementType = std::make_unique<Type>(matrixType->getElementType()->getKind());
        auto vecType = std::make_unique<shaderpulse::VectorType>(std::move(elementType), col.size());
        mlir::Value val = builder.create<spirv::CompositeConstructOp>(
          builder.getUnknownLoc(), convertShaderPulseType(&context, vecType.get(), structDeclarations), col);
        columnVectors.push_back(val);
      }

      mlir::Value val = builder.create<spirv::CompositeConstructOp>(
        builder.getUnknownLoc(), convertShaderPulseType(&context, constructorType, structDeclarations), columnVectors);

      expressionStack.push_back(std::make_pair(constructorType, val));
      break;
    }

    default:
      // Unsupported composite construct
      break;
  }
}

void MLIRCodeGen::visit(ArrayAccessExpression *arrayAccess) {
  auto array = arrayAccess->getArray();
  array->accept(this);
  std::pair<Type*, mlir::Value> mlirArray = popExpressionStack();
  Type* elementType = dynamic_cast<ArrayType*>(mlirArray.first)->getElementType();
  std::vector<mlir::Value> indices;

  for (auto &access : arrayAccess->getAccessChain()) {
    access->accept(this);
    auto val = popExpressionStack().second;
    indices.push_back(load(val));
  }

  mlir::Value accessChain = builder.create<spirv::AccessChainOp>(builder.getUnknownLoc(), mlirArray.second, indices);
  expressionStack.push_back(std::make_pair(elementType, accessChain));
}

void MLIRCodeGen::visit(MemberAccessExpression *memberAccess) {
  auto baseComposite = memberAccess->getBaseComposite();
  baseComposite->accept(this);
  mlir::Value baseCompositeValue = popExpressionStack().second;
  std::vector<mlir::Value> memberIndicesAcc;
  Type* memberType;

  if (currentBaseComposite) {
    for (auto &member : memberAccess->getMembers()) {
      if (auto var = dynamic_cast<VariableExpression*>(member.get())) {
        auto memberIndexPair = currentBaseComposite->getMemberWithIndex(var->getName());
        memberIndicesAcc.push_back(builder.create<spirv::ConstantOp>(builder.getUnknownLoc(), mlir::IntegerType::get(&context, 32, mlir::IntegerType::Signless), builder.getI32IntegerAttr(memberIndexPair.first)));

        if (memberIndexPair.second->getType()->getKind() == TypeKind::Struct) {
          auto structName = dynamic_cast<StructType*>(memberIndexPair.second->getType())->getName();

          if (structDeclarations.find(structName) != structDeclarations.end()) {
            currentBaseComposite = structDeclarations[structName];
          }
        }

        memberType = memberIndexPair.second->getType();
      // This is a duplicate of ArrayAccessExpression, idially we want to reuse that.
      } else if (auto arrayAccess = dynamic_cast<ArrayAccessExpression*>(member.get())) {
        auto varName = dynamic_cast<VariableExpression*>(arrayAccess->getArray())->getName();
        auto memberIndexPair = currentBaseComposite->getMemberWithIndex(varName);
        memberIndicesAcc.push_back(builder.create<spirv::ConstantOp>(builder.getUnknownLoc(), mlir::IntegerType::get(&context, 32, mlir::IntegerType::Signless), builder.getI32IntegerAttr(memberIndexPair.first)));
        memberType = dynamic_cast<ArrayType*>(memberIndexPair.second->getType())->getElementType();

        for (auto &access : arrayAccess->getAccessChain()) {
          access->accept(this);
          auto val = popExpressionStack().second;
          memberIndicesAcc.push_back(load(val));
        }
      }
    }

    mlir::Value accessChain = builder.create<spirv::AccessChainOp>(builder.getUnknownLoc(), baseCompositeValue, memberIndicesAcc);
    expressionStack.push_back(std::make_pair(memberType, accessChain));
  }
}

void MLIRCodeGen::visit(StructDeclaration *structDecl) {
  if (structDeclarations.find(structDecl->getName()) == structDeclarations.end()) {
    structDeclarations.insert({structDecl->getName(), structDecl});
  }
}

void MLIRCodeGen::visit(DoStatement *doStmt) {
  // TODO
}

void MLIRCodeGen::visit(IfStatement *ifStmt) {
  Block *restoreInsertionBlock = builder.getInsertionBlock();

  auto loc = builder.getUnknownLoc();

  ifStmt->getCondition()->accept(this);
  mlir::Value condition = load(popExpressionStack().second);

  auto selectionOp = builder.create<spirv::SelectionOp>(loc, spirv::SelectionControl::None);
  selectionOp.addMergeBlock();

  // Merge
  auto *mergeBlock = selectionOp.getMergeBlock();

  // Selection header
  auto *selectionHeaderBlock = new Block();
  selectionOp.getBody().getBlocks().push_front(selectionHeaderBlock);

  // True part
  auto *thenBlock = new Block();
  selectionOp.getBody().getBlocks().insert(std::next(selectionOp.getBody().begin(), 1), thenBlock);
  builder.setInsertionPointToStart(thenBlock);

  ifStmt->getTruePart()->accept(this);
  builder.create<spirv::BranchOp>(loc, mergeBlock);

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
}

void MLIRCodeGen::visit(AssignmentExpression *assignmentExp) {
  assignmentExp->getUnaryExpression()->accept(this);
  assignmentExp->getExpression()->accept(this);

  mlir::Value val = load(popExpressionStack().second);
  mlir::Value ptr = popExpressionStack().second;

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
        operands.push_back(load(popExpressionStack().second));
      }
    }

    spirv::FuncOp calledFunc = calledFuncIt->second;

    spirv::FunctionCallOp funcCall = builder.create<spirv::FunctionCallOp>(
        builder.getUnknownLoc(), calledFunc.getFunctionType().getResults(),
        SymbolRefAttr::get(&context, calledFunc.getSymName()), operands);

    // TODO: get return type of callee
    expressionStack.push_back(std::make_pair(nullptr, funcCall.getResult(0)));
  } else {
    std::cout << "Function not found." << callExp->getFunctionName()
              << std::endl;
  }
}

void MLIRCodeGen::visit(VariableExpression *varExp) {
  auto entry = symbolTable.lookup(varExp->getName());

  if (entry.isFunctionParam) {
    expressionStack.push_back(std::make_pair(entry.type, entry.value));
  } else if (entry.variable) {
    mlir::Value val;

    if (entry.isGlobal) {
      auto addressOfGlobal = builder.create<mlir::spirv::AddressOfOp>(builder.getUnknownLoc(), entry.ptrType, varExp->getName());
      val = addressOfGlobal->getResult(0);

      // If we're inside the entry point function, collect the used global variables
      if (insideEntryPoint) {
        interface.push_back(SymbolRefAttr::get(&context, varExp->getName()));
      }
    } else {
      val = entry.value;
    }

    if (entry.variable->getType()->getKind() == TypeKind::Struct) {
      auto structName = dynamic_cast<StructType*>(entry.variable->getType())->getName();

      if (structDeclarations.find(structName) != structDeclarations.end()) {
        currentBaseComposite = structDeclarations[structName];
      }
    }

    expressionStack.push_back(std::make_pair(entry.variable->getType(), val));
  } else {
    std::cout << "Unable to find variable: " << varExp->getName() << std::endl;
  }
}

void MLIRCodeGen::visit(IntegerConstantExpression *intConstExp) {
  auto type = builder.getIntegerType(32, true);
  mlir::Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      IntegerAttr::get(type, APInt(32, intConstExp->getVal(), true)));

  expressionStack.push_back(std::make_pair(intConstExp->getType(), val));
}

void MLIRCodeGen::visit(UnsignedIntegerConstantExpression *uintConstExp) {
  auto type = builder.getIntegerType(32, false);
  mlir::Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      IntegerAttr::get(type, APInt(32, uintConstExp->getVal(), false)));

  expressionStack.push_back(std::make_pair(uintConstExp->getType(), val));
}

void MLIRCodeGen::visit(FloatConstantExpression *floatConstExp) {
  auto type = builder.getF32Type();
  mlir::Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      FloatAttr::get(type, APFloat(floatConstExp->getVal())));

  expressionStack.push_back(std::make_pair(floatConstExp->getType(), val));
}

void MLIRCodeGen::visit(DoubleConstantExpression *doubleConstExp) {
  auto type = builder.getF64Type();
  mlir::Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      FloatAttr::get(type, APFloat(doubleConstExp->getVal())));

  expressionStack.push_back(std::make_pair(doubleConstExp->getType(), val));
}

void MLIRCodeGen::visit(BoolConstantExpression *boolConstExp) {
  auto type = builder.getIntegerType(1);
  mlir::Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      IntegerAttr::get(type, APInt(1, boolConstExp->getVal())));

  expressionStack.push_back(std::make_pair(boolConstExp->getType(), val));
}

void MLIRCodeGen::visit(ReturnStatement *returnStmt) {
  if (auto retExp = returnStmt->getExpression())
    returnStmt->getExpression()->accept(this);

  if (expressionStack.empty()) {
    builder.create<spirv::ReturnOp>(builder.getUnknownLoc());
  } else {
    mlir::Value val = popExpressionStack().second;
    builder.create<spirv::ReturnValueOp>(builder.getUnknownLoc(), val);
  }
}

void MLIRCodeGen::visit(BreakStatement *breakStmt) {}

void MLIRCodeGen::visit(ContinueStatement *continueStmt) {}

void MLIRCodeGen::visit(DiscardStatement *discardStmt) {}

void MLIRCodeGen::visit(FunctionDeclaration *funcDecl) {
  insideEntryPoint = funcDecl->getName() == "main";
  SymbolTableScopeT varScope(symbolTable);
  std::vector<mlir::Type> paramTypes;

  for (auto &param : funcDecl->getParams()) {
    paramTypes.push_back(convertShaderPulseType(&context, param->getType(), structDeclarations));
  }

  TypeRange resultTypeRange;
  
  if (auto resultType = convertShaderPulseType(&context, funcDecl->getReturnType(), structDeclarations)) {
    if (!resultType.isa<NoneType>()) {
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
  if (funcDecl->getReturnType()->getKind() == TypeKind::Void) {
    if (auto stmts = dynamic_cast<StatementList*>(funcDecl->getBody())) {
      auto &lastStmt = stmts->getStatements().back();

      if (!dynamic_cast<ReturnStatement*>(lastStmt.get())) {
        builder.create<spirv::ReturnOp>(builder.getUnknownLoc());
      }
    } else if (dynamic_cast<Statement*>(funcDecl->getBody()) && !dynamic_cast<ReturnStatement*>(funcDecl->getBody())) {
        builder.create<spirv::ReturnOp>(builder.getUnknownLoc());
    }
  }

  inGlobalScope = true;

  functionMap.insert({funcDecl->getName(), funcOp});

  builder.setInsertionPointToEnd(spirvModule.getBody());
}

void MLIRCodeGen::visit(DefaultLabel *defaultLabel) {}

void MLIRCodeGen::visit(CaseLabel *defaultLabel) {}

mlir::Value MLIRCodeGen::load(mlir::Value val) {
  if (val.getType().isa<spirv::PointerType>()) {
    return builder.create<spirv::LoadOp>(builder.getUnknownLoc(), val);
  }

  return val;
}

}; // namespace codegen

}; // namespace shaderpulse
