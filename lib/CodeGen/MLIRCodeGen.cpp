#include "CodeGen/MLIRCodeGen.h"
#include "AST/AST.h"
#include "CodeGen/TypeConversion.h"
#include <iostream>

namespace shaderpulse {

using namespace ast;

namespace codegen {

void MLIRCodeGen::dump() {
  spirvModule.dump();
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

Value MLIRCodeGen::popExpressionStack() {
  Value val = expressionStack.back();
  expressionStack.pop_back();
  return val;
}

void MLIRCodeGen::visit(BinaryExpression *binExp) {
  binExp->getLhs()->accept(this);
  binExp->getRhs()->accept(this);

  Value rhs = popExpressionStack();
  Value lhs = popExpressionStack();

  // TODO: handle other types than float. Need to figure out the expression this
  // BinaryExpression is part of to know what kind of spirv op to use. (float,
  // int?)

  // TODO: implement source location
  auto loc = builder.getUnknownLoc();
  Value val;

  switch (binExp->getOp()) {
  case BinaryOperator::Add:
    val = builder.create<spirv::FAddOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Sub:
    val = builder.create<spirv::FSubOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Mul:
    val = builder.create<spirv::FMulOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Div:
    val = builder.create<spirv::FDivOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Mod:
    val = builder.create<spirv::FRemOp>(loc, lhs, rhs);
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
    val = builder.create<spirv::FOrdLessThanOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Gt:
    val = builder.create<spirv::FOrdGreaterThanOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::LtEq:
    val = builder.create<spirv::FOrdLessThanEqualOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::GtEq:
    val = builder.create<spirv::FOrdGreaterThanEqualOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Eq:
    val = builder.create<spirv::FOrdEqualOp>(loc, lhs, rhs);
    expressionStack.push_back(val);
    break;
  case BinaryOperator::Neq:
    val = builder.create<spirv::FOrdNotEqualOp>(loc, lhs, rhs);
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

  Value condition = popExpressionStack();
  Value truePart = popExpressionStack();
  Value falsePart = popExpressionStack();

  Value res = builder.create<spirv::SelectOp>(
    builder.getUnknownLoc(),
    /* Harcoded, fix me */ mlir::FloatType::getF32(&context),
    condition,
    truePart,
    falsePart);

  expressionStack.push_back(res);
}

void MLIRCodeGen::visit(ForStatement *forStmt) {
  // TODO: implement me
}

void MLIRCodeGen::visit(UnaryExpression *unExp) {
  unExp->getExpression()->accept(this);
  Value rhs = popExpressionStack();

  auto loc = builder.getUnknownLoc();
  Value val;

  switch (unExp->getOp()) {
  case UnaryOperator::Inc:
    // TODO: implement post-, pre-fix increment
    break;
  case UnaryOperator::Dec:
    // TODO: implement post-, pre-fix decrement
    break;
  case UnaryOperator::Plus:
    expressionStack.push_back(rhs);
    break;
  case UnaryOperator::Dash:
    val = builder.create<spirv::FNegateOp>(loc, rhs);
    expressionStack.push_back(val);
    break;
  case UnaryOperator::Bang:
    val = builder.create<spirv::LogicalNotOp>(loc, rhs);
    expressionStack.push_back(val);
    break;
  case UnaryOperator::Tilde:
    val = builder.create<spirv::NotOp>(loc, rhs);
    expressionStack.push_back(val);
    break;
  }
}

void MLIRCodeGen::declare(SymbolTableEntry entry) {
  if (symbolTable.count(entry.variable->getIdentifierName()))
    return;


  std::cout << "Declaring " << entry.variable->getIdentifierName() << std::endl;
  symbolTable.insert(entry.variable->getIdentifierName(), entry);
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
    std::cout << "In global scope" << std::endl;
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
      Value val = popExpressionStack();
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

    declare({ mlir::Value(), varDecl, ptrType, /*isGlobal*/ true});
    // Set OpDecorate through attributes
    // example:
    // varOp->setAttr(spirv::stringifyDecoration(spirv::Decoration::Invariant),
    // builder.getUnitAttr());
  } else {
    if (varDecl->getInitialzerExpression()) {
      std::cout << "Accept init" << std::endl;
      varDecl->getInitialzerExpression()->accept(this);
    }

    Value val;

    if (expressionStack.size() > 0) {
      val = popExpressionStack();
    }

    spirv::PointerType ptrType = spirv::PointerType::get(
        convertShaderPulseType(&context, varType, structDeclarations), spirv::StorageClass::Function);

    auto var = builder.create<spirv::VariableOp>(
        builder.getUnknownLoc(), ptrType, spirv::StorageClass::Function, nullptr);

    if (varDecl->getInitialzerExpression()) {
      builder.create<spirv::StoreOp>(builder.getUnknownLoc(), var, val);
    }

    declare({ var, varDecl, nullptr, /*isGlobal*/ false});
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

  auto conditionOp = popExpressionStack();

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
      loc, conditionOp, body, ArrayRef<Value>(), merge, ArrayRef<Value>());

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
      operands.push_back(popExpressionStack());
    }
  }

  switch (constructorType->getKind()) {
    case TypeKind::Struct: {
      auto structName = dynamic_cast<StructType*>(constructorType)->getName();

      if (structDeclarations.find(structName) != structDeclarations.end()) {
        mlir::Value val = builder.create<spirv::CompositeConstructOp>(
              builder.getUnknownLoc(), convertShaderPulseType(&context, constructorType, structDeclarations), operands);
        expressionStack.push_back(val);
      }

      break;
    }

    case TypeKind::Vector:
    case TypeKind::Array: {
      mlir::Value val = builder.create<spirv::CompositeConstructOp>(
            builder.getUnknownLoc(), convertShaderPulseType(&context, constructorType, structDeclarations), operands);
      expressionStack.push_back(val);
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

      expressionStack.push_back(val);
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
  Value mlirArray = popExpressionStack();
  std::vector<mlir::Value> indices;

  for (auto &access : arrayAccess->getAccessChain()) {
    access->accept(this);
    indices.push_back(popExpressionStack());
  }

  Value accessChain = builder.create<spirv::AccessChainOp>(builder.getUnknownLoc(), mlirArray, indices);

  if (arrayAccess->isLhs()) {
    expressionStack.push_back(accessChain);
  } else {
    expressionStack.push_back(builder.create<spirv::LoadOp>(builder.getUnknownLoc(), accessChain)->getResult(0));
  }
}

void MLIRCodeGen::visit(MemberAccessExpression *memberAccess) {
  auto baseComposite = memberAccess->getBaseComposite();
  baseComposite->accept(this);
  Value baseCompositeValue = popExpressionStack();
  std::vector<int> memberIndices;
  std::vector<mlir::Value> memberIndicesAcc;

  if (currentBaseComposite) {
    for (auto &member : memberAccess->getMembers()) {
      if (auto var = dynamic_cast<VariableExpression*>(member.get())) {
        auto memberIndexPair = currentBaseComposite->getMemberWithIndex(var->getName());
        memberIndices.push_back(memberIndexPair.first);
        memberIndicesAcc.push_back(builder.create<spirv::ConstantOp>(builder.getUnknownLoc(), mlir::IntegerType::get(&context, 32, mlir::IntegerType::Signless), builder.getI32IntegerAttr(memberIndexPair.first)));

        if (memberIndexPair.second->getType()->getKind() == TypeKind::Struct) {
          auto structName = dynamic_cast<StructType*>(memberIndexPair.second->getType())->getName();

          if (structDeclarations.find(structName) != structDeclarations.end()) {
            currentBaseComposite = structDeclarations[structName];
          }
        }
      }
    }

    if (memberAccess->isLhs()) {
      Value accessChain = builder.create<spirv::AccessChainOp>(builder.getUnknownLoc(), baseCompositeValue, memberIndicesAcc);
      expressionStack.push_back(accessChain);
    } else {
      Value compositeElement = builder.create<spirv::CompositeExtractOp>(builder.getUnknownLoc(), baseCompositeValue, memberIndices);
      expressionStack.push_back(compositeElement);
    }
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
  Value condition = popExpressionStack();

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
      loc, condition, thenBlock, ArrayRef<Value>(), elseBlock, ArrayRef<Value>());
  
  builder.setInsertionPointToEnd(restoreInsertionBlock);
}

void MLIRCodeGen::visit(AssignmentExpression *assignmentExp) {
  assignmentExp->getUnaryExpression()->accept(this);
  assignmentExp->getExpression()->accept(this);

  Value val = popExpressionStack();
  Value ptr = popExpressionStack();

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
        operands.push_back(popExpressionStack());
      }
    }

    spirv::FuncOp calledFunc = calledFuncIt->second;

    spirv::FunctionCallOp funcCall = builder.create<spirv::FunctionCallOp>(
        builder.getUnknownLoc(), calledFunc.getFunctionType().getResults(),
        SymbolRefAttr::get(&context, calledFunc.getSymName()), operands);

    expressionStack.push_back(funcCall.getResult(0));
  } else {
    std::cout << "Function not found." << callExp->getFunctionName()
              << std::endl;
  }
}

void MLIRCodeGen::visit(VariableExpression *varExp) {
  std::cout << "Looking up " << varExp->getName() << std::endl;
  auto entry = symbolTable.lookup(varExp->getName());

  if (entry.variable) {
    std::cout << "Looked up and found " << varExp->getName() << std::endl;
    Value val;

    if (entry.isGlobal) {
      auto addressOfGlobal = builder.create<mlir::spirv::AddressOfOp>(builder.getUnknownLoc(), entry.ptrType, varExp->getName());
      val = varExp->isLhs() ? addressOfGlobal->getResult(0) : builder.create<spirv::LoadOp>(builder.getUnknownLoc(), addressOfGlobal)->getResult(0);

      // If we're inside the entry point function, collect the used global variables
      if (insideEntryPoint) {
        interface.push_back(SymbolRefAttr::get(&context, varExp->getName()));
      }
    } else {
      val = (varExp->isLhs() || (entry.variable->getType()->getKind() == TypeKind::Array)) ? entry.value : builder.create<spirv::LoadOp>(builder.getUnknownLoc(), entry.value);
    }

    if (entry.variable->getType()->getKind() == TypeKind::Struct) {
      auto structName = dynamic_cast<StructType*>(entry.variable->getType())->getName();

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
  Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      IntegerAttr::get(type, APInt(32, intConstExp->getVal(), true)));

  expressionStack.push_back(val);
}

void MLIRCodeGen::visit(UnsignedIntegerConstantExpression *uintConstExp) {
  auto type = builder.getIntegerType(32, false);
  Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      IntegerAttr::get(type, APInt(32, uintConstExp->getVal(), false)));

  expressionStack.push_back(val);
}

void MLIRCodeGen::visit(FloatConstantExpression *floatConstExp) {
  auto type = builder.getF32Type();
  Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      FloatAttr::get(type, APFloat(floatConstExp->getVal())));

  expressionStack.push_back(val);
}

void MLIRCodeGen::visit(DoubleConstantExpression *doubleConstExp) {
  auto type = builder.getF64Type();
  Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      FloatAttr::get(type, APFloat(doubleConstExp->getVal())));

  expressionStack.push_back(val);
}

void MLIRCodeGen::visit(BoolConstantExpression *boolConstExp) {
  auto type = builder.getIntegerType(1);
  Value val = builder.create<spirv::ConstantOp>(
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
    Value val = popExpressionStack();
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

  auto funcOp = builder.create<spirv::FuncOp>(
      builder.getUnknownLoc(), funcDecl->getName(),
      builder.getFunctionType(
          paramTypes,
          resultTypeRange
      )
  );

  inGlobalScope = false;

  auto entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  funcDecl->getBody()->accept(this);

  inGlobalScope = true;

  functionMap.insert({funcDecl->getName(), funcOp});

  builder.setInsertionPointToEnd(spirvModule.getBody());
}

void MLIRCodeGen::visit(DefaultLabel *defaultLabel) {}

void MLIRCodeGen::visit(CaseLabel *defaultLabel) {}

}; // namespace codegen

}; // namespace shaderpulse
