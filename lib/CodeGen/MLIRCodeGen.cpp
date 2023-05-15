#include "CodeGen/MLIRCodeGen.h"
#include "AST/AST.h"
#include "CodeGen/TypeConversion.h"
#include <iostream>

namespace shaderpulse {

using namespace ast;

namespace codegen {

void MLIRCodeGen::dump() { spirvModule.dump(); }

bool MLIRCodeGen::verify() { return !failed(mlir::verify(spirvModule)); }

void MLIRCodeGen::visit(TranslationUnit *unit) {
  builder.setInsertionPointToEnd(spirvModule.getBody());
  
  for (auto &extDecl : unit->getExternalDeclarations()) {
    extDecl->accept(this);
  }
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

void MLIRCodeGen::declare(VariableDeclaration *varDecl, mlir::Value value) {
  if (symbolTable.count(varDecl->getIdentifierName()))
    return;

  symbolTable.insert(varDecl->getIdentifierName(), {value, varDecl});
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

    if (auto st = getSpirvStorageClass(
            varType->getQualifier(TypeQualifierKind::Storage))) {
      storageClass = *st;
    } else {
      storageClass = spirv::StorageClass::Private;
    }

    builder.setInsertionPointToEnd(spirvModule.getBody());
    auto ptrType = spirv::PointerType::get(
        convertShaderPulseType(&context, varType), storageClass);

    Operation *initializerOp = nullptr;

    if (expressionStack.size() > 0) {
      Value val = popExpressionStack();
      initializerOp = val.getDefiningOp();
    }

    auto varOp = builder.create<spirv::GlobalVariableOp>(
        UnknownLoc::get(&context), TypeAttr::get(ptrType),
        builder.getStringAttr(varDecl->getIdentifierName()),
        (initializerOp) ? FlatSymbolRefAttr::get(initializerOp) : nullptr);

    // Set OpDecorate through attributes
    // example:
    // varOp->setAttr(spirv::stringifyDecoration(spirv::Decoration::Invariant),
    // builder.getUnitAttr());
  } else {
    if (varDecl->getInitialzerExpression())
      varDecl->getInitialzerExpression()->accept(this);

    Value val;

    if (expressionStack.size() > 0) {
      val = popExpressionStack();
    }

    auto ptrType =
        spirv::PointerType::get(convertShaderPulseType(&context, varType),
                                spirv::StorageClass::Function);

    auto var = builder.create<spirv::VariableOp>(
        builder.getUnknownLoc(), ptrType, spirv::StorageClass::Function, nullptr);

    builder.create<spirv::StoreOp>(builder.getUnknownLoc(), var, val);

    declare(varDecl, var);
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
    case TypeKind::Vector: {
      mlir::Value val = builder.create<spirv::CompositeConstructOp>(
            builder.getUnknownLoc(), convertShaderPulseType(&context, constructorType), operands);
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
          builder.getUnknownLoc(), convertShaderPulseType(&context, vecType.get()), col);
        columnVectors.push_back(val);
      }

      mlir::Value val = builder.create<spirv::CompositeConstructOp>(
        builder.getUnknownLoc(), convertShaderPulseType(&context, constructorType), columnVectors);

      expressionStack.push_back(val);
      break;
    }

    default:
      // Unsupported composite construct
      break;
  }
}

void MLIRCodeGen::visit(DoStatement *doStmt) {

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
  auto varIt = symbolTable.lookup(assignmentExp->getIdentifier());
  assignmentExp->getExpression()->accept(this);
  Value val = popExpressionStack();

  if (varIt.first) {
    builder.create<spirv::StoreOp>(builder.getUnknownLoc(), varIt.first, val);
  }
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
  auto varIt = symbolTable.lookup(varExp->getName());

  if (varIt.first) {
    Value val =
        builder.create<spirv::LoadOp>(builder.getUnknownLoc(), varIt.first);
    expressionStack.push_back(val);
  }
}

void MLIRCodeGen::visit(IntegerConstantExpression *intConstExp) {
  auto type = builder.getIntegerType(32);
  Value val = builder.create<spirv::ConstantOp>(
      builder.getUnknownLoc(), type,
      IntegerAttr::get(type, APInt(32, intConstExp->getVal(), true)));

  expressionStack.push_back(val);
}

void MLIRCodeGen::visit(UnsignedIntegerConstantExpression *uintConstExp) {
  auto type = builder.getIntegerType(32);
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
  SymbolTableScopeT varScope(symbolTable);
  std::vector<mlir::Type> paramTypes;

  for (auto &param : funcDecl->getParams()) {
    paramTypes.push_back(convertShaderPulseType(&context, param->getType()));
  }

  auto funcOp = builder.create<spirv::FuncOp>(
      builder.getUnknownLoc(), funcDecl->getName(),
      builder.getFunctionType(
          paramTypes,
          convertShaderPulseType(&context, funcDecl->getReturnType())));

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
