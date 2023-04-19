#include "CodeGen/MLIRCodeGen.h"
#include "CodeGen/TypeConversion.h"
#include "AST/AST.h"
#include <iostream>

namespace shaderpulse {

using namespace ast;

namespace codegen {

void MLIRCodeGen::dump() {
    spirvModule.dump();
}

void MLIRCodeGen::visit(TranslationUnit* unit) {
    std::cout << "Visiting translation" << std::endl;

    if (failed(mlir::verify(spirvModule))) {
      std::cout << "Verify failed" << std::endl;
    } else {
        std::cout << "Verification success" << std::endl;
    }
}

Value MLIRCodeGen::popExpressionStack() {
    Value val = expressionStack.back();
    expressionStack.pop_back();
    return val;
}

void MLIRCodeGen::visit(BinaryExpression* binExp) {
    Value rhs = popExpressionStack();
    Value lhs = popExpressionStack();

    // TODO: handle other types than float. Need to figure out the expression this BinaryExpression is
    // part of to know what kind of spirv op to use. (float, int?)

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

void MLIRCodeGen::visit(UnaryExpression* unExp) {
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

void MLIRCodeGen::declare(ValueDeclaration* valDecl, mlir::Value value) {
    if (symbolTable.count(valDecl->getIdentifierNames()[0]))
      return;

    symbolTable.insert(valDecl->getIdentifierNames()[0], {value, valDecl});
}

void MLIRCodeGen::visit(ValueDeclaration* valDecl) {
    std::cout << "Visitng val decl: " << valDecl->getIdentifierNames()[0] << std::endl;
    if (inGlobalScope) {
        builder.setInsertionPointToEnd(spirvModule.getBody());
        auto ptrType = spirv::PointerType::get(convertShaderPulseType(&context, valDecl->getType()), spirv::StorageClass::Uniform);
        auto var = builder.create<spirv::GlobalVariableOp>(
            UnknownLoc::get(&context), TypeAttr::get(ptrType),
            builder.getStringAttr(valDecl->getIdentifierNames()[0]), nullptr);
    } else {
       auto var = builder.create<spirv::VariableOp>(
          builder.getUnknownLoc(), 
          convertShaderPulseType(&context, valDecl->getType()), 
          spirv::StorageClass::Function,
          nullptr
        );

        declare(valDecl, var);
    }
}

void MLIRCodeGen::visit(SwitchStatement* switchStmt) {
}

void MLIRCodeGen::visit(WhileStatement* whileStmt) {
}

void MLIRCodeGen::visit(DoStatement* doStmt) {
}

void MLIRCodeGen::visit(IfStatement* ifStmt) {
}

void MLIRCodeGen::visit(AssignmentExpression* assignmentExp) {
    std::cout << "Visiting assignment expression" << std::endl;
    auto varIt = symbolTable.lookup(assignmentExp->getIdentifier());
    Value val = popExpressionStack();

    if (varIt.first) {
        std::cout << "Variable found in symbol table and pushed to expr stack." << std::endl;
        builder.create<spirv::StoreOp>(builder.getUnknownLoc(), varIt.first, val);
        std::cout << "Store success" << std::endl;
    }
}

void MLIRCodeGen::visit(StatementList* stmtList) {
}

void MLIRCodeGen::visit(CallExpression* callExp) {
}

void MLIRCodeGen::visit(VariableExpression* varExp) {
    std::cout << "Visiting varexp" << std::endl;
    auto varIt = symbolTable.lookup(varExp->getName());

    if (varIt.first) {
        std::cout << "Variable found in symbol table and pushed to expr stack." << std::endl;
        expressionStack.push_back(varIt.first);
    }
}

void MLIRCodeGen::visit(IntegerConstantExpression* intConstExp) {
    auto type = builder.getIntegerType(32);
    Value val = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(), type, IntegerAttr::get(type, APInt(32, intConstExp->getVal(), true)));

    expressionStack.push_back(val);
}

void MLIRCodeGen::visit(UnsignedIntegerConstantExpression* uintConstExp) {
    auto type = builder.getIntegerType(32);
    Value val = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(), type, IntegerAttr::get(type, APInt(32, uintConstExp->getVal(), false)));

    expressionStack.push_back(val);
}

void MLIRCodeGen::visit(FloatConstantExpression* floatConstExp) {
    std::cout << "Visiting float constant" << std::endl;
    auto type = builder.getF32Type();
    Value val = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(), type, FloatAttr::get(type, APFloat(floatConstExp->getVal())));

    std::cout << "Visiting float constant after" << std::endl;
    expressionStack.push_back(val);
}

void MLIRCodeGen::visit(DoubleConstantExpression* doubleConstExp) {
    auto type = builder.getF64Type();
    Value val = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(), type, FloatAttr::get(type, APFloat(doubleConstExp->getVal())));

    expressionStack.push_back(val);
}

void MLIRCodeGen::visit(BoolConstantExpression* boolConstExp) {
    auto type = builder.getIntegerType(1);
    Value val = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(), type, IntegerAttr::get(type, APInt(1, boolConstExp->getVal())));

    expressionStack.push_back(val);
}

void MLIRCodeGen::visit(ReturnStatement* returnStmt) {
}

void MLIRCodeGen::visit(BreakStatement* breakStmt) {
}

void MLIRCodeGen::visit(ContinueStatement* continueStmt) {
}

void MLIRCodeGen::visit(DiscardStatement* discardStmt) {
}

void MLIRCodeGen::visit(FunctionDeclaration* funcDecl) {
    SymbolTableScopeT varScope(symbolTable);
    std::vector<mlir::Type> paramTypes;

    for (auto& param : funcDecl->getParams()) {
        paramTypes.push_back(convertShaderPulseType(&context, param->getType()));
    }

    builder.setInsertionPointToEnd(spirvModule.getBody());
    
    auto funcOp = builder.create<spirv::FuncOp>(
      builder.getUnknownLoc(), funcDecl->getName(),
      builder.getFunctionType(paramTypes,
                               std::nullopt)
    );

    inGlobalScope = false;

    auto entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    funcDecl->getBody()->accept(this);

    inGlobalScope = true;

    functionMap.insert({funcDecl->getName(), funcOp});
}

void MLIRCodeGen::visit(DefaultLabel* defaultLabel) {
}

void MLIRCodeGen::visit(CaseLabel* defaultLabel) {
}

}; // namespace codegen

}; // namespace shaderpulse