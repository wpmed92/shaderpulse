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
    spirvModule.verify();
    
    if (failed(mlir::verify(spirvModule))) {
      std::cout << "Verify failed" << std::endl;
    } else {
        std::cout << "Verification success" << std::endl;
    }
}

void MLIRCodeGen::visit(BinaryExpression* binExp) {
   /* Value rhs = expressionStack.pop_back();
    Value lhs = expressionStack.pop_back();

    // TODO: handle other types than float. Need to figure out the expression this BinaryExpression is
    // part of to know what kind of spirv op to use. (float, int?)

    switch (binExp->getOp()) {
        case BinaryOperator::Add:
            Value val = builder.create<spirv::FAddOp>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::Sub:
            Value val = builder.create<spirv::FSubOp>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::Mul:
            Value val = builder.create<spirv::FMulOp>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::Div:
            Value val = builder.create<spirv::FDivOp>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::Mod:
            Value val = builder.create<spirv::FRemOp>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::ShiftLeft:
            Value val = builder.create<spirv::ShiftLeftLogical>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::ShiftRight:
            Value val = builder.create<spirv::ShiftRightLogical>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::Lt:
            Value val = builder.create<spirv::FOrdLessThan>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::Gt:
            Value val = builder.create<spirv::FOrdGreaterThan>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::LtEq:
            Value val = builder.create<spirv::FOrdLessThanEqual>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::GtEq:
            Value val = builder.create<spirv::FOrdGreaterThanEqual>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::Eq:
            Value val = builder.create<spirv::FOrdEqual>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::Neq:
            Value val = builder.create<spirv::FOrdNotEqual>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::BitAnd:
            Value val = builder.create<spirv::BitwiseAnd>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::BitXor:
            Value val = builder.create<spirv::BitwiseXor>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::BitIor:
            Value val = builder.create<spirv::BitwiseOr>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::LogAnd:
            Value val = builder.create<spirv::LogicalAnd>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
        case BinaryOperator::LogXor:
            // TODO: not implemented in current spirv dialect
            break;
        case BinaryOperator::LogOr:
            Value val = builder.create<spirv::LogicalOr>(loc, resultTypes, {lhs, rhs});
            expressionStack.push_back(val);
            break;
    }*/
}

void MLIRCodeGen::visit(UnaryExpression* unExp) {
    /*Value rhs = expressionStack.pop_back();

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
            Value val = builder.create<spirv::FNegate>(loc, resultTypes, rhs);
            expressionStack.push_back(val);
            break;
        case UnaryOperator::Bang:
            Value val = builder.create<spirv::LogicalNot>(loc, resultTypes, rhs);
            expressionStack.push_back(val);
            break;
        case UnaryOperator::Tilde:
            Value val = builder.create<spirv::Not>(loc, resultTypes, rhs);
            expressionStack.push_back(val);
            break;
    }*/
}

void MLIRCodeGen::visit(ValueDeclaration* valDecl) {
    builder.setInsertionPointToEnd(spirvModule.getBody());
    std::cout << "Value declaration: " << valDecl->getIdentifierNames()[0] << std::endl;
    auto ptrType = spirv::PointerType::get(convertShaderPulseType(&context, valDecl->getType()), spirv::StorageClass::Uniform);
    builder.create<spirv::GlobalVariableOp>(
        UnknownLoc::get(&context), TypeAttr::get(ptrType),
        builder.getStringAttr(valDecl->getIdentifierNames()[0]), nullptr);
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
}

void MLIRCodeGen::visit(StatementList* stmtList) {
}

void MLIRCodeGen::visit(CallExpression* callExp) {
}

void MLIRCodeGen::visit(VariableExpression* varExp) {

}

void MLIRCodeGen::visit(IntegerConstantExpression* intConstExp) {
    /*auto type = IntegerType();
    Value val = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(), type, builder.getIntegerAttr(type, APInt(32, intConstExp->getVal(), true)));

    expressionStack.push_back(val);*/
}

void MLIRCodeGen::visit(UnsignedIntegerConstantExpression* uintConstExp) {
    /*auto type = IntegerType();
    Value val = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(), type, builder.getIntegerAttr(type, APInt(32, intConstExp->getVal(), false)));

    expressionStack.push_back(val);*/
}

void MLIRCodeGen::visit(FloatConstantExpression* floatConstExp) {
    /*auto type = FloatType();
    Value val = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(), type, builder.getIntegerAttr(type, APFloat(floatConstExp->getVal())));

    expressionStack.push_back(val);*/
}

void MLIRCodeGen::visit(DoubleConstantExpression* doubleConstExp) {
    /*auto type = FloatType();
    Value val = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(), type, builder.getIntegerAttr(type, APFloat(doubleConstExp->getVal())));

    expressionStack.push_back(val);*/
}

void MLIRCodeGen::visit(BoolConstantExpression* boolConstExp) {
    /*auto type = IntegerType();
    Value val = builder.create<spirv::ConstantOp>(
        builder.getUnknownLoc(), type, builder.getIntegerAttr(type, APInt(1, boolConstExp->getVal())));*/
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
    /*for (auto argType : enumerate(funcOp.getType().getInputs())) {
        auto convertedType = typeConverter.convertType(argType.value());
        signatureConverter.addInputs(argType.index(), convertedType);
    }

    auto newFuncOp = builder.create<spirv::FuncOp>(
      builder.getUnknownLoc(), funcDecl->getName(),
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               llvm::None));*/
}

void MLIRCodeGen::visit(DefaultLabel* defaultLabel) {
}

void MLIRCodeGen::visit(CaseLabel* defaultLabel) {
}

}; // namespace codegen

}; // namespace shaderpulse