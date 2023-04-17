#pragma once
#include "AST/ASTVisitor.h"

namespace shaderpulse {

using namespace ast;

namespace codegen {

class MLIRCodeGen : public ASTVisitor {

public:
    void visit(TranslationUnit*) override;
    void visit(BinaryExpression*) override;;
    void visit(UnaryExpression*) override;;
    void visit(ValueDeclaration*) override;
    void visit(SwitchStatement*) override;
    void visit(WhileStatement*) override;
    void visit(DoStatement*) override;
    void visit(IfStatement*) override;
    void visit(AssignmentExpression*) override;
    void visit(StatementList*) override;
    void visit(CallExpression*) override;
    void visit(VariableExpression*) override;
    void visit(IntegerConstantExpression*) override;
    void visit(UnsignedIntegerConstantExpression*) override;
    void visit(FloatConstantExpression*) override;
    void visit(DoubleConstantExpression*) override;
    void visit(BoolConstantExpression*) override;
    void visit(ReturnStatement*) override;
    void visit(BreakStatement*) override;
    void visit(ContinueStatement*) override;
    void visit(DiscardStatement*) override;
    void visit(FunctionDeclaration*) override;
    void visit(DefaultLabel*) override;
    void visit(CaseLabel*) override;

};

};

};