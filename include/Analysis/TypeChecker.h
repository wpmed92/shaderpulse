#pragma once
#include "../AST/AST.h"
#include "../AST/Types.h"
#include "../AST/ASTVisitor.h"
#include "Scope.h"

namespace shaderpulse {

using namespace ast;

namespace analysis {

class TypeChecker : public ASTVisitor {

public:
  TypeChecker() {}
  void visit(TranslationUnit *) override;
  void visit(BinaryExpression *) override;
  void visit(UnaryExpression *) override;
  void visit(ConditionalExpression *) override;
  void visit(StructDeclaration *) override;
  void visit(VariableDeclaration *) override;
  void visit(VariableDeclarationList *) override;
  void visit(SwitchStatement *) override;
  void visit(WhileStatement *) override;
  void visit(ForStatement *) override;
  void visit(DoStatement *) override;
  void visit(IfStatement *) override;
  void visit(AssignmentExpression *) override;
  void visit(StatementList *) override;
  void visit(CallExpression *) override;
  void visit(VariableExpression *) override;
  void visit(IntegerConstantExpression *) override;
  void visit(UnsignedIntegerConstantExpression *) override;
  void visit(FloatConstantExpression *) override;
  void visit(ConstructorExpression *) override;
  void visit(InitializerExpression *) override;
  void visit(DoubleConstantExpression *) override;
  void visit(MemberAccessExpression *) override;
  void visit(ArrayAccessExpression *) override;
  void visit(BoolConstantExpression *) override;
  void visit(ReturnStatement *) override;
  void visit(BreakStatement *) override;
  void visit(ContinueStatement *) override;
  void visit(DiscardStatement *) override;
  void visit(FunctionDeclaration *) override;
  void visit(DefaultLabel *) override;
  void visit(CaseLabel *) override;

private:
    ScopeManager scopeManager;
    std::vector<Type*> typeStack;
    Type* currentFunctionReturnType;
    bool matchTypes(Type* a, Type* b);
};

} // namespace analysis

} // namespace shaderpulse