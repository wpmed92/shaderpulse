#pragma once

namespace shaderpulse {

namespace ast {

class TranslationUnit;
class BinaryExpression;
class UnaryExpression;
class VariableDeclaration;
class VariableDeclarationList;
class SwitchStatement;
class WhileStatement;
class DoStatement;
class IfStatement;
class AssignmentExpression;
class StatementList;
class CallExpression;
class ConstructorExpression;
class VariableExpression;
class StructDeclaration;
class IntegerConstantExpression;
class UnsignedIntegerConstantExpression;
class FloatConstantExpression;
class DoubleConstantExpression;
class BoolConstantExpression;
class ReturnStatement;
class BreakStatement;
class ContinueStatement;
class DiscardStatement;
class FunctionDeclaration;
class DefaultLabel;
class CaseLabel;

class ASTVisitor {

public:
  virtual ~ASTVisitor() = default;
  virtual void visit(TranslationUnit *) { };
  virtual void visit(BinaryExpression *) { };
  virtual void visit(UnaryExpression *) { };
  virtual void visit(StructDeclaration *) { };
  virtual void visit(VariableDeclaration *) { };
  virtual void visit(VariableDeclarationList *) { };
  virtual void visit(SwitchStatement *) { };
  virtual void visit(WhileStatement *) { };
  virtual void visit(DoStatement *) { };
  virtual void visit(IfStatement *) { };
  virtual void visit(AssignmentExpression *) { };
  virtual void visit(StatementList *) { };
  virtual void visit(CallExpression *) { };
  virtual void visit(ConstructorExpression *) { };
  virtual void visit(VariableExpression *) { };
  virtual void visit(IntegerConstantExpression *) { };
  virtual void visit(UnsignedIntegerConstantExpression *) { };
  virtual void visit(FloatConstantExpression *) { };
  virtual void visit(DoubleConstantExpression *) { };
  virtual void visit(BoolConstantExpression *) { };
  virtual void visit(ReturnStatement *) { };
  virtual void visit(BreakStatement *) { };
  virtual void visit(ContinueStatement *) { };
  virtual void visit(DiscardStatement *) { };
  virtual void visit(FunctionDeclaration *) { };
  virtual void visit(DefaultLabel *) { };
  virtual void visit(CaseLabel *) { };
};

}; // namespace ast

}; // namespace shaderpulse
