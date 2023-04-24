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
class VariableExpression;
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
  virtual void visit(TranslationUnit *) = 0;
  virtual void visit(BinaryExpression *) = 0;
  virtual void visit(UnaryExpression *) = 0;
  virtual void visit(VariableDeclaration *) = 0;
  virtual void visit(VariableDeclarationList *) = 0;
  virtual void visit(SwitchStatement *) = 0;
  virtual void visit(WhileStatement *) = 0;
  virtual void visit(DoStatement *) = 0;
  virtual void visit(IfStatement *) = 0;
  virtual void visit(AssignmentExpression *) = 0;
  virtual void visit(StatementList *) = 0;
  virtual void visit(CallExpression *) = 0;
  virtual void visit(VariableExpression *) = 0;
  virtual void visit(IntegerConstantExpression *) = 0;
  virtual void visit(UnsignedIntegerConstantExpression *) = 0;
  virtual void visit(FloatConstantExpression *) = 0;
  virtual void visit(DoubleConstantExpression *) = 0;
  virtual void visit(BoolConstantExpression *) = 0;
  virtual void visit(ReturnStatement *) = 0;
  virtual void visit(BreakStatement *) = 0;
  virtual void visit(ContinueStatement *) = 0;
  virtual void visit(DiscardStatement *) = 0;
  virtual void visit(FunctionDeclaration *) = 0;
  virtual void visit(DefaultLabel *) = 0;
  virtual void visit(CaseLabel *) = 0;
};

}; // namespace ast

}; // namespace shaderpulse
