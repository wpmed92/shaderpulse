#pragma once

namespace shaderpulse {

namespace ast {

class TranslationUnit;
class BinaryExpression;
class UnaryExpression;
class ValueDeclaration;
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
  virtual void visit(ValueDeclaration *) = 0;
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

class PrinterASTVisitor : public ASTVisitor {

public:
  void visit(TranslationUnit *) override;
  void visit(BinaryExpression *) override;
  void visit(UnaryExpression *) override;
  void visit(ValueDeclaration *) override;
  void visit(SwitchStatement *) override;
  void visit(WhileStatement *) override;
  void visit(DoStatement *) override;
  void visit(IfStatement *) override;
  void visit(AssignmentExpression *) override;
  void visit(StatementList *) override;
  void visit(CallExpression *) override;
  void visit(VariableExpression *) override;
  void visit(IntegerConstantExpression *) override;
  void visit(UnsignedIntegerConstantExpression *) override;
  void visit(FloatConstantExpression *) override;
  void visit(DoubleConstantExpression *) override;
  void visit(BoolConstantExpression *) override;
  void visit(ReturnStatement *) override;
  void visit(BreakStatement *) override;
  void visit(ContinueStatement *) override;
  void visit(DiscardStatement *) override;
  void visit(FunctionDeclaration *) override;
  void visit(DefaultLabel *) override;
  void visit(CaseLabel *) override;
};

}; // namespace ast

}; // namespace shaderpulse
