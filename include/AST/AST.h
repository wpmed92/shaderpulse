#pragma once
#include "AST/Types.h"
#include "AST/ASTVisitor.h"
#include <stdint.h>
#include <string>
#include <vector>

namespace shaderpulse {

namespace ast {

enum UnaryOperator { Inc, Dec, Plus, Dash, Bang, Tilde };

enum BinaryOperator {
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  ShiftLeft,
  ShiftRight,
  Lt,
  Gt,
  LtEq,
  GtEq,
  Eq,
  Neq,
  BitAnd,
  BitXor,
  BitIor,
  LogAnd,
  LogXor,
  LogOr
};

enum AssignmentOperator {
  Equal,
  MulAssign,
  DivAssign,
  ModAssign,
  AddAssign,
  SubAssign,
  LeftAssign,
  RightAssign,
  AndAssign,
  XorAssign,
  OrAssign
};

class ASTNode {
public:
  virtual ~ASTNode() = default;
  virtual void accept(ASTVisitor *visitor) = 0;
};

class Expression : public ASTNode {
public:
  virtual ~Expression() = default;
};

class IntegerConstantExpression : public Expression {

public:
  IntegerConstantExpression(int32_t val) : val(val) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  int32_t getVal() { return val; }

private:
  int32_t val;
};

class UnsignedIntegerConstantExpression : public Expression {

public:
  UnsignedIntegerConstantExpression(uint32_t val) : val(val) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); } 

  uint32_t getVal() { return val; }

private:
  uint32_t val;
};

class FloatConstantExpression : public Expression {
public:
  FloatConstantExpression(float val) : val(val) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  float getVal() const { return val; }

private:
  float val;
};

class DoubleConstantExpression : public Expression {
public:
  DoubleConstantExpression(double val) : val(val) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  double getVal() const { return val; }

private:
  double val;
};

class BoolConstantExpression : public Expression {
public:
  BoolConstantExpression(bool val) : val(val) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  bool getVal() const { return val; }

private:
  bool val;
};

class VariableExpression : public Expression {
public:
  VariableExpression(const std::string &name) : name(name) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  std::string getName() const { return name; }

private:
  std::string name;
};

class CallExpression : public Expression {

public:
  CallExpression(const std::string &functionName)
      : functionName(functionName) {}

  CallExpression(const std::string &functionName,
                 std::vector<std::unique_ptr<Expression>> arguments)
      : functionName(functionName), arguments(std::move(arguments)) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  const std::string &getFunctionName() const { return functionName; }
  const std::vector<std::unique_ptr<Expression>> &getArguments() const {
    return arguments;
  }

private:
  std::string functionName;
  std::vector<std::unique_ptr<Expression>> arguments;
};

class ConstructorExpression : public Expression {

public:
    ConstructorExpression(std::unique_ptr<Type> type) : 
      type(std::move(type))  {

    }

  ConstructorExpression(std::unique_ptr<Type> type, 
    std::vector<std::unique_ptr<Expression>> arguments) : 
      type(std::move(type)), 
      arguments(std::move(arguments))  {

    }

  void accept(ASTVisitor *visitor) override {
    visitor->visit(this);
  }

  Type *getType() const { return type.get(); }
  const std::vector<std::unique_ptr<Expression>> &getArguments() const {
    return arguments;
  }

private:
  std::unique_ptr<Type> type;
  std::vector<std::unique_ptr<Expression>> arguments;
};

class UnaryExpression : public Expression {

public:
  UnaryExpression(UnaryOperator op, std::unique_ptr<Expression> rhs)
      : op(op), rhs(std::move(rhs)) {}

  void accept(ASTVisitor *visitor) override {
    visitor->visit(this);
  }

  UnaryOperator getOp() const { return op; }
  Expression *getExpression() const { return rhs.get(); }

private:
  UnaryOperator op;
  std::unique_ptr<Expression> rhs;
};

class BinaryExpression : public Expression {

public:
  BinaryExpression(BinaryOperator op, std::unique_ptr<Expression> lhs,
                   std::unique_ptr<Expression> rhs)
      : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  Expression *getLhs() { return lhs.get(); }
  Expression *getRhs() { return rhs.get(); }
  BinaryOperator getOp() { return op; }

private:
  BinaryOperator op;
  std::unique_ptr<Expression> lhs;
  std::unique_ptr<Expression> rhs;
};

class Statement : virtual public ASTNode {

public:
  virtual ~Statement() = default;
};

class ExternalDeclaration : virtual public ASTNode {

public:
  virtual ~ExternalDeclaration() = default;
};

class Declaration : public ExternalDeclaration, public Statement {

public:
  virtual ~Declaration() = default;
};

class VariableDeclaration : public Declaration {

public:
  VariableDeclaration(std::unique_ptr<Type> type, const std::string &name,
                      std::unique_ptr<Expression> initializerExpr)
      : type(std::move(type)), identifierName(name),
        initializerExpr(std::move(initializerExpr)) {}

  const std::string &getIdentifierName() const { return identifierName; }
  Type *getType() const { return type.get(); }
  Expression *getInitialzerExpression() { return initializerExpr.get(); }

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

private:
  std::unique_ptr<Type> type;
  std::string identifierName;
  std::unique_ptr<Expression> initializerExpr;
};

class VariableDeclarationList : public Declaration {

public:
  VariableDeclarationList(
      std::unique_ptr<Type> type,
      std::vector<std::unique_ptr<VariableDeclaration>> declarations)
      : type(std::move(type)), declarations(std::move(declarations)) {}

  const std::vector<std::unique_ptr<VariableDeclaration>> &
  getDeclarations() const {
    return declarations;
  }
  Type *getType() const { return type.get(); }
  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

private:
  std::unique_ptr<Type> type;
  std::vector<std::unique_ptr<VariableDeclaration>> declarations;
};

class ParameterDeclaration {

public:
  ParameterDeclaration(const std::string &name, std::unique_ptr<Type> type)
      : name(name), type(std::move(type)) {}

  const std::string &getName() { return name; }
  Type *getType() { return type.get(); }

private:
  std::string name;
  std::unique_ptr<Type> type;
};

class ForStatement : public Statement {};

class SwitchStatement : public Statement {

public:
  SwitchStatement(std::unique_ptr<Expression> expression,
                  std::unique_ptr<Statement> statements)
      : expression(std::move(expression)), statements(std::move(statements)) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  Expression *getExpression() { return expression.get(); }
  Statement *getBody() { return statements.get(); }

private:
  std::unique_ptr<Expression> expression;
  std::unique_ptr<Statement> statements;
};

class CaseLabel : public Statement {

public:
  CaseLabel(std::unique_ptr<Expression> expression)
      : expression(std::move(expression)) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }
  Expression *getExpression() { return expression.get(); }

private:
  std::unique_ptr<Expression> expression;
};

class DefaultLabel : public Statement {
  void accept(ASTVisitor *visitor) override { visitor->visit(this); }
};

class StatementList : public Statement {

public:
  StatementList(std::vector<std::unique_ptr<Statement>> statements)
      : statements(std::move(statements)) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  const std::vector<std::unique_ptr<Statement>> &getStatements() {
    return statements;
  }

private:
  std::vector<std::unique_ptr<Statement>> statements;
};

class WhileStatement : public Statement {

public:
  WhileStatement(std::unique_ptr<Expression> condition,
                 std::unique_ptr<Statement> body)
      : condition(std::move(condition)), body(std::move(body)) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  Expression *getCondition() { return condition.get(); }
  Statement *getBody() { return body.get(); }

private:
  std::unique_ptr<Expression> condition;
  std::unique_ptr<Statement> body;
};

class DoStatement : public Statement {

public:
  DoStatement(std::unique_ptr<Statement> body,
              std::unique_ptr<Expression> condition)
      : body(std::move(body)), condition(std::move(condition)) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  Expression *getCondition() { return condition.get(); }
  Statement *getBody() { return body.get(); }

private:
  std::unique_ptr<Statement> body;
  std::unique_ptr<Expression> condition;
};

class IfStatement : public Statement {

public:
  IfStatement(std::unique_ptr<Expression> condition,
              std::unique_ptr<Statement> truePart,
              std::unique_ptr<Statement> falsePart)
      : condition(std::move(condition)), truePart(std::move(truePart)),
        falsePart(std::move(falsePart)) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  Expression *getCondition() { return condition.get(); }
  Statement *getTruePart() { return truePart.get(); }
  Statement *getFalsePart() { return falsePart.get(); }

private:
  std::unique_ptr<Expression> condition;
  std::unique_ptr<Statement> truePart;
  std::unique_ptr<Statement> falsePart;
};

class ReturnStatement : public Statement {

public:
  ReturnStatement() = default;
  ReturnStatement(std::unique_ptr<Expression> exp) : exp(std::move(exp)) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  Expression *getExpression() { return exp.get(); }

private:
  std::unique_ptr<Expression> exp;
};

class BreakStatement : public Statement {
  void accept(ASTVisitor *visitor) override { visitor->visit(this); }
};

class ContinueStatement : public Statement {
  void accept(ASTVisitor *visitor) override { visitor->visit(this); }
};

// Fragment shader only
class DiscardStatement : public Statement {
  void accept(ASTVisitor *visitor) override { visitor->visit(this); }
};

class AssignmentExpression : public Statement {

public:
  AssignmentExpression(const std::string &identifier, AssignmentOperator op,
                       std::unique_ptr<Expression> expression)
      : identifier(identifier), op(op), expression(std::move(expression)) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  const std::string &getIdentifier() { return identifier; }
  AssignmentOperator getOperator() { return op; }
  Expression *getExpression() { return expression.get(); }

private:
  std::string identifier;
  AssignmentOperator op;
  std::unique_ptr<Expression> expression;
};

class FunctionDeclaration : public ExternalDeclaration {

public:
  FunctionDeclaration(std::unique_ptr<Type> returnType, const std::string &name,
                      std::vector<std::unique_ptr<ParameterDeclaration>> params,
                      std::unique_ptr<Statement> body)
      : returnType(std::move(returnType)), name(name),
        params(std::move(params)), body(std::move(body)) {}

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

  const std::vector<std::unique_ptr<ParameterDeclaration>> &getParams() {
    return params;
  }
  Statement *getBody() { return body.get(); }
  Type *getReturnType() { return returnType.get(); }
  const std::string &getName() { return name; }

private:
  std::unique_ptr<Type> returnType;
  std::string name;
  std::vector<std::unique_ptr<ParameterDeclaration>> params;
  std::unique_ptr<Statement> body;
};

class TranslationUnit : public ASTNode {
public:
  TranslationUnit(
      std::vector<std::unique_ptr<ExternalDeclaration>> externalDeclarations)
      : externalDeclarations(std::move(externalDeclarations)) {}

  const std::vector<std::unique_ptr<ExternalDeclaration>> &
  getExternalDeclarations() {
    return externalDeclarations;
  }

  void accept(ASTVisitor *visitor) override { visitor->visit(this); }

private:
  std::vector<std::unique_ptr<ExternalDeclaration>> externalDeclarations;
};

}; // namespace ast

class LayoutQualifierId {

public:
LayoutQualifierId(bool isShared) : 
    isShared(isShared) {

  }

  LayoutQualifierId(const std::string &identifier) : 
    identifier(identifier) {

  }

  LayoutQualifierId(const std::string &identifier, std::unique_ptr<ast::Expression> expression) : 
    identifier(identifier), exp(std::move(expression)) {

  }

  const std::string &getIdentifier() { return identifier; }
  ast::Expression *getExpression() { return exp.get(); }
  bool getIsShader() const { return isShared; }

private:
  bool isShared;
  std::string identifier;
  std::unique_ptr<ast::Expression> exp;
};

class LayoutQualifier : public TypeQualifier {

public:
  LayoutQualifier(std::vector<std::unique_ptr<LayoutQualifierId>> layoutQualifierIds) :
     TypeQualifier(TypeQualifierKind::Layout),
     layoutQualifierIds(std::move(layoutQualifierIds)) {

  }

  const std::vector<std::unique_ptr<LayoutQualifierId>> &getLayoutQualifierIds() { return layoutQualifierIds; }

  LayoutQualifierId *getQualifierId(const std::string &id) {
    if (id.empty()) {
      return nullptr;
    }

    auto it =
        std::find_if(layoutQualifierIds.begin(), layoutQualifierIds.end(),
                     [&id](const std::unique_ptr<LayoutQualifierId> &qualifier) {
                       return qualifier->getIdentifier() == id;
                     });

    if (it != layoutQualifierIds.end()) {
      return it->get();
    } else {
      return nullptr;
    }
  }

private:
  std::vector<std::unique_ptr<LayoutQualifierId>> layoutQualifierIds;
};


}; // namespace shaderpulse
