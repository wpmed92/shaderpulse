#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include "AST/Types.h"

namespace shaderpulse {

namespace ast {

enum UnaryOperator {
    Inc,
    Dec,
    Plus,
    Dash,
    Bang,
    Tilde
};

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

class Expression {
public:
    virtual ~Expression() = default;
};

class IntegerConstantExpression : public Expression {

public:
    IntegerConstantExpression(int32_t val) : val(val) {

    }

    int32_t getVal() { return val; }
private:
    int32_t val;
};

class UnsignedIntegerConstantExpression : public Expression {

public:
    UnsignedIntegerConstantExpression(uint32_t val) : val(val) {

    }

    uint32_t getVal() { return val; }
private:
    uint32_t val;
};

class FloatConstantExpression : public Expression {
public:
    FloatConstantExpression(float val) : val(val) {

    }

    float getVal() const { return val; }
private:
    float val;
};

class DoubleConstantExpression : public Expression {
public:
    DoubleConstantExpression(double val) : val(val) {

    }

    double getVal() const { return val; }

private:
    double val;
};

class BoolConstantExpression : public Expression {
public:
    BoolConstantExpression(bool val) : val(val) {

    }

    bool getVal() const { return val; }
private:
    bool val;
};

class VariableExpression : public Expression {
public:
    VariableExpression(const std::string &name) : name(name) {

    }

    std::string getName() const { return name; }
private:
    std::string name;
};

class CallExpression : public Expression {

public:
    CallExpression(const std::string& functionName) :
        functionName(functionName) {

    }

    CallExpression(const std::string& functionName, 
    std::vector<std::unique_ptr<Expression>> arguments) :
        functionName(functionName),
        arguments(std::move(arguments)) {

    }

    const std::string& getFunctionName() const { return functionName; }
    const std::vector<std::unique_ptr<Expression>>& getArguments() const { return arguments; }

private:
    std::string functionName;
    std::vector<std::unique_ptr<Expression>> arguments;

};

class UnaryExpression : public Expression {

public:
    UnaryExpression(UnaryOperator op, std::unique_ptr<Expression> rhs)
        : op(op), rhs(std::move(rhs)) {

    }

    UnaryOperator getOp() const { return op; }
    Expression* getExpression() const { return rhs.get(); }
private:
    UnaryOperator op;
    std::unique_ptr<Expression> rhs;

};

class BinaryExpression : public Expression {

public:
    BinaryExpression(BinaryOperator op, std::unique_ptr<Expression> lhs, std::unique_ptr<Expression> rhs)
        : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {

        }

    Expression* getRhs() { return rhs.get(); }
    Expression* getLhs() { return lhs.get(); }
    BinaryOperator getOp() { return op; }
private:
    BinaryOperator op;
    std::unique_ptr<Expression> rhs;
    std::unique_ptr<Expression> lhs;

};

class Statement {

public:
    virtual ~Statement() = default;
};


class ExternalDeclaration {

public:
    virtual ~ExternalDeclaration() = default;
};


class ValueDeclaration : public ExternalDeclaration, public Statement {
    
public:
     ValueDeclaration(std::unique_ptr<Type> type, std::vector<std::string> names)
        : type(std::move(type)), names(std::move(names))  {

        }

    const std::vector<std::string>& getIdentifierNames() const { return names; }
    Type* getType() const { return type.get(); }
private:
    std::vector<std::string> names;
    std::unique_ptr<Type> type;

};

class ParameterDeclaration {

public:
    ParameterDeclaration(const std::string& name) : name(name) {

    }

    const std::string& getName() { return name; }

private:
    std::string name;
};  

class ForStatement : public Statement {

};

class SwitchStatement : public Statement {

public:
    SwitchStatement(std::unique_ptr<Expression> expression, std::unique_ptr<Statement> statements) :
        expression(std::move(expression)), statements(std::move(statements)) {

        }

    Expression* getExpression() { return expression.get(); }
    Statement* getBody() { return statements.get(); }
private:
    std::unique_ptr<Expression> expression;
    std::unique_ptr<Statement> statements;
};

class CaseLabel : public Statement {

public:
    CaseLabel(std::unique_ptr<Expression> expression) :
        expression(std::move(expression)) {

        }

private:
    std::unique_ptr<Expression> expression;
};

class DefaultLabel : public Statement {

};

class StatementList : public Statement {

public:
    StatementList(std::vector<std::unique_ptr<Statement>> statements) :
        statements(std::move(statements)) {

        }

    const std::vector<std::unique_ptr<Statement>>& getStatements() { return statements; }

private:
    std::vector<std::unique_ptr<Statement>>  statements;
};


class WhileStatement : public Statement {

public:
    WhileStatement(
        std::unique_ptr<Expression> condition,
        std::unique_ptr<Statement> body) : 
            condition(std::move(condition)), 
            body(std::move(body)) {

        }

        Expression* getCondition() { return condition.get(); }
        Statement* getBody() { return body.get(); }
private:
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Statement> body;
};

class DoStatement : public Statement {

public:
    DoStatement(std::unique_ptr<Statement> body, 
        std::unique_ptr<Expression> condition) : 
            body(std::move(body)),
            condition(std::move(condition)) {

        }

        Expression* getCondition() { return condition.get(); }
        Statement* getBody() { return body.get(); }

private:
    std::unique_ptr<Statement> body;
    std::unique_ptr<Expression> condition;
};

class IfStatement : public Statement {

public:
    IfStatement(
        std::unique_ptr<Expression> condition,
        std::unique_ptr<Statement> truePart, 
        std::unique_ptr<Statement> falsePart) : 
            condition(std::move(condition)),
            truePart(std::move(truePart)),
            falsePart(std::move(falsePart)) {

        }

    Expression* getCondition() { return condition.get(); }
    Statement* getTruePart() { return truePart.get(); }
    Statement* getFalsePart() { return falsePart.get(); }
private:
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Statement> truePart;
    std::unique_ptr<Statement> falsePart;
};

class ReturnStatement : public Statement {

public:
    ReturnStatement() = default;
    ReturnStatement(std::unique_ptr<Expression> exp) : 
    exp(std::move(exp)) {

    }

    Expression* getExpression() { return exp.get(); }
private:
    std::unique_ptr<Expression> exp;
};

class BreakStatement : public Statement {

};

class ContinueStatement : public Statement {

};

// Fragment shader only
class DiscardStatement : public Statement {

};

class AssignmentExpression : public Statement {

public:
    AssignmentExpression(const std::string& identifier, AssignmentOperator op, std::unique_ptr<Expression> expression) :
        identifier(identifier), op(op), expression(std::move(expression)) {

        }

        const std::string& getIdentifier() { return identifier; }
        AssignmentOperator getOperator() { return op; }
        Expression* getExpression() { return expression.get(); }

private:
    std::string identifier;
    AssignmentOperator op;
    std::unique_ptr<Expression> expression;
};

class FunctionDeclaration : public ExternalDeclaration {

public :
    FunctionDeclaration(
        std::unique_ptr<Type> returnType,
        const std::string &name,
        std::vector<std::unique_ptr<ParameterDeclaration>> params, 
        std::unique_ptr<Statement>  body
    ) : returnType(std::move(returnType)), name(name), params(std::move(params)), body(std::move(body)) { }

    const std::vector<std::unique_ptr<ParameterDeclaration>>& getParams() { return params; }
    Statement* getBody() { return body.get(); }
    Type* getReturnType() { return returnType.get(); }
    const std::string& getName() { return name; }

private:
    std::unique_ptr<Type> returnType;
    std::string name;
    std::vector<std::unique_ptr<ParameterDeclaration>> params;
    std::unique_ptr<Statement>  body;
};

class TranslationUnit {
public:
    TranslationUnit(std::vector<std::unique_ptr<ExternalDeclaration>> externalDeclarations) :
        externalDeclarations(std::move(externalDeclarations)) { }

    const std::vector<std::unique_ptr<ExternalDeclaration>>& getExternalDeclarations() { 
        return externalDeclarations; 
    }

private:
    std::vector<std::unique_ptr<ExternalDeclaration>> externalDeclarations;
};

}; // namespace ast

}; // namespace shaderpulse