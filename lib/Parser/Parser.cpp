#include "Parser/Parser.h"
#include <iostream>

namespace shaderpulse {

using namespace ast;

namespace parser {

std::map<BinaryOperator, int> Parser::binopPrecedence = {
    { BinaryOperator::LogOr,       10 },
    { BinaryOperator::LogXor,      20 },
    { BinaryOperator::LogAnd,      30 },
    { BinaryOperator::BitIor,      40 },
    { BinaryOperator::BitXor,      50 },
    { BinaryOperator::BitAnd,      60 },
    { BinaryOperator::Eq,          70 },
    { BinaryOperator::Neq,         70 },
    { BinaryOperator::Gt,          80 },
    { BinaryOperator::Lt,          80 },
    { BinaryOperator::GtEq,        80 },
    { BinaryOperator::LtEq,        80 },
    { BinaryOperator::ShiftLeft,   90 },
    { BinaryOperator::ShiftRight,  90 },
    { BinaryOperator::Add,        100 },
    { BinaryOperator::Sub,        100 },
    { BinaryOperator::Mul,        110 },
    { BinaryOperator::Div,        110 },
    { BinaryOperator::Mod,        110 },
};

std::unique_ptr<TranslationUnit> Parser::parseTranslationUnit() {
    std::vector<std::unique_ptr<ExternalDeclaration>> externalDeclarations;

    while ((size_t) cursor < tokenStream.size() - 1) {
        // External declaration
        if (auto externalDecl = parseExternalDeclaration()) {
            externalDeclarations.push_back(std::move(externalDecl));
        } else {
            break;
        }
    }

    return std::make_unique<TranslationUnit>(std::move(externalDeclarations));
}

std::unique_ptr<ExternalDeclaration> Parser::parseExternalDeclaration() {
    if (auto funcDecl = parseFunctionDeclaration()) {
        return funcDecl;
    } else if (auto decl = parseDeclaration()) {
        return decl;
    } else {
        return nullptr;
    }
}

std::unique_ptr<FunctionDeclaration> Parser::parseFunctionDeclaration() {
    if (auto type = Parser::getTypeFromTokenKind(curToken->getTokenKind())) {
        if (!(peek(1)->is(TokenKind::Identifier) && peek(2)->is(TokenKind::lParen))) {
            return nullptr;
        }

        advanceToken();

        auto returnType = std::move(*type);
        const std::string& functionName = curToken->getIdentifierName();

        advanceToken(); // eat lparen

        auto params = parseFunctionParameters();

        if (!curToken->is(TokenKind::rParen)) {
            std::cout << "Expected a ')' after function parameter declaration." << std::endl;
            return nullptr;
        }

        advanceToken();

        if (auto body = parseStatement()) {
            return std::make_unique<FunctionDeclaration>(std::move(returnType), functionName, std::move(params), std::move(body));
        } else {
            return nullptr;
        }
    } else {
        return nullptr;
    }
}

std::unique_ptr<ValueDeclaration> Parser::parseDeclaration() {
    if (auto type = Parser::getTypeFromTokenKind(curToken->getTokenKind())) {
        advanceToken();

        if (curToken->is(TokenKind::semiColon)) {
            return std::make_unique<ValueDeclaration>(std::move(*type), std::vector<std::string>());
        }

        std::vector<std::string> names;

        names.push_back(curToken->getIdentifierName());

        advanceToken();

        if (curToken->is(TokenKind::semiColon)) {
            advanceToken();
            return std::make_unique<ValueDeclaration>(std::move(*type), std::move(names));
        } else if (curToken->is(TokenKind::comma)) {
            while (curToken->is(TokenKind::comma)) {
                advanceToken();
                names.push_back(curToken->getIdentifierName());
                advanceToken();
            }

            advanceToken();
            return std::make_unique<ValueDeclaration>(std::move(*type), std::move(names));
        } else {
            return nullptr;
        }
    }

    return nullptr;
}


std::unique_ptr<ast::Expression> Parser::parseExpression() {
    auto lhs = parseUnaryExpression();

    advanceToken();

    if (!lhs) {
        return nullptr;
    }

    return parseRhs(0, std::move(lhs));
}

std::unique_ptr<ast::Expression> Parser::parseRhs(int exprPrec, std::unique_ptr<ast::Expression> lhs) {
    while (true) {
        auto binop = getBinaryOperatorFromTokenKind(curToken->getTokenKind());
        int tokPrec = -1;

        if (binop.has_value()) {
            auto precIt = binopPrecedence.find(*binop);

            if (precIt != binopPrecedence.end()) {
                tokPrec = precIt->second;
            }
        }

        if (tokPrec < exprPrec)
            return lhs;

        advanceToken();

        auto rhs = parsePrimaryExpression();

        if (!rhs)
            return nullptr;

        advanceToken();

        // If BinOp binds less tightly with RHS than the operator after RHS, let
        // the pending operator take RHS as its LHS.
        int nextPrec = -1;
        auto nextBinop = getBinaryOperatorFromTokenKind(curToken->getTokenKind());

        if (nextBinop.has_value()) {
            auto precIt = binopPrecedence.find(*nextBinop);

            if (precIt != binopPrecedence.end()) {
                nextPrec = precIt->second;
            }
        }

        if (tokPrec < nextPrec) {
            rhs = parseRhs(tokPrec + 1, std::move(rhs));

            if (!rhs)
                return nullptr;
        }
    
        // Merge LHS/RHS.
        lhs = std::make_unique<BinaryExpression>(*binop, std::move(lhs), std::move(rhs));
    }
}

std::optional<BinaryOperator> Parser::getBinaryOperatorFromTokenKind(TokenKind kind) {
    switch (kind) {
        case TokenKind::plus:
            return BinaryOperator::Add;
        case TokenKind::minus:
            return BinaryOperator::Sub;
        case TokenKind::mul:
            return BinaryOperator::Mul;
        case TokenKind::div:
            return BinaryOperator::Div;
        case TokenKind::shiftLeft:
            return BinaryOperator::ShiftLeft;
        case TokenKind::shiftRight:
            return BinaryOperator::ShiftRight;
        case TokenKind::modulo:
            return BinaryOperator::Mod;
        case TokenKind::eq:
            return BinaryOperator::Eq;
        case TokenKind::neq:
            return BinaryOperator::Neq;
        case TokenKind::gt:
            return BinaryOperator::Gt;
        case TokenKind::lt:
            return BinaryOperator::Lt;
        case TokenKind::gtEq:
            return BinaryOperator::GtEq;
        case TokenKind::ltEq:
            return BinaryOperator::LtEq;
        case TokenKind::bor:
            return BinaryOperator::BitIor;
        case TokenKind::bxor:
            return BinaryOperator::BitXor;
        case TokenKind::band:
            return BinaryOperator::BitAnd;
        case TokenKind::land:
            return BinaryOperator::LogAnd;
        case TokenKind::lor:
            return BinaryOperator::LogOr;
        case TokenKind::lxor:
            return BinaryOperator::LogXor;
        default:
            return std::nullopt;
    }
}

std::optional<ast::UnaryOperator> Parser::getUnaryOperatorFromTokenKind(TokenKind kind) {
    switch (kind) {
        case TokenKind::plus:
            return UnaryOperator::Plus;
        case TokenKind::minus:
            return UnaryOperator::Dash;
        case TokenKind::lnot:
            return UnaryOperator::Bang;
        case TokenKind::bnot:
            return UnaryOperator::Tilde;
        default:
            return std::nullopt;
    }
}

std::optional<ast::AssignmentOperator> Parser::getAssignmentOperatorFromTokenKind(TokenKind kind) {
    switch (kind) {
        case TokenKind::eq:
            return AssignmentOperator::Equal;
        case TokenKind::mulAssign:
            return AssignmentOperator::MulAssign;
        case TokenKind::divAssign:
            return AssignmentOperator::DivAssign;
        case TokenKind::modAssign:
            return AssignmentOperator::ModAssign;
        case TokenKind::addAssign:
            return AssignmentOperator::AddAssign;
        case TokenKind::subAssign:
            return AssignmentOperator::SubAssign;
        case TokenKind::shiftLeftAssign:
            return AssignmentOperator::LeftAssign;
        case TokenKind::shiftRightAssign:
            return AssignmentOperator::RightAssign;
        case TokenKind::landAssign:
            return AssignmentOperator::AndAssign;
        case TokenKind::lxorAssign:
            return AssignmentOperator::XorAssign;
        case TokenKind::lorAssign:
            return AssignmentOperator::OrAssign;
        default:
            return std::nullopt;
    }
}

std::optional<std::unique_ptr<shaderpulse::Type>> Parser::getTypeFromTokenKind(TokenKind kind) {
    switch (kind) {
        // Sclar types
        case TokenKind::kw_int:
            return std::make_unique<shaderpulse::Type>(TypeKind::Integer);
        case TokenKind::kw_bool:
            return std::make_unique<shaderpulse::Type>(TypeKind::Bool);
        case TokenKind::kw_uint:
            return std::make_unique<shaderpulse::Type>(TypeKind::UnsignedInteger);
        case TokenKind::kw_float:
            return std::make_unique<shaderpulse::Type>(TypeKind::Float);
        case TokenKind::kw_double:
            return std::make_unique<shaderpulse::Type>(TypeKind::Double);
        case TokenKind::kw_void:
            return std::make_unique<shaderpulse::Type>(TypeKind::Void);

        // Vector types
        case TokenKind::kw_vec2:
            return makeVectorType(TypeKind::Float, 2);
        case TokenKind::kw_vec3:
            return makeVectorType(TypeKind::Float, 3);
        case TokenKind::kw_vec4:
            return makeVectorType(TypeKind::Float, 4);

        case TokenKind::kw_bvec2:
            return makeVectorType(TypeKind::Bool, 2);
        case TokenKind::kw_bvec3:
            return makeVectorType(TypeKind::Bool, 3);
        case TokenKind::kw_bvec4:
            return makeVectorType(TypeKind::Bool, 4);

        case TokenKind::kw_ivec2:
            return makeVectorType(TypeKind::Integer, 2);
        case TokenKind::kw_ivec3:
            return makeVectorType(TypeKind::Integer, 3);
        case TokenKind::kw_ivec4:
            return makeVectorType(TypeKind::Integer, 4);

        case TokenKind::kw_uvec2:
            return makeVectorType(TypeKind::UnsignedInteger, 2);
        case TokenKind::kw_uvec3:
            return makeVectorType(TypeKind::UnsignedInteger, 3);
        case TokenKind::kw_uvec4:
            return makeVectorType(TypeKind::UnsignedInteger, 4);
        
        case TokenKind::kw_dvec2:
            return makeVectorType(TypeKind::Double, 2);
        case TokenKind::kw_dvec3:
            return makeVectorType(TypeKind::Double, 3);
        case TokenKind::kw_dvec4:
            return makeVectorType(TypeKind::Double, 4);

        // Matrix types
        case TokenKind::kw_mat2:
        case TokenKind::kw_mat2x2:
            return makeMatrixType(TypeKind::Float, 2, 2);

        case TokenKind::kw_mat3:
        case TokenKind::kw_mat3x3:
            return makeMatrixType(TypeKind::Float, 3, 3);

        case TokenKind::kw_mat4:
        case TokenKind::kw_mat4x4:
            return makeMatrixType(TypeKind::Float, 4, 4);

        case TokenKind::kw_dmat2:
        case TokenKind::kw_dmat2x2:
            return makeMatrixType(TypeKind::Double, 2, 2);

        case TokenKind::kw_dmat3:
        case TokenKind::kw_dmat3x3:
            return makeMatrixType(TypeKind::Double, 3, 3);

        case TokenKind::kw_dmat4:
        case TokenKind::kw_dmat4x4:
            return makeMatrixType(TypeKind::Double, 4, 4);

        case TokenKind::kw_mat2x3:
            return makeMatrixType(TypeKind::Float, 2, 3);
        
        case TokenKind::kw_mat2x4:
            return makeMatrixType(TypeKind::Float, 2, 4);

        case TokenKind::kw_mat3x2:
            return makeMatrixType(TypeKind::Float, 3, 2);

        case TokenKind::kw_mat3x4:
            return makeMatrixType(TypeKind::Float, 3, 4);
        
        case TokenKind::kw_mat4x2:
            return makeMatrixType(TypeKind::Float, 4, 2);

        case TokenKind::kw_mat4x3:
            return makeMatrixType(TypeKind::Float, 4, 3);

        case TokenKind::kw_dmat2x3:
            return makeMatrixType(TypeKind::Double, 2, 3);
        
        case TokenKind::kw_dmat2x4:
            return makeMatrixType(TypeKind::Double, 2, 4);

        case TokenKind::kw_dmat3x2:
            return makeMatrixType(TypeKind::Double, 3, 2);

        case TokenKind::kw_dmat3x4:
            return makeMatrixType(TypeKind::Double, 3, 4);
        
        case TokenKind::kw_dmat4x2:
            return makeMatrixType(TypeKind::Double, 4, 2);

        case TokenKind::kw_dmat4x3:
            return makeMatrixType(TypeKind::Double, 4, 3);

        default:
            // Ignore
            break;
    }

    if (kind >= TokenKind::kw_sampler1D && kind <= TokenKind::kw_uimageBuffer) {
        return std::make_unique<shaderpulse::Type>(TypeKind::Opaque);
    }

    return std::nullopt;
};

std::unique_ptr<shaderpulse::VectorType> Parser::makeVectorType(TypeKind kind, int length) {
    return std::make_unique<shaderpulse::VectorType>(
        std::make_unique<shaderpulse::Type>(kind), 
        length
    );
}

std::unique_ptr<shaderpulse::MatrixType> Parser::makeMatrixType(TypeKind kind, int rows, int cols) {
    return std::make_unique<shaderpulse::MatrixType>(
        std::make_unique<shaderpulse::Type>(kind), 
        rows, 
        cols
    );
}

std::vector<std::unique_ptr<ParameterDeclaration>> Parser::parseFunctionParameters() {
    std::vector<std::unique_ptr<ParameterDeclaration>> params;

    do {
        advanceToken();

        if (auto type = Parser::getTypeFromTokenKind(curToken->getTokenKind())) {
            advanceToken();

            if (curToken->is(TokenKind::Identifier)) {
                auto param = std::make_unique<ParameterDeclaration>(
                    curToken->getIdentifierName(),
                    std::move(*type)
                );
                advanceToken();
                params.push_back(std::move(param));
            }
        } else {
            // Error: expected type name
            return {};
        }

    } while(curToken->is(TokenKind::comma));

    return params;
}

std::unique_ptr<ForStatement> Parser::parseForLoop() {
    advanceToken();

    if (!curToken->is(TokenKind::kw_for)) {
        return nullptr;
    }

    advanceToken();

    if (!curToken->is(TokenKind::lParen)) {
        return nullptr; // Expected lparen
    }

    advanceToken();

    return nullptr;
}

std::unique_ptr<SwitchStatement> Parser::parseSwitchStatement() {
    if (!curToken->is(TokenKind::kw_switch)) {
        return nullptr;
    }

    advanceToken();

    if (!curToken->is(TokenKind::lParen)) {
        std::cout << "Expected a '(' after switch keyword." << std::endl;
        return nullptr;
    }

    advanceToken();

    auto exp = parseExpression();

    if (!exp) {
        return nullptr;
    }

    if (!curToken->is(TokenKind::rParen)) {
        std::cout << "Expected a ')' after condition." << std::endl;
        return nullptr;
    }
    
    advanceToken();

    auto stmt = parseStatement();

    if (!stmt) {
        return nullptr;
    }

    return std::make_unique<SwitchStatement>(std::move(exp), std::move(stmt));
}

std::unique_ptr<WhileStatement> Parser::parseWhileStatement() {
    if (!curToken->is(TokenKind::kw_while)) {
        return nullptr;
    }

    advanceToken();

    if (!curToken->is(TokenKind::lParen)) {
        std::cout << "Expected a '(' after while keyword." << std::endl;
        return nullptr;
    }

    advanceToken();

    auto exp = parseExpression();

    if (!curToken->is(TokenKind::rParen)) {
        std::cout << "Expected a ')' after condition expression." << cursor << std::endl;
        return nullptr;
    }

    advanceToken();

    if (auto stmt = parseStatement()) {
        return std::make_unique<WhileStatement>(std::move(exp), std::move(stmt));
    } else {
        return nullptr;
    }
}

std::unique_ptr<DoStatement> Parser::parseDoStatement() {
    if (!curToken->is(TokenKind::kw_do)) {
        return nullptr;
    }

    advanceToken();

    if (auto stmt = parseStatement()) {
        if (!curToken->is(TokenKind::kw_while)) {
            std::cout << "Expected while keyword after statement list." << std::endl;
            return nullptr;
        }

        advanceToken();

        if (!curToken->is(TokenKind::lParen)) {
            std::cout << "Expected a '(' after while keyword." << std::endl;
            return nullptr;
        }

        advanceToken();

        auto exp = parseExpression();

        if (!curToken->is(TokenKind::rParen)) {
            std::cout << "Expected a ')' after condition expression." << cursor << std::endl;
            return nullptr;
        }

        advanceToken();

        if (!curToken->is(TokenKind::semiColon)) {
            std::cout << "Expected a semicolon after while statement." << std::endl;
            return nullptr;
        }

        advanceToken();

        return std::make_unique<DoStatement>(std::move(stmt), std::move(exp));
    } else {
        return nullptr;
    }
}

std::unique_ptr<IfStatement> Parser::parseIfStatement() {
    if (!curToken->is(TokenKind::kw_if)) {
        return nullptr;
    }

    advanceToken();

    if (!curToken->is(TokenKind::lParen)) {
        std::cout << "Expected a '(' after if keyword." << std::endl;
        return nullptr;
    }

    advanceToken();

    auto exp = parseExpression();

    if (!exp) {
        return nullptr;
    }

    if (!curToken->is(TokenKind::rParen)) {
        std::cout << "Expected a ')' after condition." << cursor << std::endl;
        return nullptr;
    }

    advanceToken();

    auto truePart = parseStatement();

    if (!truePart) {
        std::cout << "If statement is null" << std::endl;
        return nullptr;
    }

    if (curToken->is(TokenKind::kw_else)) {
        advanceToken();

        auto falsePart = parseStatement();

        if (!falsePart) {
            return nullptr;
        }

        return std::make_unique<IfStatement>(std::move(exp), std::move(truePart), std::move(falsePart));

    } else {
        return std::make_unique<IfStatement>(std::move(exp), std::move(truePart), nullptr);
    }
}

std::unique_ptr<CaseLabel> Parser::parseCaseLabel() {
    if (!curToken->is(TokenKind::kw_case)) {
        return nullptr;
    }
    
    advanceToken();

    auto exp = parseExpression();

    if (!exp) {
        return nullptr;
    }

    if (!curToken->is(TokenKind::colon)) {
        std::cout << "Expected a ':' after case label." << std::endl;
        return nullptr;
    }

    advanceToken();

    return std::make_unique<CaseLabel>(std::move(exp));
}

std::unique_ptr<DefaultLabel> Parser::parseDefaultLabel() {
    if (!curToken->is(TokenKind::kw_default)) {
        return nullptr;
    }

    advanceToken();

    if (!curToken->is(TokenKind::colon)) {
        std::cout << "Expected a ':' after case label." << std::endl;
        return nullptr;
    }

    advanceToken();

    return std::make_unique<DefaultLabel>();
}

std::unique_ptr<Statement> Parser::parseStatement() {
    if (auto stmt = parseSimpleStatement()) {
        return stmt;
    } else if (auto compStmt = parseCompoundStatement()) {
        return compStmt;
    } else {
        return nullptr;
    }
}

std::unique_ptr<Statement> Parser::parseCompoundStatement() {
    if (!curToken->is(TokenKind::lCurly)) {
        return nullptr;
    }

    advanceToken();

    if (auto stmtList = parseStatementList()) {
        if (!curToken->is(TokenKind::rCurly)) {
            std::cout << "Expected '}' after statement list." << cursor << std::endl;
            return nullptr;
        } else {
            advanceToken();
            return std::move(stmtList);
        }
    } else {
        return nullptr;
    }
}

std::unique_ptr<StatementList> Parser::parseStatementList() {
    std::vector<std::unique_ptr<Statement>> statements;

    while (true) {
        if (auto stmt = parseStatement()) {
            statements.push_back(std::move(stmt));
        } else {
            break;
        }
    }

    return std::make_unique<StatementList>(std::move(statements));
}

std::unique_ptr<Statement> Parser::parseSimpleStatement() {
    if (auto decl = parseDeclaration()) {
        return std::move(decl);
    } else if (auto assignment = parseAssignmentExpression()) {
        return std::move(assignment);
    } else if (auto switchStmt = parseSwitchStatement()) {
        return std::move(switchStmt);
    } else if (auto caseLabelStmt = parseCaseLabel()) {
        return std::move(caseLabelStmt);
    } else if (auto defaultLabelStmt = parseDefaultLabel()) {
        return std::move(defaultLabelStmt);
    } else if (auto whileStmt = parseWhileStatement()) {
        return std::move(whileStmt);
    } else if (auto doStmt = parseDoStatement()) {
        return std::move(doStmt);
    } else if (auto ifStmt = parseIfStatement()) {
        return std::move(ifStmt);
    } else if (auto returnStmt = parseReturn()) {
        return std::move(returnStmt);
    } else if (auto breakStmt = parseBreak()) {
        return std::move(breakStmt);
    } else if (auto continueStmt = parseContinue()) {
        return std::move(continueStmt);
    } else if (auto discardStmt = parseDiscard()) {
        return std::move(discardStmt);
    } else {
        return nullptr;
    }
}

std::unique_ptr<ReturnStatement> Parser::parseReturn() {
    if (!curToken->is(TokenKind::kw_return)) {
        return nullptr;
    }

    advanceToken();

    if (curToken->is(TokenKind::semiColon)) {
        return std::make_unique<ReturnStatement>();
    }

    auto exp = parseExpression();

    if (!curToken->is(TokenKind::semiColon)) {
        std::cout << "Expected semicolon after return statement." << std::endl;
        return nullptr;
    }

    advanceToken();

    return std::make_unique<ReturnStatement>(std::move(exp));
}

std::unique_ptr<AssignmentExpression> Parser::parseAssignmentExpression() {
    if (!curToken->is(TokenKind::Identifier) && !Parser::getAssignmentOperatorFromTokenKind(peek(1)->getTokenKind())) {
        return nullptr;
    }

    auto name = curToken->getIdentifierName();

    advanceToken();

    auto op = Parser::getAssignmentOperatorFromTokenKind(curToken->getTokenKind());

    advanceToken();

    auto exp = parseExpression();

    if (!curToken->is(TokenKind::semiColon)) {
        std::cout << "Expected a semicolon after assignment expression." << cursor << std::endl;
    }

    advanceToken();

    return std::make_unique<AssignmentExpression>(name, *op, std::move(exp));
}


std::unique_ptr<DiscardStatement> Parser::parseDiscard() {
    if (!curToken->is(TokenKind::kw_discard)) {
        return nullptr;
    }

    advanceToken();

    if (!curToken->is(TokenKind::semiColon)) {
        std::cout << "Expected semicolon after discard statement." << std::endl;
        return nullptr;
    }

    advanceToken();

    return std::make_unique<DiscardStatement>();
}

std::unique_ptr<BreakStatement> Parser::parseBreak() {
    if (!curToken->is(TokenKind::kw_break)) {
        return nullptr;
    }

    advanceToken();

    if (!curToken->is(TokenKind::semiColon)) {
        std::cout << "Expected semicolon after break statement." << std::endl;
        return nullptr;
    }

    advanceToken();

    return std::make_unique<BreakStatement>();
}

std::unique_ptr<ContinueStatement> Parser::parseContinue() {
    if (!curToken->is(TokenKind::kw_continue)) {
        return nullptr;
    }

    advanceToken();

     if (!curToken->is(TokenKind::semiColon)) {
        std::cout << "Expected semicolon after continue statement." << std::endl;
        return nullptr;
    }

    advanceToken();

    return std::make_unique<ContinueStatement>();
}

std::unique_ptr<Expression> Parser::parsePrimaryExpression() {
    if (auto callExp = parseCallExpression()) {
        return callExp;
    } else if (curToken->is(TokenKind::Identifier)) {
        return std::make_unique<VariableExpression>(curToken->getIdentifierName());
    } else if (curToken->is(TokenKind::IntegerConstant)) {
        auto int_const = dynamic_cast<IntegerLiteral*>(curToken->getLiteralData());

        return std::make_unique<IntegerConstantExpression>(int_const->getVal());
    } else if (curToken->is(TokenKind::UnsignedIntegerConstant)) {
        auto uint_const = dynamic_cast<UnsignedIntegerLiteral*>(curToken->getLiteralData());

        return std::make_unique<UnsignedIntegerConstantExpression>(uint_const->getVal());
    } else if (curToken->is(TokenKind::FloatConstant)) {
        auto float_const = dynamic_cast<FloatLiteral*>(curToken->getLiteralData());

        return std::make_unique<FloatConstantExpression>(float_const->getVal());
    } else if (curToken->is(TokenKind::DoubleConstant)) {
        auto double_const = dynamic_cast<DoubleLiteral*>(curToken->getLiteralData());

        return std::make_unique<DoubleConstantExpression>(double_const->getVal());
    } else if (curToken->is(TokenKind::kw_true) || curToken->is(TokenKind::kw_false)) {
        return std::make_unique<BoolConstantExpression>(curToken->is(TokenKind::kw_true));
    } else if (curToken->is(TokenKind::lParen)) {
        advanceToken();
        auto exp = parseExpression();

        if (!exp) {
            return nullptr;
        }

        if (!curToken->is(TokenKind::rParen)) {
            std::cout << "Expected a ')' after expression." << std::endl;
            return nullptr;
        }
        
        return exp;
    }

    return nullptr;
}

std::unique_ptr<Expression> Parser::parseUnaryExpression() {
    if (curToken->is(TokenKind::increment)) {
        advanceToken();
        auto unaryOp = UnaryOperator::Inc;

        return std::make_unique<UnaryExpression>(unaryOp, parseUnaryExpression());
    } else if (curToken->is(TokenKind::decrement)) {
        advanceToken();

        auto unaryOp = UnaryOperator::Dec;

        return std::make_unique<UnaryExpression>(unaryOp, parseUnaryExpression());
    } else if (auto unop = Parser::getUnaryOperatorFromTokenKind(curToken->getTokenKind())) {
        advanceToken();

        return std::make_unique<UnaryExpression>(*unop, parseUnaryExpression());
    } else {
        return parsePostfixExpression();
    }
}

std::unique_ptr<Expression> Parser::parsePostfixExpression() {
    if (auto callExpr = parseCallExpression()) {
        return callExpr;
    } else if (auto primary = parsePrimaryExpression()) {
        return primary;
    } else {
        return nullptr;
    }
}

std::unique_ptr<CallExpression> Parser::parseCallExpression() {
    if (!(curToken->is(TokenKind::Identifier) && peek(1)->is(TokenKind::lParen))) {
        return nullptr;
    }

    const std::string& name = curToken->getIdentifierName();

    advanceToken();
    advanceToken();

    if (curToken->is(TokenKind::rParen)) {
        return std::make_unique<CallExpression>(name);
    }

    std::vector<std::unique_ptr<Expression>> arguments;

    do {
        auto exp = parseExpression();

        if (!exp) {
            return nullptr;
        }

        arguments.push_back(std::move(exp));

        if (curToken->is(TokenKind::comma)) {
            advanceToken();
        } else {
            break;
        }
    } while(true);

    if (!curToken->is(TokenKind::rParen)) {
        std::cout << "Expected a ')' after parameter list." << std::endl;
        return nullptr;
    }

    return std::make_unique<CallExpression>(name, std::move(arguments));
}

void Parser::advanceToken() {
    curToken = tokenStream[++cursor].get();
}

const Token* Parser::peek(int k) {
    if ((size_t)(cursor + k) >= tokenStream.size()) {
        auto tok = std::make_unique<Token>();
        tok->setTokenKind(TokenKind::Eof);
        tokenStream.push_back(std::move(tok));
        return tokenStream[tokenStream.size() - 1].get();
    } else {
        return tokenStream[cursor + k].get();
    }
}

};

};
