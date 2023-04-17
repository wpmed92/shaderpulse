#pragma once
#include "Lexer/Token.h"
#include "AST/AST.h"
#include "AST/Types.h"
#include <vector>
#include <map>
#include <optional>

using namespace shaderpulse::lexer;

namespace shaderpulse {

namespace parser {

class Parser {

public:
    Parser(std::vector<std::unique_ptr<Token>>& tokens) : 
        tokenStream(tokens), 
        cursor(-1),
        curToken(nullptr) {
           advanceToken();
    }

    std::unique_ptr<ast::TranslationUnit> parseTranslationUnit();
    std::unique_ptr<ast::ExternalDeclaration> parseExternalDeclaration();
    std::unique_ptr<ast::FunctionDeclaration> parseFunctionDeclaration();
    std::vector<std::unique_ptr<ast::ParameterDeclaration>> parseFunctionParameters();
    std::unique_ptr<ast::ValueDeclaration> parseDeclaration();
    std::unique_ptr<ast::Expression> parsePrimaryExpression();
    std::unique_ptr<ast::ForStatement> parseForLoop();
    std::unique_ptr<ast::SwitchStatement> parseSwitchStatement();
    std::unique_ptr<ast::WhileStatement> parseWhileStatement();
    std::unique_ptr<ast::DoStatement> parseDoStatement();
    std::unique_ptr<ast::IfStatement> parseIfStatement();
    std::unique_ptr<ast::Expression> parseExpression();
    std::unique_ptr<ast::Expression> parseRhs(int, std::unique_ptr<ast::Expression>);
    std::unique_ptr<ast::ReturnStatement> parseReturn();
    std::unique_ptr<ast::BreakStatement> parseBreak();
    std::unique_ptr<ast::ContinueStatement> parseContinue();
    std::unique_ptr<ast::DiscardStatement> parseDiscard();
    std::unique_ptr<ast::AssignmentExpression> parseAssignmentExpression();
    std::unique_ptr<ast::StatementList> parseStatementList();
    std::unique_ptr<ast::Statement> parseStatement();
    std::unique_ptr<ast::Statement> parseSimpleStatement();
    std::unique_ptr<ast::Statement> parseCompoundStatement();
    std::unique_ptr<ast::CaseLabel> parseCaseLabel();
    std::unique_ptr<ast::DefaultLabel> parseDefaultLabel();
    std::unique_ptr<ast::CallExpression> parseCallExpression();
    std::unique_ptr<ast::Expression> parseUnaryExpression();
    std::unique_ptr<ast::Expression> parsePostfixExpression();

private:
    std::vector<std::unique_ptr<Token>>& tokenStream;
    int cursor;
    Token* curToken;
    void advanceToken();
    const Token* peek(int);

    // TODO: move these to ast helpers
    static std::map<ast::BinaryOperator, int> binopPrecedence;
    static std::optional<ast::BinaryOperator> getBinaryOperatorFromTokenKind(TokenKind);
    static std::optional<ast::UnaryOperator> getUnaryOperatorFromTokenKind(TokenKind);
    static std::optional<ast::AssignmentOperator> getAssignmentOperatorFromTokenKind(TokenKind);
    static std::optional<std::unique_ptr<shaderpulse::Type>> getTypeFromTokenKind(TokenKind);
    static std::unique_ptr<shaderpulse::VectorType> makeVectorType(TypeKind, int);
    static std::unique_ptr<shaderpulse::MatrixType> makeMatrixType(TypeKind, int, int);
};

} // namespace shaderpulse

} // namespace parser