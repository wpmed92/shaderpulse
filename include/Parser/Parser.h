#pragma once
#include "AST/AST.h"
#include "AST/Types.h"
#include "Lexer/Token.h"
#include <map>
#include <optional>
#include <vector>

using namespace shaderpulse::lexer;

namespace shaderpulse {

using namespace ast;

namespace parser {

enum ParserErrorKind {
  None,
  UnexpectedToken,
  ExpectedToken
};

struct ParserError {
  ParserError() : kind(ParserErrorKind::None) {}
  ParserError(ParserErrorKind kind, const std::string &msg) : kind(kind), msg(msg) {}

  ParserErrorKind kind;
  std::string msg;

  bool none() { return kind == ParserErrorKind::None; }
};

class Parser {

public:
  Parser(std::vector<std::unique_ptr<Token>> &tokens)
      : tokenStream(tokens), cursor(-1), curToken(nullptr), parsingLhsExpression(false) {
    advanceToken();
  }

  std::unique_ptr<TranslationUnit> parseTranslationUnit();
  std::unique_ptr<ExternalDeclaration> parseExternalDeclaration();
  std::unique_ptr<FunctionDeclaration> parseFunctionDeclaration();
  std::vector<std::unique_ptr<ParameterDeclaration>> parseFunctionParameters();
  std::unique_ptr<Declaration> parseDeclaration();
  std::unique_ptr<VariableDeclarationList>
  parseVariableDeclarationList(std::unique_ptr<Type>, const std::string &,
                               std::unique_ptr<Expression>);
  std::unique_ptr<Expression> parsePrimaryExpression();
  std::unique_ptr<ForStatement> parseForStatement();
  std::unique_ptr<SwitchStatement> parseSwitchStatement();
  std::unique_ptr<WhileStatement> parseWhileStatement();
  std::unique_ptr<DoStatement> parseDoStatement();
  std::unique_ptr<IfStatement> parseIfStatement();
  std::unique_ptr<Expression> parseExpression();
  std::unique_ptr<Expression> parseRhs(int, std::unique_ptr<Expression>);
  std::unique_ptr<ReturnStatement> parseReturn();
  std::unique_ptr<BreakStatement> parseBreak();
  std::unique_ptr<ContinueStatement> parseContinue();
  std::unique_ptr<DiscardStatement> parseDiscard();
  std::unique_ptr<AssignmentExpression> parseAssignmentExpression();
  std::unique_ptr<StatementList> parseStatementList();
  std::unique_ptr<Statement> parseStatement();
  std::unique_ptr<Statement> parseSimpleStatement();
  std::unique_ptr<Statement> parseCompoundStatement();
  std::unique_ptr<CaseLabel> parseCaseLabel();
  std::unique_ptr<DefaultLabel> parseDefaultLabel();
  std::unique_ptr<CallExpression> parseCallExpression();
  std::unique_ptr<Expression> parseConditionalExpression();
  std::unique_ptr<ConstructorExpression> parseConstructorExpression();
  std::unique_ptr<Expression> parseUnaryExpression();
  std::unique_ptr<Expression> parsePostfixExpression(bool parsingSubExpression = false);
  std::vector<std::unique_ptr<TypeQualifier>> parseQualifiers();
  std::unique_ptr<TypeQualifier> parseQualifier();
  std::unique_ptr<Type> parseType();
  std::unique_ptr<LayoutQualifier> parseLayoutQualifier();
  std::unique_ptr<StructDeclaration> parseStructDeclaration();
  std::optional<std::vector<std::unique_ptr<Expression>>> parseMemberAccessChain();
  std::optional<std::vector<std::unique_ptr<Expression>>> parseArrayAccess();

private:
  std::vector<std::unique_ptr<Token>> &tokenStream;
  std::map<std::string, bool> structDeclarations;
  std::vector<int> savedPositions;
  
  int cursor;
  Token *curToken;
  void advanceToken();
  const Token *peek(int);
  ParserError error;
  void reportError(ParserErrorKind kind, const std::string &msg);
  bool parsingLhsExpression;

  // TODO: move these to ast helpers
  static std::map<BinaryOperator, int> binopPrecedence;
  static std::optional<BinaryOperator>
      getBinaryOperatorFromTokenKind(TokenKind);
  static std::optional<UnaryOperator> getUnaryOperatorFromTokenKind(TokenKind);
  static std::optional<AssignmentOperator>
      getAssignmentOperatorFromTokenKind(TokenKind);
  std::unique_ptr<shaderpulse::Type>
      getTypeFromTokenKind(std::vector<std::unique_ptr<TypeQualifier>>,
                           TokenKind);
  static std::unique_ptr<shaderpulse::VectorType>
  makeVectorType(std::vector<std::unique_ptr<TypeQualifier>>, TypeKind, int);
  static std::unique_ptr<shaderpulse::MatrixType>
  makeMatrixType(std::vector<std::unique_ptr<TypeQualifier>>, TypeKind, int,
                 int);

  void savePosition() {
    savedPositions.push_back(cursor);
  }

  void rollbackPosition() {
    cursor = savedPositions.back() - 1;
    savedPositions.pop_back();
    advanceToken();
  }
};

} // namespace parser

} // namespace shaderpulse
