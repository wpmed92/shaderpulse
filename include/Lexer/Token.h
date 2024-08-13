#pragma once
#include "AST/AST.h"
#include <memory>
#include <string>

namespace shaderpulse {

namespace lexer {

class NumericLiteral {
public:
  virtual ~NumericLiteral() = 0;
};

class IntegerLiteral : public NumericLiteral {
public:
  IntegerLiteral(int32_t val);
  int32_t getVal();

private:
  int32_t val;
};

class UnsignedIntegerLiteral : public NumericLiteral {
public:
  UnsignedIntegerLiteral(uint32_t val);
  uint32_t getVal();

private:
  uint32_t val;
};

class FloatLiteral : public NumericLiteral {
public:
  FloatLiteral(float val);
  float getVal();

private:
  float val;
};

class DoubleLiteral : public NumericLiteral {
public:
  DoubleLiteral(double val);
  double getVal();

private:
  double val;
};

enum TokenKind {
  IntegerConstant,
  UnsignedIntegerConstant,
  FloatConstant,
  DoubleConstant,
  Identifier,
  Eof,
#define TOK(X) X,
#include "TokenDefs.h"
};

class Token {
public:
  Token() = default;
  Token(const Token &token) = delete;
  Token &operator=(const Token &other) = delete;
  Token(Token &&o) = default;

  void setTokenKind(TokenKind tokenKind);
  TokenKind getTokenKind() const;
  bool is(TokenKind tokenKind) const;
  void setIdentifierName(const std::string &);
  std::string getIdentifierName() const;
  bool isIntegerConstant() const;
  bool isFloatConstant() const;
  bool isUnsignedIntegerConstant() const;
  bool isDoubleConstant() const;
  NumericLiteral *getLiteralData() const;
  void setLiteralData(std::unique_ptr<NumericLiteral>);
  void setSourceLocation(ast::SourceLocation loc);
  ast::SourceLocation getSourceLocation() const;
  void setRawData(const std::string&);
  std::string getRawData() const;

private:
  TokenKind tokenKind;
  ast::SourceLocation sourceLoc;
  std::string identifierName;
  std::string rawData;
  std::unique_ptr<NumericLiteral> literalData;
};

}; // namespace lexer

}; // namespace shaderpulse
