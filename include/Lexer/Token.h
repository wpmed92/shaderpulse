#pragma once
#include <memory>
#include <string>

namespace shaderpulse {

namespace lexer {

struct SourceLocation {
  SourceLocation() = default;
  SourceLocation(int line, int col) : line(line), col(col) { }
  int line;
  int col;
};

class NumericLiteral {
public:
  virtual ~NumericLiteral() = 0;
};

class IntegerLiteral : public NumericLiteral {
public:
  IntegerLiteral(int32_t val) : val(val) {}
  int32_t getVal() { return val; }

private:
  int32_t val;
};

class UnsignedIntegerLiteral : public NumericLiteral {
public:
  UnsignedIntegerLiteral(uint32_t val) : val(val) {}
  uint32_t getVal() { return val; }

private:
  uint32_t val;
};

class FloatLiteral : public NumericLiteral {
public:
  FloatLiteral(float val) : val(val) {}
  float getVal() { return val; }

private:
  float val;
};

class DoubleLiteral : public NumericLiteral {
public:
  DoubleLiteral(double val) : val(val) {}
  double getVal() { return val; }

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
  void setSourceLocation(SourceLocation loc);
  SourceLocation getSourceLocation() const;

private:
  TokenKind tokenKind;
  SourceLocation sourceLoc;
  std::string identifierName;
  std::unique_ptr<NumericLiteral> literalData;
};

}; // namespace lexer

}; // namespace shaderpulse
