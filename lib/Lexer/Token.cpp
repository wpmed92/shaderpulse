#include "Lexer/Token.h"

namespace shaderpulse {

namespace lexer {

// Literals
NumericLiteral::~NumericLiteral(){};

// signed
IntegerLiteral::IntegerLiteral(int32_t val) : val(val) {}

int32_t IntegerLiteral:: getVal() { return val; }

// unsigned
UnsignedIntegerLiteral::UnsignedIntegerLiteral(uint32_t val) : val(val) {}

uint32_t UnsignedIntegerLiteral::getVal() { return val; }

// float
FloatLiteral::FloatLiteral(float val) : val(val) {}

float FloatLiteral::getVal() { return val; }

// double
DoubleLiteral::DoubleLiteral(double val) : val(val) {}

double DoubleLiteral::getVal() { return val; }

// Token
void Token::setTokenKind(TokenKind kind) { tokenKind = kind; }

TokenKind Token::getTokenKind() const { return tokenKind; }

bool Token::is(TokenKind kind) const { return tokenKind == kind; }

void Token::setIdentifierName(const std::string &idName) {
  identifierName = idName;
}

std::string Token::getIdentifierName() const { return identifierName; }

bool Token::isIntegerConstant() const {
  return tokenKind == TokenKind::IntegerConstant;
}

bool Token::isUnsignedIntegerConstant() const {
  return tokenKind == TokenKind::UnsignedIntegerConstant;
}

bool Token::isFloatConstant() const {
  return tokenKind == TokenKind::FloatConstant;
}

bool Token::isDoubleConstant() const {
  return tokenKind == TokenKind::DoubleConstant;
}

NumericLiteral *Token::getLiteralData() const { return literalData.get(); }

void Token::setLiteralData(std::unique_ptr<NumericLiteral> literalData) {
  this->literalData = std::move(literalData);
}

void Token::setSourceLocation(ast::SourceLocation loc) {
  sourceLoc = loc;
}

ast::SourceLocation Token::getSourceLocation() const {
  return sourceLoc;
}

void Token::setRawData(const std::string& rawData) {
  this->rawData = rawData;
}

std::string Token::getRawData() const {
  return rawData;
}

}; // namespace lexer

}; // namespace shaderpulse
