#include "Lexer/Lexer.h"
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

namespace shaderpulse {

namespace lexer {

Lexer::Lexer(const std::string &characters) : characters(characters), curCharPos(0), savedCharPos(0), lineNum(1), col(1) { }

Error::Error() : kind(ErrorKind::None) { }

Error::Error(ErrorKind kind, const std::string &msg) : kind(kind), msg(msg) { }

bool Error::none() { return kind == ErrorKind::None; }

std::unordered_map<std::string, TokenKind> Lexer::stringKeywordToTokenKind = {
#define KEYWORD(X) {TOSTRING(X), TokenKind::kw_##X},
#include "Lexer/TokenDefs.h"
};

tl::expected<std::reference_wrapper<std::vector<std::unique_ptr<Token>>>, Error>
Lexer::lexCharacterStream() {
  std::string token;
  Error error;

  while ((size_t)curCharPos < characters.size()) {
    if (handleIdentifier(error))
      ;
    else if (handleHexLiteral(error))
      ;
    else if (handleOctalLiteral(error))
      ;
    else if (handleDecimalOrFloatLiteral(error))
      ;
    else if (handlePunctuator(error))
      ;
    else if (handleWhiteSpace(error))
      ;
    else if (handleNewLine(error))
      ;
    else if (!error.none()) {
      return tl::unexpected(error);
    } else {
      return tl::unexpected(
          Error(ErrorKind::UnexpectedCharacter,
                "Unexpected character: " + std::string(1, getCurChar())));
    }
  }

  auto tok = std::make_unique<Token>();
  tok->setTokenKind(TokenKind::Eof);
  tok->setSourceLocation(ast::SourceLocation(lineNum, col, lineNum, col));
  tokenStream.push_back(std::move(tok));

  return tokenStream;
}

bool Lexer::handleWhiteSpace(Error &error) {
  if (!error.none()) {
    return false;
  }

  if (isWhiteSpace(getCurChar())) {
    skipWhiteSpaces();
  } else {
    return false;
  }

  return true;
}

bool Lexer::handleNewLine(Error &error) {
  if (!error.none()) {
    return false;
  }

  if (isNewLine(getCurChar())) {
    lineNum++;
    advanceChar();
    col = 1;
    return true;
  }

  return false;
}

bool Lexer::handleIdentifier(Error &error) {
  if (!error.none()) {
    return false;
  }

  if (!isStartOfIdentifier(getCurChar())) {
    return false;
  }

  std::string token;
  int startCol = col;

  while (!isStopCharacter(getCurChar())) {
    if (!isAlphaNumeric(getCurChar())) {
      error = {ErrorKind::InvalidIdentififer,
               "Expected an alphanumeric character, got " +
                   std::string(1, getCurChar())};

      return false;
    }

    token += getCurChar();
    advanceChar();
  }

  auto tok = std::make_unique<Token>();
  tok->setIdentifierName(token);
  auto kwKind = getKwTokenKindFromString(token);

  if (kwKind.has_value()) {
    tok->setTokenKind(*kwKind);
  } else {
    tok->setTokenKind(TokenKind::Identifier);
  }

  tok->setSourceLocation(ast::SourceLocation(lineNum, startCol, lineNum, col));
  tok->setRawData(token);
  tokenStream.push_back(std::move(tok));

  return true;
}

// Handle literal tokens
bool Lexer::handleHexLiteral(Error &error) {
  if (!error.none()) {
    return false;
  }

  if (!isStartOfHexLiteral(getCurChar(), peekChar())) {
    return false;
  }

  bool hasDigit = false;
  std::string literalConstant;
  int startCol = col;
  literalConstant += getCurChar();
  advanceChar();
  literalConstant += getCurChar();
  advanceChar();

  while (!isStopCharacter(getCurChar()) && isHexDigit(getCurChar())) {
    literalConstant += getCurChar();
    advanceChar();
    hasDigit = true;
  }

  bool isUnsigned = isUnsignedSuffix(getCurChar(), peekChar());

  if (hasDigit && (isStopCharacter(getCurChar()) || isUnsigned)) {
    auto tok = std::make_unique<Token>();

    if (isUnsigned) {
      tok->setLiteralData(std::make_unique<UnsignedIntegerLiteral>(
          std::stoi(literalConstant, 0, 16)));
      advanceChar();
    } else {
      tok->setLiteralData(
          std::make_unique<IntegerLiteral>(std::stoi(literalConstant, 0, 16)));
    }

    tok->setTokenKind(isUnsigned ? TokenKind::UnsignedIntegerConstant
                                 : TokenKind::IntegerConstant);
    tok->setSourceLocation(ast::SourceLocation(lineNum, startCol, lineNum, col));
    tok->setRawData(literalConstant);
    tokenStream.push_back(std::move(tok));

    return true;
  } else {
    error = {ErrorKind::InvalidHexConstant,
             std::string("Expected a hexadecimal digit (0-F), but got ") +
                 std::string(1, getCurChar())};

    return false;
  }
}

bool Lexer::handleOctalLiteral(Error &error) {
  if (!error.none()) {
    return false;
  }

  if (getCurChar() != '0' || peekChar() == '.') {
    return false;
  }

  std::string literalConstant;
  bool hasDigit = false;
  int startCol = col;

  while (!isStopCharacter(getCurChar()) && isOctalDigit(getCurChar())) {
    literalConstant += getCurChar();
    advanceChar();
    hasDigit = true;
  }

  bool isUnsigned = isUnsignedSuffix(getCurChar(), peekChar());

  if (hasDigit && (isStopCharacter(getCurChar()) || isUnsigned)) {
    auto tok = std::make_unique<Token>();

    if (isUnsigned) {
      tok->setLiteralData(std::make_unique<UnsignedIntegerLiteral>(
          std::stoi(literalConstant, 0, 8)));
      advanceChar();
    } else {
      tok->setLiteralData(
          std::make_unique<IntegerLiteral>(std::stoi(literalConstant, 0, 8)));
    }

    tok->setTokenKind(isUnsigned ? TokenKind::UnsignedIntegerConstant
                                 : TokenKind::IntegerConstant);
    tok->setSourceLocation(ast::SourceLocation(lineNum, startCol, lineNum, col));
    tok->setRawData(literalConstant);
    tokenStream.push_back(std::move(tok));

    return true;
  } else {
    error = {ErrorKind::InvalidOctalConstant,
             std::string("Expected a octal digit (0-7), but got ") +
                 std::string(1, getCurChar())};

    return false;
  }
}

bool Lexer::handleDecimalOrFloatLiteral(Error &error) {
  if (!error.none()) {
    return false;
  }

  bool hasDigit = false;
  std::string literalConstant;
  int startCol = col;

  // Parse decimal literal constant
  while (!isStopCharacter(getCurChar()) && isDecimalDigit(getCurChar())) {
    literalConstant += getCurChar();
    advanceChar();
    hasDigit = true;
  }

  // Handle 1e-14 form
  if (hasDigit && handleExponentialForm(literalConstant, error))
    return true;
  // Handle fractional part digit . fractional exp, . fractional exp
  else if (getCurChar() == '.' && (peekExponentialPart() || isDecimalDigit(peekChar()) || hasDigit)) {
    literalConstant += getCurChar();
    advanceChar();

    while (!isStopCharacter(getCurChar()) && isDecimalDigit(getCurChar())) {
      literalConstant += getCurChar();
      advanceChar();
    }

    if (handleExponentialForm(literalConstant, error)) {
      return true;
    }

    SuffixCheckResult suffixCheck = handleFloatSuffix();

    if (isStopCharacter(getCurChar())) {
      auto tok = std::make_unique<Token>();

      if (suffixCheck == SuffixCheckResult::Double) {
        tok->setTokenKind(TokenKind::DoubleConstant);
        tok->setLiteralData(
            std::make_unique<DoubleLiteral>(std::stod(literalConstant)));
      } else {
        tok->setTokenKind(TokenKind::FloatConstant);
        tok->setLiteralData(
            std::make_unique<FloatLiteral>(std::stof(literalConstant)));
      }

      tok->setSourceLocation(ast::SourceLocation(lineNum, startCol, lineNum, col));
      tok->setRawData(literalConstant);
      tokenStream.push_back(std::move(tok));
      return true;
    }
  } else if (hasDigit) {
    auto tok = std::make_unique<Token>();

    if (isUnsignedSuffix(getCurChar(), peekChar())) {
      tok->setLiteralData(std::make_unique<UnsignedIntegerLiteral>(
          std::stoi(literalConstant, 0, 10)));
      tok->setTokenKind(TokenKind::UnsignedIntegerConstant);
      advanceChar();
    } else {
      tok->setLiteralData(
          std::make_unique<IntegerLiteral>(std::stoi(literalConstant, 0, 10)));
      tok->setTokenKind(TokenKind::IntegerConstant);
    }

    tok->setSourceLocation(ast::SourceLocation(lineNum, startCol, lineNum, col));
    tok->setRawData(literalConstant);
    tokenStream.push_back(std::move(tok));
    return true;
  } else {
    return false;
  }
}

bool Lexer::handleExponentialForm(std::string &literalConstant, Error &error) {
  // TODO: combine with 'peekExponentialPart'
  if (!(isExponentFlag(getCurChar()) && (peekChar() == '-' || peekChar() == '+' || isDecimalDigit(peekChar())))) {
    return false;
  }

  literalConstant += getCurChar();
  advanceChar();
  int startCol = col;

  if (isSign(getCurChar())) {
    literalConstant += getCurChar();
    advanceChar();
  }

  std::string decimalPart;

  // Parse decimal literal constant
  while (!isStopCharacter(getCurChar()) && isDecimalDigit(getCurChar())) {
    decimalPart += getCurChar();
    advanceChar();
  }

  if (decimalPart.empty()) {
    error = {ErrorKind::UnexpectedCharacter,
             "Expected a decimal digit, but got " +
                 std::string(1, getCurChar())};

    return false;
  }

  SuffixCheckResult suffixCheck = handleFloatSuffix();

  if (isStopCharacter(getCurChar())) {
    literalConstant += decimalPart;
    auto tok = std::make_unique<Token>();

    if (suffixCheck == SuffixCheckResult::Double) {
      tok->setLiteralData(
          std::make_unique<DoubleLiteral>(std::stod(literalConstant)));
      tok->setTokenKind(TokenKind::DoubleConstant);
    } else {
      tok->setLiteralData(
          std::make_unique<FloatLiteral>(std::stof(literalConstant)));
      tok->setTokenKind(TokenKind::FloatConstant);
    }

    tok->setSourceLocation(ast::SourceLocation(lineNum, startCol, lineNum, col));
    tok->setRawData(literalConstant);
    tokenStream.push_back(std::move(tok));

    return true;
  } else {
    error = {ErrorKind::UnexpectedCharacter,
             "Unexpected character:  " + std::string(1, getCurChar())};

    return false;
  }
}

bool Lexer::handlePunctuator(Error &error) {
  if (!error.none()) {
    return false;
  }

  if (!isPunctuator(getCurChar())) {
    return false;
  }

  savedCharPos = curCharPos;

  switch (getCurChar()) {
  case '<': {
    if (peekChar() == '<') {
      advanceChar();

      if (peekChar() == '=') {
        advanceChar();
        addToken(TokenKind::shiftLeftAssign);
      } else {
        addToken(TokenKind::shiftLeft);
      }
    } else if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::ltEq);
    } else {
      addToken(TokenKind::lt);
    }

    break;
  }
  case '>': {
    if (peekChar() == '>') {
      advanceChar();

      if (peekChar() == '=') {
        advanceChar();
        addToken(TokenKind::shiftRightAssign);
      } else {
        addToken(TokenKind::shiftRight);
      }
    } else if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::gtEq);
    } else {
      addToken(TokenKind::gt);
    }

    break;
  }
  case '=': {
    if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::eq);
    } else {
      addToken(TokenKind::assign);
    }

    break;
  }
  case '!': {
    if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::neq);
    } else {
      addToken(TokenKind::lnot);
    }

    break;
  }
  case '+': {
    if (peekChar() == '+') {
      advanceChar();
      addToken(TokenKind::increment);
    } else if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::addAssign);
    } else {
      addToken(TokenKind::plus);
    }

    break;
  }
  case '-': {
    if (peekChar() == '-') {
      advanceChar();
      addToken(TokenKind::decrement);
    } else if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::subAssign);
    } else {
      addToken(TokenKind::minus);
    }

    break;
  }
  case '*': {
    if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::mulAssign);
    } else {
      addToken(TokenKind::mul);
    }

    break;
  }
  case '/': {
    if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::divAssign);
    } else {
      addToken(TokenKind::div);
    }

    break;
  }
  case '%': {
    if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::modAssign);
    } else {
      addToken(TokenKind::modulo);
    }

    break;
  }
  case '&': {
    if (peekChar() == '&') {
      advanceChar();
      addToken(TokenKind::land);
    } else if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::landAssign);
    } else {
      addToken(TokenKind::band);
    }

    break;
  }
  case '^': {
    if (peekChar() == '^') {
      advanceChar();
      addToken(TokenKind::lxor);
    } else if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::lxorAssign);
    } else {
      addToken(TokenKind::bxor);
    }

    break;
  }
  case '|': {
    if (peekChar() == '|') {
      advanceChar();
      addToken(TokenKind::lor);
    } else if (peekChar() == '=') {
      advanceChar();
      addToken(TokenKind::lorAssign);
    } else {
      addToken(TokenKind::bor);
    }

    break;
  }
  case ',':
    addToken(TokenKind::comma);
    break;
  case '~':
    addToken(TokenKind::bnot);
    break;
  case '(':
    addToken(TokenKind::lParen);
    break;
  case ')':
    addToken(TokenKind::rParen);
    break;
  case '[':
    addToken(TokenKind::lBracket);
    break;
  case ']':
    addToken(TokenKind::rBracket);
    break;
  case '{':
    addToken(TokenKind::lCurly);
    break;
  case '}':
    addToken(TokenKind::rCurly);
    break;
  case '.':
    addToken(TokenKind::dot);
    break;
  case ';':
    addToken(TokenKind::semiColon);
    break;
  case ':':
    addToken(TokenKind::colon);
    break;
  case '?':
    addToken(TokenKind::question);
    break;
  default:
    return false;
  }

  advanceChar();

  return true;
}

SuffixCheckResult Lexer::handleFloatSuffix() {
  if (getCurChar() == 'f' || getCurChar() == 'F') {
    advanceChar();
    return SuffixCheckResult::Float;
  } else if ((getCurChar() == 'l' && peekChar() == 'f') || (getCurChar() == 'L' && peekChar() == 'F')) {
    advanceChar();
    advanceChar();
    return SuffixCheckResult::Double;
  } else {
    return SuffixCheckResult::None;
  }
}

bool Lexer::peekExponentialPart() {
  return isExponentFlag(peekChar()) && (peekChar(2) == '-' || peekChar(2) == '+' || isDecimalDigit(peekChar(2)));
}

void Lexer::addToken(TokenKind kind) {
  auto tok = std::make_unique<Token>();
  tok->setTokenKind(kind);
  int tokenLength = curCharPos - savedCharPos + 1;
  tok->setSourceLocation(ast::SourceLocation(lineNum, col - tokenLength, lineNum, col));
  std::string rawData = characters.substr(savedCharPos, tokenLength);
  tok->setRawData(rawData);
  tokenStream.push_back(std::move(tok));
}

void Lexer::skipWhiteSpaces() {
  while (isWhiteSpace(getCurChar()) && (size_t)curCharPos < characters.size()) {
    curCharPos++;
    col++;
  }
}

// Helpers

bool Lexer::isNewLine(char c) const { return c == '\n' || c == '\r' || c == '\t'; }

bool Lexer::isWhiteSpace(char c) const { return c == ' '; }

bool Lexer::isSign(char c) const { return (c == '+') || (c == '-'); }

bool Lexer::isExponentFlag(char curChar) const {
  return curChar == 'e' || curChar == 'E';
}

bool Lexer::isUnsignedSuffix(char curChar, char nextChar) const {
  return (isWhiteSpace(nextChar) || isPunctuator(nextChar) ||
          isNewLine(nextChar)) &&
         (curChar == 'u' || curChar == 'U');
}

bool Lexer::isDecimalDigit(char c) const { return c >= '0' && c <= '9'; }

bool Lexer::isOctalDigit(char c) const { return c >= '0' && c <= '7'; }

bool Lexer::isHexDigit(char c) const {
  return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F');
}

bool Lexer::isAlphaNumeric(char c) const {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || (c == '_');
}

bool Lexer::isStartOfIdentifier(char c) const {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
}

bool Lexer::isStartOfHexLiteral(char c, char nc) const {
  return c == '0' && (nc == 'x' || nc == 'X');
}

bool Lexer::isPunctuator(char c) const {
  return c == '(' || c == ')' || c == '*' || c == '-' || c == '+' || c == '{' ||
         c == '}' || c == '[' || c == ']' || c == '/' || c == '%' || c == '!' ||
         c == ',' || c == '.' || c == ';' || c == '=' || c == '<' || c == '>' ||
         c == '&' || c == '|' || c == '^' || c == '~' || c == '?' || c == ':';
}

bool Lexer::isStopCharacter(char c) const {
  return isWhiteSpace(c) || isNewLine(c) || isPunctuator(c);
}

bool Lexer::isStartOfFractionalPart(char c) const {
  return c == '.' || isExponentFlag(c);
}

std::string Lexer::getSpelling(TokenKind kind) {
  switch (kind) {
#define PUNCTUATOR(X, Y)                                                       \
  case TokenKind::X:                                                           \
    return TOSTRING(X);
#define KEYWORD(X)                                                             \
  case TokenKind::kw_##X:                                                      \
    return TOSTRING(kw_##X);
#include "Lexer/TokenDefs.h"
  case TokenKind::IntegerConstant:
    return "integer literal";
  case TokenKind::Identifier:
    return "identifier";
  case TokenKind::FloatConstant:
    return "float literal";
  case TokenKind::DoubleConstant:
    return "double literal";
  case TokenKind::UnsignedIntegerConstant:
    return "unsigned integer literal";
  case TokenKind::Eof:
    return "eof";
  }

  return "";
}

std::string
Lexer::printTokens(const std::vector<std::unique_ptr<Token>> &tokenStream) {
  std::string textForm;

  for (auto &tok : tokenStream) {
    textForm += "'" + getSpelling(tok->getTokenKind()) + "'";

    if (!tok->getIdentifierName().empty()) {
      textForm += " " + tok->getIdentifierName();
    }

    if (tok->isDoubleConstant()) {
      textForm +=
          " " +
          std::to_string(
              dynamic_cast<DoubleLiteral *>(tok->getLiteralData())->getVal());
    } else if (tok->isFloatConstant()) {
      textForm +=
          " " +
          std::to_string(
              dynamic_cast<FloatLiteral *>(tok->getLiteralData())->getVal());
    } else if (tok->isIntegerConstant()) {
      textForm +=
          " " +
          std::to_string(
              dynamic_cast<IntegerLiteral *>(tok->getLiteralData())->getVal());
    } else if (tok->isUnsignedIntegerConstant()) {
      textForm += " " + std::to_string(dynamic_cast<UnsignedIntegerLiteral *>(
                                           tok->getLiteralData())
                                           ->getVal());
    }

    textForm += "\n";
  }

  return textForm;
}

std::optional<TokenKind>
Lexer::getKwTokenKindFromString(const std::string &kw) const {
  auto kwKind = stringKeywordToTokenKind.find(kw);

  if (kwKind != stringKeywordToTokenKind.end()) {
    return kwKind->second;
  }

  return std::nullopt;
}

void Lexer::advanceChar() { col++; curCharPos++; }

char Lexer::getCurChar() const {
  if (curCharPos < characters.size()) {
    return characters[curCharPos];
  } else {
    return ' ';
  }
}

char Lexer::peekChar(int n) const { 
  if (curCharPos < characters.size() - n) {
    return characters[curCharPos + n]; 
  } else {
    return ' ';
  }
}

const std::vector<std::unique_ptr<Token>> &Lexer::getTokenStream() const {
  return tokenStream;
}

}; // namespace lexer

}; // namespace shaderpulse
