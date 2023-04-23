
#pragma once
#include "Token.h"
#include <expected.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace shaderpulse {

namespace lexer {

enum ErrorKind {
  None,
  InvalidOctalConstant,
  InvalidHexConstant,
  InvalidDecimalConstant,
  InvalidFloatConstant,
  InvalidConstant,
  InvalidIdentififer,
  UnexpectedCharacter
};

struct Error {
  Error() : kind(ErrorKind::None) {}
  Error(ErrorKind kind, const std::string &msg) : kind(kind), msg(msg) {}

  ErrorKind kind;
  std::string msg;

  bool none() { return kind == ErrorKind::None; }
};

class Lexer {
public:
  Lexer(const std::string &characters)
      : characters(characters), curCharPos(0), lineNum(0) {}

  tl::expected<std::reference_wrapper<std::vector<std::unique_ptr<Token>>>,
               Error>
  lexCharacterStream();
  static std::string
  printTokens(const std::vector<std::unique_ptr<Token>> &tokenStream);
  const std::vector<std::unique_ptr<Token>> &getTokenStream() const;

private:
  std::string characters;
  int curCharPos;
  int lineNum;
  std::vector<std::unique_ptr<Token>> tokenStream;
  char getCurChar() const;
  void advanceChar();
  char peekChar() const;
  void skipWhiteSpaces();
  void addToken(TokenKind);

  // Helpers
  bool isStartOfIdentifier(char) const;
  bool isStartOfHexLiteral(char, char) const;
  bool isPunctuator(char) const;
  bool isWhiteSpace(char) const;
  bool isSign(char) const;
  bool isExponentFlag(char) const;
  bool isUnsignedSuffix(char, char) const;
  bool isDecimalDigit(char) const;
  bool isOctalDigit(char) const;
  bool isHexDigit(char) const;
  bool isAlphaNumeric(char) const;
  bool isNewLine(char) const;
  bool isStartOfFractionalPart(char) const;
  bool isPrecisionSubfix(char, char) const;
  bool isStopCharacter(char) const;

  // Handlers
  bool handlePunctuator(Error &);
  bool handleIdentifier(Error &);
  bool handleWhiteSpace(Error &);
  bool handleNewLine(Error &);
  bool handleHexLiteral(Error &);
  bool handleOctalLiteral(Error &);
  bool handleExponentialForm(std::string &, Error &);
  bool handleDecimalOrFloatLiteral(Error &);

  std::optional<TokenKind>
  getKwTokenKindFromString(const std::string &kw) const;
  static std::string getSpelling(TokenKind);
  static std::unordered_map<std::string, TokenKind> stringKeywordToTokenKind;
};

}; // namespace lexer

}; // namespace shaderpulse
