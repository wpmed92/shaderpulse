
#pragma once
#include "Token.h"
#include <expected.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace shaderpulse {

namespace lexer {

enum class ErrorKind {
  None,
  InvalidOctalConstant,
  InvalidHexConstant,
  InvalidDecimalConstant,
  InvalidFloatConstant,
  InvalidConstant,
  InvalidIdentififer,
  UnexpectedCharacter
};

enum class SuffixCheckResult {
  None,
  Float,
  Double
};

struct Error {
  Error();
  Error(ErrorKind kind, const std::string &msg);

  ErrorKind kind;
  std::string msg;

  bool none();
};

class Lexer {
public:
  Lexer(const std::string &characters);
  tl::expected<std::reference_wrapper<std::vector<std::unique_ptr<Token>>>,
               Error>
  lexCharacterStream();
  static std::string
  printTokens(const std::vector<std::unique_ptr<Token>> &tokenStream);
  const std::vector<std::unique_ptr<Token>> &getTokenStream() const;

private:
  std::string characters;
  int curCharPos;
  int savedCharPos;
  int lineNum;
  int col;
  std::vector<std::unique_ptr<Token>> tokenStream;
  char getCurChar() const;
  void advanceChar();
  char peekChar(int n = 1) const;
  bool peekExponentialPart();
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

  SuffixCheckResult handleFloatSuffix();

  std::optional<TokenKind>
  getKwTokenKindFromString(const std::string &kw) const;
  static std::string getSpelling(TokenKind);
  static std::unordered_map<std::string, TokenKind> stringKeywordToTokenKind;
};

}; // namespace lexer

}; // namespace shaderpulse
