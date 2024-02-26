#include "Parser/Parser.h"
#include <iostream>

namespace shaderpulse {

using namespace ast;

namespace parser {

std::map<BinaryOperator, int> Parser::binopPrecedence = {
    {BinaryOperator::LogOr, 10},     {BinaryOperator::LogXor, 20},
    {BinaryOperator::LogAnd, 30},    {BinaryOperator::BitIor, 40},
    {BinaryOperator::BitXor, 50},    {BinaryOperator::BitAnd, 60},
    {BinaryOperator::Eq, 70},        {BinaryOperator::Neq, 70},
    {BinaryOperator::Gt, 80},        {BinaryOperator::Lt, 80},
    {BinaryOperator::GtEq, 80},      {BinaryOperator::LtEq, 80},
    {BinaryOperator::ShiftLeft, 90}, {BinaryOperator::ShiftRight, 90},
    {BinaryOperator::Add, 100},      {BinaryOperator::Sub, 100},
    {BinaryOperator::Mul, 110},      {BinaryOperator::Div, 110},
    {BinaryOperator::Mod, 110},
};

std::unique_ptr<TranslationUnit> Parser::parseTranslationUnit() {
  std::vector<std::unique_ptr<ExternalDeclaration>> externalDeclarations;

  while ((size_t)cursor < tokenStream.size() - 1) {
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
  } else if (auto structDecl = parseStructDeclaration()) {
    return structDecl;
  } else {
    return nullptr;
  }
}

std::unique_ptr<FunctionDeclaration> Parser::parseFunctionDeclaration() {
  savePosition();

  if (parseQualifier()) {
    rollbackPosition();
    return nullptr;
  }

  if (auto type = parseType()) {
    if (!(peek(1)->is(TokenKind::Identifier) &&
          peek(2)->is(TokenKind::lParen))) {
      return nullptr;
    }

    advanceToken();

    auto returnType = std::move(type);
    const std::string &functionName = curToken->getIdentifierName();

    advanceToken(); // eat lparen

    auto params = parseFunctionParameters();

    if (!curToken->is(TokenKind::rParen)) {
      std::cout << "Expected a ')' after function parameter declaration."
                << std::endl;
      return nullptr;
    }

    advanceToken();

    if (auto body = parseStatement()) {
      return std::make_unique<FunctionDeclaration>(
          std::move(returnType), functionName, std::move(params),
          std::move(body));
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

std::unique_ptr<Declaration> Parser::parseDeclaration() {
  if (auto type = parseType()) {
    advanceToken();

    // Only type, no variable name
    if (curToken->is(TokenKind::semiColon)) {
      advanceToken();
      return std::make_unique<VariableDeclaration>(std::move(type),
                                                   std::string(), nullptr);
    }

    const std::string &name = curToken->getIdentifierName();

    advanceToken();

    // Type and variable name, no initializer expression
    if (curToken->is(TokenKind::semiColon)) {
      advanceToken();
      return std::make_unique<VariableDeclaration>(std::move(type), name,
                                                   nullptr);
    } else if (curToken->is(TokenKind::assign)) {
      // Initializer expression
      advanceToken();

      auto exp = parseExpression();

      // Declaration list
      if (curToken->is(TokenKind::comma)) {
        advanceToken();
        auto declarations =
            parseVariableDeclarationList(std::move(type), name, std::move(exp));

        return declarations;
      } else if (curToken->is(TokenKind::semiColon)) {
        advanceToken();
        return std::make_unique<VariableDeclaration>(std::move(type), name,
                                                     std::move(exp));
      } else {
        std::cout << "Parser error: unexpected token." << cursor << std::endl;
      }
    } else if (curToken->is(TokenKind::comma)) {
      advanceToken();
      auto declarations =
          parseVariableDeclarationList(std::move(type), name, nullptr);

      return declarations;
    } else {
      return nullptr;
    }
  }

  return nullptr;
}

std::unique_ptr<VariableDeclarationList> Parser::parseVariableDeclarationList(
    std::unique_ptr<Type> type, const std::string &identifierName,
    std::unique_ptr<Expression> initializerExpression) {
  std::vector<std::unique_ptr<VariableDeclaration>> declarations;
  declarations.push_back(std::make_unique<VariableDeclaration>(
      nullptr, identifierName, std::move(initializerExpression)));

  do {
    if (!curToken->is(TokenKind::Identifier)) {
      std::cout << "Parser error: expected identifier" << std::endl;
    }

    const std::string &varName = curToken->getIdentifierName();

    advanceToken();

    if (curToken->is(TokenKind::assign)) {
      advanceToken();

      auto exp = parseExpression();

      if (!exp) {
        return nullptr;
      }

      declarations.push_back(std::make_unique<VariableDeclaration>(
          nullptr, varName, std::move(exp)));
    }
  } while (curToken->is(TokenKind::comma));

  if (!curToken->is(TokenKind::semiColon)) {
    std::cout << "Expected semicolon after variable declaration list."
              << std::endl;
  }

  advanceToken();
  return std::make_unique<VariableDeclarationList>(std::move(type),
                                                   std::move(declarations));
}

std::unique_ptr<ast::Expression> Parser::parseExpression() {
  auto lhs = parseUnaryExpression();

  advanceToken();

  if (!lhs) {
    return nullptr;
  }

  return parseRhs(0, std::move(lhs));
}

std::unique_ptr<ast::Expression>
Parser::parseRhs(int exprPrec, std::unique_ptr<ast::Expression> lhs) {
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
    lhs = std::make_unique<BinaryExpression>(*binop, std::move(lhs),
                                             std::move(rhs));
  }
}

std::optional<BinaryOperator>
Parser::getBinaryOperatorFromTokenKind(TokenKind kind) {
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

std::optional<ast::UnaryOperator>
Parser::getUnaryOperatorFromTokenKind(TokenKind kind) {
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

std::unique_ptr<TypeQualifier> Parser::parseQualifier() {
  switch (curToken->getTokenKind()) {
  // Storage
  case TokenKind::kw_uniform:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Uniform);
  case TokenKind::kw_buffer:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Buffer);
  case TokenKind::kw_const:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Const);
  case TokenKind::kw_in:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::In);
  case TokenKind::kw_out:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Out);
  case TokenKind::kw_inout:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Inout);
  case TokenKind::kw_centroid:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Centroid);
  case TokenKind::kw_patch:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Patch);
  case TokenKind::kw_sample:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Sample);
  case TokenKind::kw_shared:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Shared);
  case TokenKind::kw_coherent:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Coherent);
  case TokenKind::kw_volatile:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Volatile);
  case TokenKind::kw_restrict:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Restrict);
  case TokenKind::kw_readonly:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Readonly);
  case TokenKind::kw_writeonly:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Writeonly);
  case TokenKind::kw_subroutine:
    return std::make_unique<StorageQualifier>(StorageQualifierKind::Subroutine);

  // Precision
  case TokenKind::kw_highp:
    return std::make_unique<PrecisionQualifier>(PrecisionQualifierKind::High);
  case TokenKind::kw_mediump:
    return std::make_unique<PrecisionQualifier>(PrecisionQualifierKind::Medium);
  case TokenKind::kw_lowp:
    return std::make_unique<PrecisionQualifier>(PrecisionQualifierKind::Low);

  // Interpolation
  case TokenKind::kw_smooth:
    return std::make_unique<InterpolationQualifier>(
        InterpolationQualifierKind::Smooth);
  case TokenKind::kw_flat:
    return std::make_unique<InterpolationQualifier>(
        InterpolationQualifierKind::Flat);
  case TokenKind::kw_noperspective:
    return std::make_unique<InterpolationQualifier>(
        InterpolationQualifierKind::Noperspective);

  // Precise
  case TokenKind::kw_precise:
    return std::make_unique<PreciseQualifier>();

  // Invariant
  case TokenKind::kw_invariant:
    return std::make_unique<InvariantQualifier>();

  // Layout
  case TokenKind::kw_layout:
    return parseLayoutQualifier();

  default:
    return nullptr;
  }
}

std::unique_ptr<LayoutQualifier> Parser::parseLayoutQualifier() {
  advanceToken();

  if (!curToken->is(TokenKind::lParen)) {
    return nullptr;
  }

  std::vector<std::unique_ptr<LayoutQualifierId>> layoutQualifierIds;

  do {
    advanceToken();

    if (curToken->is(TokenKind::Identifier)) {
      auto id = curToken->getIdentifierName();
      

      advanceToken();

      if (curToken->is(TokenKind::assign)) {
        
        advanceToken();

        auto exp = parseExpression();

        layoutQualifierIds.push_back(std::make_unique<LayoutQualifierId>(id, std::move(exp)));
      } else {
        layoutQualifierIds.push_back(std::make_unique<LayoutQualifierId>(id));
      }
    } else if (curToken->is(TokenKind::kw_shared)) {
      layoutQualifierIds.push_back(std::make_unique<LayoutQualifierId>(/*isShared*/ true));
      advanceToken();
    } else {
      // Unexpected Token
      return nullptr;
    }
  } while (curToken->is(TokenKind::comma));

  if (!curToken->is(TokenKind::rParen)) {
    return nullptr;
  }
  
  return std::make_unique<LayoutQualifier>(std::move(layoutQualifierIds));
}

std::vector<std::unique_ptr<TypeQualifier>> Parser::parseQualifiers() {
  std::vector<std::unique_ptr<TypeQualifier>> qualifiers;
 
  while (true) {
    auto qualifier = parseQualifier();

    if (!qualifier) {
      break;
    }

    qualifiers.push_back(std::move(qualifier));
    advanceToken();
  }

  return qualifiers;
}

std::unique_ptr<Type> Parser::parseType() {
  auto qualifiers = parseQualifiers();
  auto type = getTypeFromTokenKind(std::move(qualifiers),
                                           curToken->getTokenKind());
  return type;
}

std::optional<ast::AssignmentOperator>
Parser::getAssignmentOperatorFromTokenKind(TokenKind kind) {
  switch (kind) {
  case TokenKind::assign:
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

std::unique_ptr<shaderpulse::Type> Parser::getTypeFromTokenKind(
    std::vector<std::unique_ptr<TypeQualifier>> qualifiers, TokenKind kind) {
  switch (kind) {
  case TokenKind::Identifier: {
    if (structDeclarations.find(curToken->getIdentifierName()) != structDeclarations.end()) {
      return std::make_unique<StructType>(std::move(qualifiers), curToken->getIdentifierName());
    }

    return nullptr;
  }
  // Sclar types
  case TokenKind::kw_int:
    return std::make_unique<shaderpulse::Type>(TypeKind::Integer,
                                               std::move(qualifiers));
  case TokenKind::kw_bool:
    return std::make_unique<shaderpulse::Type>(TypeKind::Bool,
                                               std::move(qualifiers));
  case TokenKind::kw_uint:
    return std::make_unique<shaderpulse::Type>(TypeKind::UnsignedInteger,
                                               std::move(qualifiers));
  case TokenKind::kw_float:
    return std::make_unique<shaderpulse::Type>(TypeKind::Float,
                                               std::move(qualifiers));
  case TokenKind::kw_double:
    return std::make_unique<shaderpulse::Type>(TypeKind::Double,
                                               std::move(qualifiers));
  case TokenKind::kw_void:
    return std::make_unique<shaderpulse::Type>(TypeKind::Void,
                                               std::move(qualifiers));

  // Vector types
  case TokenKind::kw_vec2:
    return makeVectorType(std::move(qualifiers), TypeKind::Float, 2);
  case TokenKind::kw_vec3:
    return makeVectorType(std::move(qualifiers), TypeKind::Float, 3);
  case TokenKind::kw_vec4:
    return makeVectorType(std::move(qualifiers), TypeKind::Float, 4);

  case TokenKind::kw_bvec2:
    return makeVectorType(std::move(qualifiers), TypeKind::Bool, 2);
  case TokenKind::kw_bvec3:
    return makeVectorType(std::move(qualifiers), TypeKind::Bool, 3);
  case TokenKind::kw_bvec4:
    return makeVectorType(std::move(qualifiers), TypeKind::Bool, 4);

  case TokenKind::kw_ivec2:
    return makeVectorType(std::move(qualifiers), TypeKind::Integer, 2);
  case TokenKind::kw_ivec3:
    return makeVectorType(std::move(qualifiers), TypeKind::Integer, 3);
  case TokenKind::kw_ivec4:
    return makeVectorType(std::move(qualifiers), TypeKind::Integer, 4);

  case TokenKind::kw_uvec2:
    return makeVectorType(std::move(qualifiers), TypeKind::UnsignedInteger, 2);
  case TokenKind::kw_uvec3:
    return makeVectorType(std::move(qualifiers), TypeKind::UnsignedInteger, 3);
  case TokenKind::kw_uvec4:
    return makeVectorType(std::move(qualifiers), TypeKind::UnsignedInteger, 4);

  case TokenKind::kw_dvec2:
    return makeVectorType(std::move(qualifiers), TypeKind::Double, 2);
  case TokenKind::kw_dvec3:
    return makeVectorType(std::move(qualifiers), TypeKind::Double, 3);
  case TokenKind::kw_dvec4:
    return makeVectorType(std::move(qualifiers), TypeKind::Double, 4);

  // Matrix types
  case TokenKind::kw_mat2:
  case TokenKind::kw_mat2x2:
    return makeMatrixType(std::move(qualifiers), TypeKind::Float, 2, 2);

  case TokenKind::kw_mat3:
  case TokenKind::kw_mat3x3:
    return makeMatrixType(std::move(qualifiers), TypeKind::Float, 3, 3);

  case TokenKind::kw_mat4:
  case TokenKind::kw_mat4x4:
    return makeMatrixType(std::move(qualifiers), TypeKind::Float, 4, 4);

  case TokenKind::kw_dmat2:
  case TokenKind::kw_dmat2x2:
    return makeMatrixType(std::move(qualifiers), TypeKind::Double, 2, 2);

  case TokenKind::kw_dmat3:
  case TokenKind::kw_dmat3x3:
    return makeMatrixType(std::move(qualifiers), TypeKind::Double, 3, 3);

  case TokenKind::kw_dmat4:
  case TokenKind::kw_dmat4x4:
    return makeMatrixType(std::move(qualifiers), TypeKind::Double, 4, 4);

  case TokenKind::kw_mat2x3:
    return makeMatrixType(std::move(qualifiers), TypeKind::Float, 2, 3);

  case TokenKind::kw_mat2x4:
    return makeMatrixType(std::move(qualifiers), TypeKind::Float, 2, 4);

  case TokenKind::kw_mat3x2:
    return makeMatrixType(std::move(qualifiers), TypeKind::Float, 3, 2);

  case TokenKind::kw_mat3x4:
    return makeMatrixType(std::move(qualifiers), TypeKind::Float, 3, 4);

  case TokenKind::kw_mat4x2:
    return makeMatrixType(std::move(qualifiers), TypeKind::Float, 4, 2);

  case TokenKind::kw_mat4x3:
    return makeMatrixType(std::move(qualifiers), TypeKind::Float, 4, 3);

  case TokenKind::kw_dmat2x3:
    return makeMatrixType(std::move(qualifiers), TypeKind::Double, 2, 3);

  case TokenKind::kw_dmat2x4:
    return makeMatrixType(std::move(qualifiers), TypeKind::Double, 2, 4);

  case TokenKind::kw_dmat3x2:
    return makeMatrixType(std::move(qualifiers), TypeKind::Double, 3, 2);

  case TokenKind::kw_dmat3x4:
    return makeMatrixType(std::move(qualifiers), TypeKind::Double, 3, 4);

  case TokenKind::kw_dmat4x2:
    return makeMatrixType(std::move(qualifiers), TypeKind::Double, 4, 2);

  case TokenKind::kw_dmat4x3:
    return makeMatrixType(std::move(qualifiers), TypeKind::Double, 4, 3);

  default:
    // Ignore
    break;
  }

  if (kind >= TokenKind::kw_sampler1D && kind <= TokenKind::kw_uimageBuffer) {
    return std::make_unique<shaderpulse::Type>(TypeKind::Opaque,
                                               std::move(qualifiers));
  }

  return nullptr;
};

std::unique_ptr<shaderpulse::VectorType>
Parser::makeVectorType(std::vector<std::unique_ptr<TypeQualifier>> qualifiers,
                       TypeKind kind, int length) {
  return std::make_unique<shaderpulse::VectorType>(
      std::move(qualifiers),
      std::make_unique<shaderpulse::Type>(
          kind, std::vector<std::unique_ptr<TypeQualifier>>()),
      length);
}

std::unique_ptr<shaderpulse::MatrixType>
Parser::makeMatrixType(std::vector<std::unique_ptr<TypeQualifier>> qualifiers,
                       TypeKind kind, int rows, int cols) {
  return std::make_unique<shaderpulse::MatrixType>(
      std::move(qualifiers),
      std::make_unique<shaderpulse::Type>(
          kind, std::vector<std::unique_ptr<TypeQualifier>>()),
      rows, cols);
}

std::vector<std::unique_ptr<ParameterDeclaration>>
Parser::parseFunctionParameters() {
  std::vector<std::unique_ptr<ParameterDeclaration>> params;

  do {
    advanceToken();

    if (auto type = parseType()) {
      advanceToken();

      if (curToken->is(TokenKind::Identifier)) {
        auto param = std::make_unique<ParameterDeclaration>(
            curToken->getIdentifierName(), std::move(type));
        advanceToken();
        params.push_back(std::move(param));
      }
    } else {
      // Error: expected type name
      return {};
    }

  } while (curToken->is(TokenKind::comma));

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
    std::cout << "Expected a ')' after condition expression." << cursor
              << std::endl;
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
      std::cout << "Expected a ')' after condition expression." << cursor
                << std::endl;
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
    return nullptr;
  }

  if (curToken->is(TokenKind::kw_else)) {
    advanceToken();

    auto falsePart = parseStatement();

    if (!falsePart) {
      return nullptr;
    }

    return std::make_unique<IfStatement>(std::move(exp), std::move(truePart),
                                         std::move(falsePart));

  } else {
    return std::make_unique<IfStatement>(std::move(exp), std::move(truePart),
                                         nullptr);
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
  } else if (auto structDecl = parseStructDeclaration()) {
    return std::move(structDecl);
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
  } else if (auto assignment = parseAssignmentExpression()) {
    return std::move(assignment);
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
    advanceToken();
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
  parsingLhsExpression = true;
  savePosition();
  auto unaryExpression = parseUnaryExpression();

  advanceToken();
  if (!curToken->is(TokenKind::assign)) {
    rollbackPosition();
    parsingLhsExpression = false;
    return nullptr;
  }

  parsingLhsExpression = false;


  if (!unaryExpression) {
    return nullptr;
  }

  auto op = Parser::getAssignmentOperatorFromTokenKind(curToken->getTokenKind());

  advanceToken();
  auto exp = parseExpression();

  if (!curToken->is(TokenKind::semiColon)) {
    std::cout << "Expected a semicolon after assignment expression." << cursor
              << std::endl;
  }

  advanceToken();

  return std::make_unique<AssignmentExpression>(std::move(unaryExpression), *op, std::move(exp));
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

std::optional<std::vector<std::unique_ptr<Expression>>> Parser::parseMemberAccessChain() {
  if (peek(1)->is(TokenKind::dot)) {
    advanceToken();
    std::vector<std::unique_ptr<Expression>> members;

    do {
      advanceToken();

      if (auto member = parsePostfixExpression(/*parsingMemberAccess*/ true)) {
        members.push_back(std::move(member));
        advanceToken();
      }
    } while (curToken->is(TokenKind::dot));

    cursor = cursor - 1;
    return std::move(members);
  }

  return std::nullopt;
}

std::unique_ptr<Expression> Parser::parsePrimaryExpression() {
   if (auto constructorExp = parseConstructorExpression()) {
    return constructorExp;
  } else if (auto callExp = parseCallExpression()) {
    return callExp;
  } else if (curToken->is(TokenKind::Identifier)) {
    
    return std::make_unique<VariableExpression>(curToken->getIdentifierName(), parsingLhsExpression);
  } else if (curToken->is(TokenKind::IntegerConstant)) {
    auto int_const = dynamic_cast<IntegerLiteral *>(curToken->getLiteralData());

    return std::make_unique<IntegerConstantExpression>(int_const->getVal());
  } else if (curToken->is(TokenKind::UnsignedIntegerConstant)) {
    auto uint_const =
        dynamic_cast<UnsignedIntegerLiteral *>(curToken->getLiteralData());

    return std::make_unique<UnsignedIntegerConstantExpression>(
        uint_const->getVal());
  } else if (curToken->is(TokenKind::FloatConstant)) {
    auto float_const = dynamic_cast<FloatLiteral *>(curToken->getLiteralData());

    return std::make_unique<FloatConstantExpression>(float_const->getVal());
  } else if (curToken->is(TokenKind::DoubleConstant)) {
    auto double_const =
        dynamic_cast<DoubleLiteral *>(curToken->getLiteralData());

    return std::make_unique<DoubleConstantExpression>(double_const->getVal());
  } else if (curToken->is(TokenKind::kw_true) ||
             curToken->is(TokenKind::kw_false)) {
    return std::make_unique<BoolConstantExpression>(
        curToken->is(TokenKind::kw_true));
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
  } else if (auto unop = Parser::getUnaryOperatorFromTokenKind(
                 curToken->getTokenKind())) {
    advanceToken();

    return std::make_unique<UnaryExpression>(*unop, parseUnaryExpression());
  } else {
    return parsePostfixExpression();
  }
}

std::unique_ptr<Expression> Parser::parsePostfixExpression(bool parsingMemberAccess) {
  if (auto primary = parsePrimaryExpression()) {
    if (!parsingMemberAccess) {
      if (auto members = parseMemberAccessChain()) {
        return std::make_unique<MemberAccessExpression>(std::move(primary), std::move(*members), parsingLhsExpression);
      } else {
        return primary;
      }
    } else {
      return primary;
    }
  }

  return nullptr;
}

std::unique_ptr<ConstructorExpression> Parser::parseConstructorExpression() {
  if (!(getTypeFromTokenKind(std::vector<std::unique_ptr<TypeQualifier>>(), curToken->getTokenKind()) &&
        peek(1)->is(TokenKind::lParen))) {
    return nullptr;
  }

  auto type = getTypeFromTokenKind(std::vector<std::unique_ptr<TypeQualifier>>(), curToken->getTokenKind());

  advanceToken();
  advanceToken(); // eat lparen

  if (curToken->is(TokenKind::rParen)) {
    return std::make_unique<ConstructorExpression>(std::move(type));
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
  } while (true);

  if (!curToken->is(TokenKind::rParen)) {
    std::cout << "Expected a ')' after parameter list in constructor." << std::endl;
    return nullptr;
  }

  return std::make_unique<ConstructorExpression>(std::move(type), std::move(arguments));
}

std::unique_ptr<StructDeclaration> Parser::parseStructDeclaration() {
  if (!curToken->is(TokenKind::kw_struct)) {
    return nullptr;
  }

  advanceToken();

  if (!curToken->is(TokenKind::Identifier)) {
    std::cout << "Expect an identifier after struct keyword" << std::endl;
    return nullptr;
  }

  auto structName = curToken->getIdentifierName();

  advanceToken();

  if (!curToken->is(TokenKind::lCurly)) {
    std::cout << "Expected '{' after struct name." << std::endl;
    return nullptr;
  }

  advanceToken();

  std::vector<std::unique_ptr<Declaration>> memberDeclarations;

  while (auto memberDecl = parseDeclaration()) {
    memberDeclarations.push_back(std::move(memberDecl));
  }

  if (!curToken->is(TokenKind::rCurly)) {
    std::cout << "Expected a '}' after struct member declaration.";
    return nullptr;
  }

  advanceToken();
  advanceToken();

  if (structDeclarations.find(structName) == structDeclarations.end()) {
    structDeclarations.insert({structName, true});
  }

  return std::make_unique<StructDeclaration>(nullptr, structName, std::move(memberDeclarations));
}

std::unique_ptr<CallExpression> Parser::parseCallExpression() {
  if (!(curToken->is(TokenKind::Identifier) &&
        peek(1)->is(TokenKind::lParen))) {
    return nullptr;
  }

  const std::string &name = curToken->getIdentifierName();

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
  } while (true);

  if (!curToken->is(TokenKind::rParen)) {
    std::cout << "Expected a ')' after parameter list." << std::endl;
    return nullptr;
  }

  return std::make_unique<CallExpression>(name, std::move(arguments));
}

void Parser::advanceToken() {
  if ((cursor > -1) && ((size_t)cursor >= (tokenStream.size() - 1))) {
    auto tok = std::make_unique<Token>();
    tok->setTokenKind(TokenKind::Eof);
    tokenStream.push_back(std::move(tok));
    curToken = tokenStream.back().get();
  } else {
    curToken = tokenStream[++cursor].get();
    std::cout << "Cur token is at line: " << curToken->getSourceLocation().line << ", col: " << curToken->getSourceLocation().col << std::endl;
  }
}

const Token *Parser::peek(int k) {
  if ((size_t)(cursor + k) >= tokenStream.size()) {
    auto tok = std::make_unique<Token>();
    tok->setTokenKind(TokenKind::Eof);
    tokenStream.push_back(std::move(tok));
    return tokenStream[tokenStream.size() - 1].get();
  } else {
    return tokenStream[cursor + k].get();
  }
}

}; // namespace parser

}; // namespace shaderpulse
