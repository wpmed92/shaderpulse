#include <gtest/gtest.h>
#include <vector>
#include "Lexer/Lexer.h"
#include "Lexer/Token.h"

using namespace shaderpulse::lexer;

TEST(LexerTest, Literals) {
    static std::string literals = R"(
    0xFF 0100 1234 10e5 1.444 .66 .2e-2 .2e+3 2. 314.e-3 12u 0x122u 010u 0x1u
    )";
    auto lexer = Lexer(literals);
    auto resp = lexer.lexCharacterStream();

    EXPECT_TRUE(resp.has_value());
    
    auto& tokens = (*resp).get();

    EXPECT_EQ(tokens.size(), 14);

    EXPECT_EQ(dynamic_cast<IntegerLiteral*>(tokens.at(0)->getLiteralData())->getVal(), 255);
    EXPECT_EQ(dynamic_cast<IntegerLiteral*>(tokens.at(1)->getLiteralData())->getVal(), 64);
    EXPECT_EQ(dynamic_cast<IntegerLiteral*>(tokens.at(2)->getLiteralData())->getVal(), 1234);
    EXPECT_EQ(dynamic_cast<FloatLiteral*>(tokens.at(3)->getLiteralData())->getVal(), 1000000.0f);
    EXPECT_EQ(dynamic_cast<FloatLiteral*>(tokens.at(4)->getLiteralData())->getVal(), 1.444f);
    EXPECT_EQ(dynamic_cast<FloatLiteral*>(tokens.at(5)->getLiteralData())->getVal(), 0.66f);
    EXPECT_EQ(dynamic_cast<FloatLiteral*>(tokens.at(6)->getLiteralData())->getVal(), 0.002f);
    EXPECT_EQ(dynamic_cast<FloatLiteral*>(tokens.at(7)->getLiteralData())->getVal(), 200.0f);
    EXPECT_EQ(dynamic_cast<FloatLiteral*>(tokens.at(8)->getLiteralData())->getVal(), 2.0f);
    EXPECT_EQ(dynamic_cast<FloatLiteral*>(tokens.at(9)->getLiteralData())->getVal(), 0.314f);
    EXPECT_EQ(dynamic_cast<UnsignedIntegerLiteral*>(tokens.at(10)->getLiteralData())->getVal(), 12u);
    EXPECT_EQ(dynamic_cast<UnsignedIntegerLiteral*>(tokens.at(11)->getLiteralData())->getVal(), 0x122u);
    EXPECT_EQ(dynamic_cast<UnsignedIntegerLiteral*>(tokens.at(12)->getLiteralData())->getVal(), 8u);
    EXPECT_EQ(dynamic_cast<UnsignedIntegerLiteral*>(tokens.at(13)->getLiteralData())->getVal(), 1u);
}

TEST(LexerText, Identifiers) {
    static std::string identifiers = R"(
    vertexBuffer LightIntensity viewPos2 customVec4 __sin projetion_matrix a1b2c3 MYBUFFER light
    )";
    static std::vector<std::string> identifiersList = {
        "vertexBuffer", "LightIntensity", "viewPos2", "customVec4", "__sin", 
        "projetion_matrix", "a1b2c3", "MYBUFFER", "light"
    };

    auto lexer = Lexer(identifiers);
    auto resp = lexer.lexCharacterStream();

    EXPECT_TRUE(resp.has_value());
    
    auto& tokens = (*resp).get();

    EXPECT_EQ(tokens.size(), identifiersList.size());

    for (int i = 0; i < tokens.size(); i++) {
        EXPECT_EQ(tokens[i].get()->getIdentifierName(), identifiersList[i]);
        EXPECT_EQ(tokens[i].get()->getTokenKind(), TokenKind::Identifier);
    }
}

TEST(LexerTest, Punctuators) {
    std::string punctuators = R"(
    ( ) [ ] . ++ -- + - * / % << >> < > <= >= == != & ^ | ~ ! && ^^ || ? : = += -= *= /= %= <<= >>= &= ^= |= , ; { }
    )";

    auto lexer = Lexer(punctuators);
    auto resp = lexer.lexCharacterStream();
    EXPECT_TRUE(resp.has_value());
    std::cout << "Has value" << std::endl;

    auto& tokens = (*resp).get();

    EXPECT_EQ(tokens.size(), 45);

    EXPECT_EQ(tokens.at(0)->getTokenKind(), TokenKind::lParen);
    EXPECT_EQ(tokens.at(1)->getTokenKind(), TokenKind::rParen);
    EXPECT_EQ(tokens.at(2)->getTokenKind(), TokenKind::lBracket);
    EXPECT_EQ(tokens.at(3)->getTokenKind(), TokenKind::rBracket);
    EXPECT_EQ(tokens.at(4)->getTokenKind(), TokenKind::dot);
    EXPECT_EQ(tokens.at(5)->getTokenKind(), TokenKind::increment);
    EXPECT_EQ(tokens.at(6)->getTokenKind(), TokenKind::decrement);
    EXPECT_EQ(tokens.at(7)->getTokenKind(), TokenKind::plus);
    EXPECT_EQ(tokens.at(8)->getTokenKind(), TokenKind::minus);
    EXPECT_EQ(tokens.at(9)->getTokenKind(), TokenKind::mul);
    EXPECT_EQ(tokens.at(10)->getTokenKind(), TokenKind::div);
    EXPECT_EQ(tokens.at(11)->getTokenKind(), TokenKind::modulo);
    EXPECT_EQ(tokens.at(12)->getTokenKind(), TokenKind::shiftLeft);
    EXPECT_EQ(tokens.at(13)->getTokenKind(), TokenKind::shiftRight);
    EXPECT_EQ(tokens.at(14)->getTokenKind(), TokenKind::lt);
    EXPECT_EQ(tokens.at(15)->getTokenKind(), TokenKind::gt);
    EXPECT_EQ(tokens.at(16)->getTokenKind(), TokenKind::ltEq);
    EXPECT_EQ(tokens.at(17)->getTokenKind(), TokenKind::gtEq);
    EXPECT_EQ(tokens.at(18)->getTokenKind(), TokenKind::eq);
    EXPECT_EQ(tokens.at(19)->getTokenKind(), TokenKind::neq);
    EXPECT_EQ(tokens.at(20)->getTokenKind(), TokenKind::band);
    EXPECT_EQ(tokens.at(21)->getTokenKind(), TokenKind::bxor);
    EXPECT_EQ(tokens.at(22)->getTokenKind(), TokenKind::bor);
    EXPECT_EQ(tokens.at(23)->getTokenKind(), TokenKind::bnot);
    EXPECT_EQ(tokens.at(24)->getTokenKind(), TokenKind::lnot);
    EXPECT_EQ(tokens.at(25)->getTokenKind(), TokenKind::land);
    EXPECT_EQ(tokens.at(26)->getTokenKind(), TokenKind::lxor);
    EXPECT_EQ(tokens.at(27)->getTokenKind(), TokenKind::lor);
    EXPECT_EQ(tokens.at(28)->getTokenKind(), TokenKind::question);
    EXPECT_EQ(tokens.at(29)->getTokenKind(), TokenKind::colon);
    EXPECT_EQ(tokens.at(30)->getTokenKind(), TokenKind::assign);
    EXPECT_EQ(tokens.at(31)->getTokenKind(), TokenKind::addAssign);
    EXPECT_EQ(tokens.at(32)->getTokenKind(), TokenKind::subAssign);
    EXPECT_EQ(tokens.at(33)->getTokenKind(), TokenKind::mulAssign);
    EXPECT_EQ(tokens.at(34)->getTokenKind(), TokenKind::divAssign);
    EXPECT_EQ(tokens.at(35)->getTokenKind(), TokenKind::modAssign);
    EXPECT_EQ(tokens.at(36)->getTokenKind(), TokenKind::shiftLeftAssign);
    EXPECT_EQ(tokens.at(37)->getTokenKind(), TokenKind::shiftRightAssign);
    EXPECT_EQ(tokens.at(38)->getTokenKind(), TokenKind::landAssign);
    EXPECT_EQ(tokens.at(39)->getTokenKind(), TokenKind::lxorAssign);
    EXPECT_EQ(tokens.at(40)->getTokenKind(), TokenKind::lorAssign);
    EXPECT_EQ(tokens.at(41)->getTokenKind(), TokenKind::comma);
    EXPECT_EQ(tokens.at(42)->getTokenKind(), TokenKind::semiColon);
    EXPECT_EQ(tokens.at(43)->getTokenKind(), TokenKind::lCurly);
    EXPECT_EQ(tokens.at(44)->getTokenKind(), TokenKind::rCurly);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
