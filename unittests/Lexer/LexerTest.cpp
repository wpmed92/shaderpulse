#include <gtest/gtest.h>
#include <vector>
#include "Lexer/Lexer.h"
#include "Lexer/Token.h"

using namespace shaderpulse::lexer;

std::vector<TokenKind> getAllExpectedKeywords() {
    return std::vector<TokenKind>{
        #define KEYWORD(X) TokenKind::kw_##X,
        #include "Lexer/TokenDefs.h"
        #undef KEYWORD
    };
}

std::string getAllKeywordsString() {
    return std::string{
        #define KEYWORD(X) #X " "
        #include "Lexer/TokenDefs.h"
        #undef KEYWORD
    };
}

std::vector<TokenKind> getAllExpectedPunctuators() {
    return std::vector<TokenKind>{
        #define PUNCTUATOR(X, Y) TokenKind::X,
        #include "Lexer/TokenDefs.h"
        #undef PUNCTUATOR
    };
}

std::string getAllPunctuatorsString() {
    std::string punctuators;
    #define PUNCTUATOR(X, Y) punctuators += std::string{Y} + " ";
    #include "Lexer/TokenDefs.h"
    #undef PUNCTUATOR
    return punctuators;
}

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

    for (size_t i = 0; i < tokens.size(); i++) {
        EXPECT_EQ(tokens[i].get()->getIdentifierName(), identifiersList[i]);
        EXPECT_EQ(tokens[i].get()->getTokenKind(), TokenKind::Identifier);
    }
}

TEST(LexerTest, Keywords) {
    std::vector<TokenKind> expectedTokenKinds = getAllExpectedKeywords();
    std::string keywords = getAllKeywordsString();

    auto lexer = Lexer(keywords);
    auto resp = lexer.lexCharacterStream();
    EXPECT_TRUE(resp.has_value());

    auto& tokens = (*resp).get();

    EXPECT_EQ(tokens.size(), expectedTokenKinds.size());

    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i]->getTokenKind(), expectedTokenKinds[i]);
    }
}

TEST(LexerTest, Punctuators) {
    std::string punctuators = getAllPunctuatorsString();
    std::vector<TokenKind> expectedTokenKinds = getAllExpectedPunctuators();

    auto lexer = Lexer(punctuators);
    auto resp = lexer.lexCharacterStream();
    EXPECT_TRUE(resp.has_value());

    auto& tokens = (*resp).get();

    EXPECT_EQ(tokens.size(), expectedTokenKinds.size());

    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i]->getTokenKind(), expectedTokenKinds[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
