#include <gtest/gtest.h>
#include <vector>
#include <stdint.h>
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

TEST(LexerTest, IntegerLiterals) {
    static std::string intLiterals = R"(
        0xFF 0100 1234 12u 0x122u 010u 0x1u
    )";
    std::vector<uint32_t> expectedValues = {
        0xFF, 64, 1234, 12, 0x122, 8, 1
    };
    auto lexer = Lexer(intLiterals);
    auto resp = lexer.lexCharacterStream();

    EXPECT_TRUE(resp.has_value());

    auto& tokens = (*resp).get();

    EXPECT_EQ(tokens.size()-1, expectedValues.size());

    for (size_t i = 0; i < tokens.size()-1; i++) {
        if (i > 2) {
            EXPECT_EQ(tokens[i].get()->getTokenKind(), TokenKind::UnsignedIntegerConstant);
            EXPECT_EQ(dynamic_cast<UnsignedIntegerLiteral*>(tokens.at(i)->getLiteralData())->getVal(), expectedValues[i]);
        } else {
            EXPECT_EQ(tokens[i].get()->getTokenKind(), TokenKind::IntegerConstant);
            EXPECT_EQ(dynamic_cast<IntegerLiteral*>(tokens.at(i)->getLiteralData())->getVal(), expectedValues[i]);
        }
    }
}

TEST(LexerTest, FloatLiterals) {
    std::string floatLiterals = R"(
        10e5 1.444 .66 .2e-2 .2e+3 2. 314.e-3 1.0f 1.f 10.123f .18f 1.0F 1.F 10.123F .18F 
        10e5f .2e-2f 314.e-3f 10e5F .2e-2F 314.e-3F
    )";
    std::vector<float> expectedValues = {
        1000000.0f, 1.444f, 0.66f, 0.002f, 200.0f, 2.0f, 0.314f, 1.0f, 1.0f, 10.123f, 0.18f, 1.0f, 1.0f, 10.123f, 0.18f,
        1000000.0f, 0.002f, 0.314f, 1000000.0f, 0.002f, 0.314f
    };
    auto lexer = Lexer(floatLiterals);
    auto resp = lexer.lexCharacterStream();

    EXPECT_TRUE(resp.has_value());
    
    auto& tokens = (*resp).get();

    EXPECT_EQ(tokens.size()-1, expectedValues.size());

    for (size_t i = 0; i < tokens.size()-1; i++) {
        EXPECT_EQ(tokens[i].get()->getTokenKind(), TokenKind::FloatConstant);
        EXPECT_EQ(dynamic_cast<FloatLiteral*>(tokens.at(i)->getLiteralData())->getVal(), expectedValues[i]);
    }
}

TEST(LexerTest, DoubleLiterals) {
    std::string doubleLiterals = R"(
        1.0lf 1.lf 10.123lf .18lf 1.0LF 1.LF 10.123LF .18LF
        10e5lf .2e-2lf 314.e-3lf 10e5LF .2e-2LF 314.e-3LF
    )";
    std::vector<double> expectedValues = {
        1.0, 1.0, 10.123, 0.18, 1.0, 1.0, 10.123, 0.18,
        1000000.0, 0.002, 0.314, 1000000.0, 0.002, 0.314
    };
    auto lexer = Lexer(doubleLiterals);
    auto resp = lexer.lexCharacterStream();

    EXPECT_TRUE(resp.has_value());

    auto& tokens = (*resp).get();

    EXPECT_EQ(tokens.size()-1, expectedValues.size());

    for (size_t i = 0; i < tokens.size()-1; i++) {
        EXPECT_EQ(tokens[i].get()->getTokenKind(), TokenKind::DoubleConstant);
        EXPECT_EQ(dynamic_cast<DoubleLiteral*>(tokens.at(i)->getLiteralData())->getVal(), expectedValues[i]);
    }
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

    EXPECT_EQ(tokens.size()-1, identifiersList.size());

    for (size_t i = 0; i < tokens.size()-1; i++) {
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

    EXPECT_EQ(tokens.size()-1, expectedTokenKinds.size());

    for (size_t i = 0; i < tokens.size()-1; ++i) {
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

    EXPECT_EQ(tokens.size()-1, expectedTokenKinds.size());

    for (size_t i = 0; i < tokens.size()-1; ++i) {
        EXPECT_EQ(tokens[i]->getTokenKind(), expectedTokenKinds[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
