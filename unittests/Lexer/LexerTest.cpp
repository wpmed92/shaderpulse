#include <gtest/gtest.h>
#include "Lexer/Lexer.h"
#include "Lexer/Token.h"

static std::string literalTest = R"(
    0xFF 0100 1234 10e5 1.444 .66 .2e-2 .2e+3 2. 314.e-3 12u 0x122u 010u 0x1u
)";

// Demonstrate some basic assertions.
using namespace shaderpulse::lexer;

TEST(LexerTest, Literals) {
    auto lexer = Lexer(literalTest);
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
