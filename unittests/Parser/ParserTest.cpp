#include <gtest/gtest.h>
#include <iostream>
#include "Parser/Parser.h"
#include "Lexer/Lexer.h"

using namespace shaderpulse;
using namespace shaderpulse::lexer;
using namespace shaderpulse::ast;
using namespace shaderpulse::parser;

static std::string functionDeclarationTestString = 
R"(
    void myFunc() {
    }
)";

static std::string switchTestString = 
R"(
    switch (a) {
        case 1:
            break;
        case 2:
            break;
        default:
            break;
    }
)";

static std::string ifTestString =
R"(
    if (a)
        float b;
)";

static std::string ifStatementListTestString =
R"(
    if (a) {
        float b;
        float c;
    }
)";

static std::string ifElseTestString =
R"(
    if (a == 0)
        float b;
    else
        float c;
)";

static std::string ifElseChainTestString =
R"(
    if (a) {
        float b;
    } else if (c) {
        float d;
    } else if (e) {
        float f;
    }
)";


static std::string whileTestString =
R"(
    while (a < b) 
        float b;
)";

static std::string whileStatementListTestString =
R"(
    while (a < b) {
        float c;
        int d;
    }
)";

static std::string doTestString =
R"(
    do float a; while (b < c);
)";

static std::string doStatementListTestString =
R"(
    do {
        float a;
        int b;
    } while (c < d);
)";

static inline std::string wrapInFuncDecl(const std::string& code) {
    return "void myFunc() {" + code + "}";
}

static StatementList* getTestFunctionBody(TranslationUnit* unit) {
    auto& externalDecl = unit->getExternalDeclarations();
    auto funcDecl = dynamic_cast<FunctionDeclaration*>(externalDecl.at(0).get());
    auto statementList = dynamic_cast<StatementList*>(funcDecl->getBody());

    return statementList;
}

static std::unique_ptr<TranslationUnit> parse(const std::string& code) {
    auto lexer = Lexer(code);
    auto resp = lexer.lexCharacterStream();

    if (!resp.has_value()) {
        return nullptr;
    }
    
    auto &tokens = (*resp).get();
    auto parser = Parser(tokens);
    return std::move(parser.parseTranslationUnit());
}

TEST(ParserTest, ParseFunctionDeclaration) {
    auto unit = parse(functionDeclarationTestString);

    EXPECT_TRUE(unit);

    auto& externalDecl = unit->getExternalDeclarations();
    auto funcDecl = dynamic_cast<FunctionDeclaration*>(externalDecl.at(0).get());
    EXPECT_TRUE(funcDecl);
    EXPECT_EQ(funcDecl->getName(), "myFunc");
    EXPECT_EQ(funcDecl->getReturnType()->getKind(), TypeKind::Void);
}


TEST(ParserTest, ParseSwitchStatement) {
    auto unit = parse(wrapInFuncDecl(switchTestString));

    EXPECT_TRUE(unit);

    unit->accept(std::make_unique<PrinterASTVisitor>().get());

    const auto& stmts = getTestFunctionBody(unit.get())->getStatements();

    EXPECT_EQ(stmts.size(), 1);

    auto switchStmt = dynamic_cast<SwitchStatement*>(stmts.at(0).get());
    EXPECT_TRUE(switchStmt);

    auto switchLabel = dynamic_cast<VariableExpression*>(switchStmt->getExpression());
    EXPECT_TRUE(switchLabel);

    auto switchBody = dynamic_cast<StatementList*>(switchStmt->getBody());
    EXPECT_TRUE(switchBody);
    EXPECT_EQ(switchBody->getStatements().size(), 6);
}

TEST(ParserTest, ParseIfStatement) {
    auto unit = parse(wrapInFuncDecl(ifTestString));

    EXPECT_TRUE(unit);
    
    const auto& stmts = getTestFunctionBody(unit.get())->getStatements();

    EXPECT_EQ(stmts.size(), 1);

    auto ifStmt = dynamic_cast<IfStatement*>(stmts.at(0).get());
    EXPECT_TRUE(ifStmt);

    auto condition = dynamic_cast<VariableExpression*>(ifStmt->getCondition());
    EXPECT_TRUE(condition);

    auto valueDeclaration = dynamic_cast<ValueDeclaration*>(ifStmt->getTruePart());
    EXPECT_TRUE(valueDeclaration);
    EXPECT_FALSE(ifStmt->getFalsePart());
}

TEST(ParserTest, ParseIfStatementList) {
    auto unit = parse(wrapInFuncDecl(ifStatementListTestString));

    EXPECT_TRUE(unit);
    
    const auto& stmts = getTestFunctionBody(unit.get())->getStatements();

    EXPECT_EQ(stmts.size(), 1);

    auto ifStmt = dynamic_cast<IfStatement*>(stmts.at(0).get());
    EXPECT_TRUE(ifStmt);

    auto condition = dynamic_cast<VariableExpression*>(ifStmt->getCondition());
    EXPECT_TRUE(condition);

    auto truePart = dynamic_cast<StatementList*>(ifStmt->getTruePart());
    EXPECT_TRUE(truePart);

    EXPECT_EQ(truePart->getStatements().size(), 2);
    EXPECT_FALSE(ifStmt->getFalsePart());
}

TEST(ParserTest, ParseIfElse) {
    auto unit = parse(wrapInFuncDecl(ifElseTestString));

    EXPECT_TRUE(unit);
    const auto& stmts = getTestFunctionBody(unit.get())->getStatements();

    EXPECT_EQ(stmts.size(), 1);

    auto ifStmt = dynamic_cast<IfStatement*>(stmts.at(0).get());
    EXPECT_TRUE(ifStmt);

    auto condition = dynamic_cast<BinaryExpression*>(ifStmt->getCondition());
    EXPECT_TRUE(condition);

    EXPECT_TRUE(ifStmt->getTruePart());
    EXPECT_TRUE(ifStmt->getFalsePart());
}

TEST(ParserTest, ParseIfElseChain) {
    auto unit = parse(wrapInFuncDecl(ifElseChainTestString));

    EXPECT_TRUE(unit);
    const auto& stmts = getTestFunctionBody(unit.get())->getStatements();

    EXPECT_EQ(stmts.size(), 1);

    // First if statement
    auto ifStmt = dynamic_cast<IfStatement*>(stmts.at(0).get());
    EXPECT_TRUE(ifStmt);

    auto condition = dynamic_cast<VariableExpression*>(ifStmt->getCondition());
    EXPECT_TRUE(condition);

    EXPECT_TRUE(ifStmt->getTruePart());
    EXPECT_TRUE(ifStmt->getFalsePart());

    // else if chain
    auto firstElseIfPart = dynamic_cast<IfStatement*>(ifStmt->getFalsePart());
    EXPECT_TRUE(firstElseIfPart);

    auto firstElseIfCondition = dynamic_cast<VariableExpression*>(firstElseIfPart->getCondition());
    EXPECT_TRUE(firstElseIfCondition);

    auto secondElseIfPart = dynamic_cast<IfStatement*>(firstElseIfPart->getFalsePart());
    EXPECT_TRUE(secondElseIfPart);

    auto secondElseICondition = dynamic_cast<VariableExpression*>(secondElseIfPart->getCondition());
    EXPECT_TRUE(secondElseICondition);
}

TEST(ParserTest, ParseWhile) {
    auto unit = parse(wrapInFuncDecl(whileTestString));

    EXPECT_TRUE(unit);
    const auto& stmts = getTestFunctionBody(unit.get())->getStatements();

    EXPECT_EQ(stmts.size(), 1);

    auto whileStmt = dynamic_cast<WhileStatement*>(stmts.at(0).get());
    EXPECT_TRUE(whileStmt);

    auto condition = dynamic_cast<BinaryExpression*>(whileStmt->getCondition());
    EXPECT_TRUE(condition);

    auto body = dynamic_cast<ValueDeclaration*>(whileStmt->getBody());
    EXPECT_TRUE(body);
}

TEST(ParserTest, ParseWhileStatementList) {
    auto unit = parse(wrapInFuncDecl(whileStatementListTestString));

    EXPECT_TRUE(unit);
    const auto& stmts = getTestFunctionBody(unit.get())->getStatements();

    EXPECT_EQ(stmts.size(), 1);

    auto whileStmt = dynamic_cast<WhileStatement*>(stmts.at(0).get());
    EXPECT_TRUE(whileStmt);

    auto condition = dynamic_cast<BinaryExpression*>(whileStmt->getCondition());
    EXPECT_TRUE(condition);

    auto body = dynamic_cast<StatementList*>(whileStmt->getBody());
    EXPECT_TRUE(body);
    EXPECT_EQ(body->getStatements().size(), 2);
}

TEST(ParserTest, ParseDo) {
    auto unit = parse(wrapInFuncDecl(doTestString));

    EXPECT_TRUE(unit);
    const auto& stmts = getTestFunctionBody(unit.get())->getStatements();

    EXPECT_EQ(stmts.size(), 1);

    auto doStmt = dynamic_cast<DoStatement*>(stmts.at(0).get());
    EXPECT_TRUE(doStmt);

    auto condition = dynamic_cast<BinaryExpression*>(doStmt->getCondition());
    EXPECT_TRUE(condition);

    auto body = dynamic_cast<ValueDeclaration*>(doStmt->getBody());
    EXPECT_TRUE(body);
}

TEST(ParserTest, ParseDoStatementList) {
    auto unit = parse(wrapInFuncDecl(doStatementListTestString));

    EXPECT_TRUE(unit);
    const auto& stmts = getTestFunctionBody(unit.get())->getStatements();

    EXPECT_EQ(stmts.size(), 1);

    auto doStmt = dynamic_cast<DoStatement*>(stmts.at(0).get());
    EXPECT_TRUE(doStmt);

    auto condition = dynamic_cast<BinaryExpression*>(doStmt->getCondition());
    EXPECT_TRUE(condition);

    auto body = dynamic_cast<StatementList*>(doStmt->getBody());
    EXPECT_TRUE(body);
    EXPECT_EQ(body->getStatements().size(), 2);
}
