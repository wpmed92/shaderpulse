#include "CodeGen/MLIRCodeGen.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"

static std::string functionDeclarationTestString = 
R"(
    void myFunc() {
    }
)";

using namespace shaderpulse;
using namespace shaderpulse::ast;
using namespace shaderpulse::lexer;
using namespace shaderpulse::parser;

int main(int argc, char** argv) {
    auto lexer = Lexer(functionDeclarationTestString);
    auto resp = lexer.lexCharacterStream();

    if (!resp.has_value()) {
        return 0;
    }
    
    auto &tokens = (*resp).get();
    auto parser = Parser(tokens);
    auto translationUnit = parser.parseTranslationUnit();
    auto mlirCodeGen = std::make_unique<codegen::MLIRCodeGen>();
    
    translationUnit->accept(mlirCodeGen.get());
}