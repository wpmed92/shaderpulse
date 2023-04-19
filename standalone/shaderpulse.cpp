#include "CodeGen/MLIRCodeGen.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include <iostream>

static std::string functionDeclarationTestString = 
R"( 
    float a;
    int b;
    uint c;
    vec3 d;
    mat2x2 e;

    void myFunc(vec2 arg1, bool arg2) {
        float f;
        float g;
        f = 1.0;
        g = f + 2.0;
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
    mlirCodeGen->dump();
}