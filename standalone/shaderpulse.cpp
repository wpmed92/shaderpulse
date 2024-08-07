#include "CodeGen/MLIRCodeGen.h"
#include "Analysis/SemanticAnalyzer.h"
#include "AST/PrinterASTVisitor.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "Preprocessor/Preprocessor.h"
#include <iostream>
#include <fstream>

static std::string functionDeclarationTestString = 
R"( 
    uniform highp float a;
    uniform int b;
    uint c;
    vec3 d;
    mat2x2 e;

    float foo() {
        return 1.0;
    }

    float myFunc(vec2 arg1, bool arg2) {
        float f;
        float g;
        f = 1.0;
        g = f + 2.0;
        return g + foo();
    }
)";

using namespace shaderpulse;
using namespace shaderpulse::ast;
using namespace shaderpulse::lexer;
using namespace shaderpulse::parser;
using namespace shaderpulse::analysis;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Missing input file." << std::endl;
        return -1;
    }
    
    bool printAST = false;
    bool codeGen = true;

    for (size_t i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--print-ast") {
            printAST = true;
        } else if (arg == "--no-codegen") {
            codeGen = false;
        } else {
            std::cout << "Unrecognized argument: '" << arg << "'." << std::endl;
            return -1;
        }
    }


    std::ifstream glslIn(argv[1]);
    std::stringstream shaderCodeBuffer;
    shaderCodeBuffer << glslIn.rdbuf();
    auto sourceCode = shaderCodeBuffer.str();

    auto preprocessor = preprocessor::Preprocessor(sourceCode);
    preprocessor.process();
    auto processedCode = preprocessor.getProcessedSource();
    std::cout << processedCode;
    auto lexer = Lexer(processedCode);
    auto resp = lexer.lexCharacterStream();
    if (!resp.has_value()) {
        std::cout << "Lexer error " << std::endl;
        return 0;
    }
    
    auto &tokens = (*resp).get();
    auto parser = Parser(tokens);
    auto translationUnit = parser.parseTranslationUnit();

    if (printAST) {
        auto printer = PrinterASTVisitor();
        translationUnit->accept(&printer);
    }

    if (codeGen) {
        auto analyzer =  SemanticAnalyzer();
        translationUnit->accept(&analyzer);
        auto mlirCodeGen = codegen::MLIRCodeGen();
        translationUnit->accept(&mlirCodeGen);
        mlirCodeGen.dump();

        if (mlirCodeGen.verify()) {
            std::cout << "SPIR-V module verified" << std::endl;
        }
    }

    return 0;
}
