#include "CodeGen/MLIRCodeGen.h"
#include "Analysis/SemanticAnalyzer.h"
#include "AST/PrinterASTVisitor.h"
#include "Lexer/Lexer.h"
#include "Parser/Parser.h"
#include "Preprocessor/Preprocessor.h"
#include <iostream>
#include <fstream>
#include <filesystem>

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

    std::filesystem::path inputPath = argv[1];

    if (!std::filesystem::exists(inputPath)) {
        std::cout << "File " << inputPath << " does not exist." << std::endl;
        return -1;
    }

    std::filesystem::path outputPath = inputPath;
    outputPath.replace_extension(".mlir");

    bool printAST = false;
    bool codeGen = true;
    bool analyze = true;

    for (size_t i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--print-ast") {
            printAST = true;
        } else if (arg == "--no-codegen") {
            codeGen = false;
        } else if (arg == "--no-analyze") {
            analyze = false;
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
        if (analyze) {
            auto analyzer = SemanticAnalyzer();
            translationUnit->accept(&analyzer);
        }

        auto mlirCodeGen = codegen::MLIRCodeGen();
        translationUnit->accept(&mlirCodeGen);
        mlirCodeGen.print();

        if (!mlirCodeGen.verify()) {
            std::cout << "Error verifying the SPIR-V module." << std::endl;
            return -1;
        }

        bool success = mlirCodeGen.saveToFile(outputPath);

        if (!success) {
            std::cout << "Failed to save spirv mlir to file.";
            return -1;
        }

        success = mlirCodeGen.emitSpirv(outputPath.replace_extension(".spv"));

        if (!success) {
            std::cout << "Failed to emit spirv binary.";
            return -1;
        }
    }

    return 0;
}
