#pragma once
#include "AST/ASTVisitor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include <vector>

namespace shaderpulse {

using namespace mlir;
using namespace ast;

namespace codegen {

class MLIRCodeGen : public ASTVisitor {

public:
    MLIRCodeGen() : builder(&context) {
        context.getOrLoadDialect<spirv::SPIRVDialect>();
        initModuleOp();
    }

    void initModuleOp() {
        OperationState state(UnknownLoc::get(&context),
                            spirv::ModuleOp::getOperationName());
        state.addAttribute("addressing_model",
                        builder.getAttr<spirv::AddressingModelAttr>(
                            spirv::AddressingModel::Logical));
        state.addAttribute("memory_model", builder.getAttr<spirv::MemoryModelAttr>(
                                            spirv::MemoryModel::GLSL450));
        state.addAttribute("vce_triple",
                        spirv::VerCapExtAttr::get(
                            spirv::Version::V_1_0, llvm::ArrayRef<spirv::Capability>(),
                            llvm::ArrayRef<spirv::Extension>(), &context));
        spirv::ModuleOp::build(builder, state);
        spirvModule = cast<spirv::ModuleOp>(Operation::create(state));
    }

    void dump();
    void visit(TranslationUnit*) override;
    void visit(BinaryExpression*) override;;
    void visit(UnaryExpression*) override;;
    void visit(ValueDeclaration*) override;
    void visit(SwitchStatement*) override;
    void visit(WhileStatement*) override;
    void visit(DoStatement*) override;
    void visit(IfStatement*) override;
    void visit(AssignmentExpression*) override;
    void visit(StatementList*) override;
    void visit(CallExpression*) override;
    void visit(VariableExpression*) override;
    void visit(IntegerConstantExpression*) override;
    void visit(UnsignedIntegerConstantExpression*) override;
    void visit(FloatConstantExpression*) override;
    void visit(DoubleConstantExpression*) override;
    void visit(BoolConstantExpression*) override;
    void visit(ReturnStatement*) override;
    void visit(BreakStatement*) override;
    void visit(ContinueStatement*) override;
    void visit(DiscardStatement*) override;
    void visit(FunctionDeclaration*) override;
    void visit(DefaultLabel*) override;
    void visit(CaseLabel*) override;

private:
/// A "module" matches a Toy source file: containing a list of functions.
  MLIRContext context;
  spirv::ModuleOp spirvModule;
  OpBuilder builder;

  // Stack used to hold intermediary values while generating code for an expression
  std::vector<Value> expressionStack;

  llvm::StringMap<FunctionDeclaration*> functionMap;
};

};

};