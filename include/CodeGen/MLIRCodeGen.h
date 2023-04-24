#pragma once
#include "AST/ASTVisitor.h"
#include "AST/Types.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include <vector>

using namespace mlir;

namespace shaderpulse {

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
                           spirv::Version::V_1_0,
                           llvm::ArrayRef<spirv::Capability>(),
                           llvm::ArrayRef<spirv::Extension>(), &context));
    spirv::ModuleOp::build(builder, state);
    spirvModule = cast<spirv::ModuleOp>(Operation::create(state));
  }

  void dump();
  bool verify();
  void visit(TranslationUnit *) override;
  void visit(BinaryExpression *) override;
  void visit(UnaryExpression *) override;
  void visit(VariableDeclaration *) override;
  void visit(VariableDeclarationList *) override;
  void visit(SwitchStatement *) override;
  void visit(WhileStatement *) override;
  void visit(DoStatement *) override;
  void visit(IfStatement *) override;
  void visit(AssignmentExpression *) override;
  void visit(StatementList *) override;
  void visit(CallExpression *) override;
  void visit(VariableExpression *) override;
  void visit(IntegerConstantExpression *) override;
  void visit(UnsignedIntegerConstantExpression *) override;
  void visit(FloatConstantExpression *) override;
  void visit(DoubleConstantExpression *) override;
  void visit(BoolConstantExpression *) override;
  void visit(ReturnStatement *) override;
  void visit(BreakStatement *) override;
  void visit(ContinueStatement *) override;
  void visit(DiscardStatement *) override;
  void visit(FunctionDeclaration *) override;
  void visit(DefaultLabel *) override;
  void visit(CaseLabel *) override;

private:
  /// A "module" matches a Toy source file: containing a list of functions.
  MLIRContext context;
  spirv::ModuleOp spirvModule;
  OpBuilder builder;

  // Stack used to hold intermediary values while generating code for an
  // expression
  std::vector<Value> expressionStack;

  llvm::StringMap<spirv::FuncOp> functionMap;
  bool inGlobalScope = true;

  llvm::ScopedHashTable<llvm::StringRef,
                        std::pair<mlir::Value, VariableDeclaration *>>
      symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<StringRef,
                                 std::pair<mlir::Value, VariableDeclaration *>>;

  void declare(VariableDeclaration *, mlir::Value);
  void createVariable(shaderpulse::Type *, VariableDeclaration *);
  mlir::Value popExpressionStack();
};

}; // namespace codegen

}; // namespace shaderpulse
