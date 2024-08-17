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
#include <map>
#include <utility>

using namespace mlir;

namespace shaderpulse {

using namespace ast;

namespace codegen {

struct SymbolTableEntry {
  mlir::Value value;
  VariableDeclaration* variable = nullptr;
  spirv::PointerType ptrType = nullptr;
  bool isGlobal = false;
};

class MLIRCodeGen : public ASTVisitor {

public:
  MLIRCodeGen();
  void initModuleOp();
  void dump();
  bool verify();
  void visit(TranslationUnit *) override;
  void visit(BinaryExpression *) override;
  void visit(UnaryExpression *) override;
  void visit(ConditionalExpression *) override;
  void visit(VariableDeclaration *) override;
  void visit(VariableDeclarationList *) override;
  void visit(SwitchStatement *) override;
  void visit(WhileStatement *) override;
  void visit(ForStatement *) override;
  void visit(DoStatement *) override;
  void visit(IfStatement *) override;
  void visit(AssignmentExpression *) override;
  void visit(StatementList *) override;
  void visit(CallExpression *) override;
  void visit(ConstructorExpression *) override;
  void visit(InitializerExpression *) override;
  void visit(MemberAccessExpression *) override;
  void visit(ArrayAccessExpression *) override;
  void visit(StructDeclaration *) override;
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
  bool insideEntryPoint = false;

  MLIRContext context;
  spirv::ModuleOp spirvModule;
  OpBuilder builder;

  bool inGlobalScope = true;
  llvm::StringMap<spirv::FuncOp> functionMap;
  llvm::StringMap<StructDeclaration*> structDeclarations;
  std::vector<std::pair<Type*, Value>> expressionStack;
  StructDeclaration* currentBaseComposite = nullptr;

  llvm::ScopedHashTable<llvm::StringRef, SymbolTableEntry>
      symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<StringRef, SymbolTableEntry>;

  SymbolTableScopeT globalScope;
  SmallVector<Attribute, 4> interface;

  void declare(SymbolTableEntry);
  void createVariable(shaderpulse::Type *, VariableDeclaration *);
  void insertEntryPoint();
  
  std::pair<Type*, Value> popExpressionStack();
  mlir::Value currentBasePointer;
  Type* typeContext;
};

}; // namespace codegen

}; // namespace shaderpulse
