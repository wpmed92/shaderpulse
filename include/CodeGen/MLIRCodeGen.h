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
#include <functional>
#include <unordered_map>
#include <filesystem>

using namespace mlir;

namespace shaderpulse {

using namespace ast;

namespace codegen {

struct SymbolTableEntry {
  mlir::Value value;
  VariableDeclaration* variable = nullptr;
  spirv::PointerType ptrType = nullptr;
  bool isGlobal = false;
  bool isFunctionParam = false;
  Type* type = nullptr;
};

class MLIRCodeGen : public ASTVisitor {

public:
  MLIRCodeGen();
  void initModuleOp();
  void print();
  bool saveToFile(const std::filesystem::path& outputPath);
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
  void visit(InterfaceBlock *) override;
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

  mlir::MLIRContext context;
  spirv::ModuleOp spirvModule;
  mlir::OpBuilder builder;

  bool inGlobalScope = true;
  llvm::StringMap<spirv::FuncOp> functionMap;
  llvm::StringMap<StructDeclaration*> structDeclarations;
  std::vector<mlir::Value> expressionStack;
  StructDeclaration* currentBaseComposite = nullptr;
  mlir::Operation *execModeOp = nullptr;

  llvm::ScopedHashTable<llvm::StringRef, SymbolTableEntry>
      symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<llvm::StringRef, SymbolTableEntry>;
  using BuiltInFunc = std::function<mlir::Value(mlir::MLIRContext &, mlir::OpBuilder &, mlir::ValueRange)>;

  std::unordered_map<std::string, BuiltInFunc> builtInFuncMap;
  SmallVector<Attribute, 4> interface;

  void declare(StringRef name, SymbolTableEntry entry);
  void createVariable(shaderpulse::TypeQualifierList *,shaderpulse::Type *, VariableDeclaration *);
  void insertEntryPoint();
  void initBuiltinFuncMap();
  bool callBuiltIn(CallExpression* exp);
  void createBuiltinComputeVar(const std::string &varName, const std::string &mlirName);
  mlir::Value load(mlir::Value);
  mlir::Value popExpressionStack();
  mlir::Value currentBasePointer;
  mlir::Value convertOp(ConstructorExpression* constructorExp, mlir::Value val);
};

}; // namespace codegen

}; // namespace shaderpulse
