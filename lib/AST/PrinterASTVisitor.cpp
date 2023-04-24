#include "AST/PrinterASTVisitor.h"
#include <iostream>

namespace shaderpulse {

namespace ast {

void PrinterASTVisitor::visit(TranslationUnit *unit) {
  std::cout << "Visiting TranslationUnit" << std::endl;
}

void PrinterASTVisitor::visit(BinaryExpression *binExp) {
  std::cout << "Visiting BinaryExpression" << std::endl;
}

void PrinterASTVisitor::visit(UnaryExpression *unExp) {
  std::cout << "Visiting UnaryExpression" << std::endl;
}

void PrinterASTVisitor::visit(VariableDeclaration *valDecl) {
  std::cout << "Visiting VariableDeclaration" << std::endl;
}

void PrinterASTVisitor::visit(SwitchStatement *switchStmt) {
  std::cout << "Visiting SwitchStatement" << std::endl;
}

void PrinterASTVisitor::visit(WhileStatement *whileStmt) {
  std::cout << "Visiting WhileStatement" << std::endl;
}

void PrinterASTVisitor::visit(DoStatement *doStmt) {
  std::cout << "Visiting DoStatement" << std::endl;
}

void PrinterASTVisitor::visit(IfStatement *ifStmt) {
  std::cout << "Visiting IfStatement" << std::endl;
}

void PrinterASTVisitor::visit(AssignmentExpression *assignmentExp) {
  std::cout << "Visiting AssignmentExpression" << std::endl;
}

void PrinterASTVisitor::visit(StatementList *stmtList) {
  std::cout << "Visiting StatementList" << std::endl;
}

void PrinterASTVisitor::visit(CallExpression *callExp) {
  std::cout << "Visiting CallExpression" << std::endl;
}

void PrinterASTVisitor::visit(VariableExpression *varExp) {
  std::cout << "Visiting VariableExpression" << std::endl;
}

void PrinterASTVisitor::visit(IntegerConstantExpression *intConstExp) {
  std::cout << "Visiting IntegerConstantExpression" << std::endl;
}

void PrinterASTVisitor::visit(UnsignedIntegerConstantExpression *uintConstExp) {
  std::cout << "Visiting UnsignedIntegerConstantExpression" << std::endl;
}

void PrinterASTVisitor::visit(FloatConstantExpression *floatConstExp) {
  std::cout << "Visiting FloatConstantExpression" << std::endl;
}

void PrinterASTVisitor::visit(DoubleConstantExpression *doubleConstExp) {
  std::cout << "Visiting DoubleConstantExpression" << std::endl;
}

void PrinterASTVisitor::visit(BoolConstantExpression *boolConstExp) {
  std::cout << "Visiting BoolConstantExpression" << std::endl;
}

void PrinterASTVisitor::visit(ReturnStatement *returnStmt) {
  std::cout << "Visiting ReturnStatement" << std::endl;
}

void PrinterASTVisitor::visit(BreakStatement *breakStmt) {
  std::cout << "Visiting BreakStatement" << std::endl;
}

void PrinterASTVisitor::visit(ContinueStatement *continueStmt) {
  std::cout << "Visiting ContinueStatement" << std::endl;
}

void PrinterASTVisitor::visit(DiscardStatement *discardStmt) {
  std::cout << "Visiting DiscardStatement" << std::endl;
}

void PrinterASTVisitor::visit(FunctionDeclaration *funcDecl) {
  std::cout << "Visiting FunctionDeclaration" << std::endl;
}

void PrinterASTVisitor::visit(DefaultLabel *defaultLabel) {
  std::cout << "Visiting DefaultLabel" << std::endl;
}

void PrinterASTVisitor::visit(CaseLabel *defaultLabel) {
  std::cout << "Visiting CaseLabel" << std::endl;
}

} // namespace ast
} // namespace shaderpulse
