#include "AST/AST.h"
#include "AST/Util.h"
#include "AST/PrinterASTVisitor.h"
#include <iostream>

namespace shaderpulse {

namespace ast {

void PrinterASTVisitor::print(const std::string &text) {
  for (int i = 0; i < indentationLevel; i++) {
    if (i == 0)
      std::cout << "|";

    std::cout << "-";
  }

  std::cout << text << std::endl;
}

void PrinterASTVisitor::indent() {
  prevIndentationLevel = indentationLevel++;
}

void PrinterASTVisitor::resetIndent() {
  indentationLevel = prevIndentationLevel;
}

void PrinterASTVisitor::visit(TranslationUnit *unit) {
  print("|-TranslationUnit");
  indent();

  for (auto &extDecl : unit->getExternalDeclarations()) {
    extDecl->accept(this);
  }

  resetIndent();
}


void PrinterASTVisitor::visit(VariableDeclarationList *varDeclList) {
  print("|-VariableDeclarationList");
  indent();

  for (auto &var : varDeclList->getDeclarations()) {
    var->accept(this);
  }

  resetIndent();
}

void PrinterASTVisitor::visit(VariableDeclaration *valDecl) {
  print("|-VariableDeclaration: name=" + valDecl->getIdentifierName());

  if (auto exp = valDecl->getInitialzerExpression())
    exp->accept(this);
}

void PrinterASTVisitor::visit(SwitchStatement *switchStmt) {
  print("|-SwitchStatement");

  switchStmt->getExpression()->accept(this);
  indent();
  switchStmt->getBody()->accept(this);
}

void PrinterASTVisitor::visit(WhileStatement *whileStmt) {
  print("|-WhileStatement");

  whileStmt->getCondition()->accept(this);
  indent();
  whileStmt->getBody()->accept(this);
  resetIndent();
}

void PrinterASTVisitor::visit(DoStatement *doStmt) {
  print("|-DoStatement");

  doStmt->getCondition()->accept(this);
  indent();
  doStmt->getBody()->accept(this);
  resetIndent();
}

void PrinterASTVisitor::visit(IfStatement *ifStmt) {
  print("|-IfStatement");

  ifStmt->getCondition()->accept(this);
  indent();
  ifStmt->getTruePart()->accept(this);
  resetIndent();

  if (auto falsePart = ifStmt->getFalsePart()) {
    indent();
    falsePart->accept(this);
    resetIndent();
  }
}

void PrinterASTVisitor::visit(AssignmentExpression *assignmentExp) {
  print("|-AssignmentExpression: variable name=" + assignmentExp->getIdentifier());
  indent();
  assignmentExp->getExpression()->accept(this);
  resetIndent();
}

void PrinterASTVisitor::visit(StatementList *stmtList) { 
  print("|-StatementList");

  indent();

  for (auto &stmt : stmtList->getStatements()) {
    stmt->accept(this);
  }

  resetIndent();
}


void PrinterASTVisitor::visit(UnaryExpression *unExp) {
  print("-UnaryExpression: op=" + std::to_string(unExp->getOp()));

  indent();
  unExp->getExpression()->accept(this);
  resetIndent();
}

void PrinterASTVisitor::visit(BinaryExpression *binExp) {
  print("-BinaryExpression: op=" + std::to_string(binExp->getOp()));
  indent();
  binExp->getLhs()->accept(this);
  resetIndent();

  indent();
  binExp->getRhs()->accept(this);
  resetIndent();
}

void PrinterASTVisitor::visit(CallExpression *callee) {
  print("-CallExpression: name=" + callee->getFunctionName());

  indent();

  for (auto &arg : callee->getArguments()) {
    arg->accept(this);
  }

  resetIndent();
}

void PrinterASTVisitor::visit(VariableExpression *varExp) {
  print("-VariableExpression: name=" + varExp->getName());
}

void PrinterASTVisitor::visit(IntegerConstantExpression *intExp) { 
  print("-IntegerConstantExpression: value=" + std::to_string(intExp->getVal()));
}

void PrinterASTVisitor::visit(UnsignedIntegerConstantExpression *uintExp) {
  print("-UnsignedIntegerConstantExpression: value=" + std::to_string(uintExp->getVal()));
}

void PrinterASTVisitor::visit(FloatConstantExpression *floatExp) {
  print("-FloatConstantExpression: value=" + std::to_string(floatExp->getVal()));
}

void PrinterASTVisitor::visit(DoubleConstantExpression *doubleExp) {
  print("-DoubleConstantExpression: value=" + std::to_string(doubleExp->getVal()));
}

void PrinterASTVisitor::visit(BoolConstantExpression *boolExp) {
  print("-BoolConstantExpression: value=" + std::to_string(boolExp->getVal()));
}

void PrinterASTVisitor::visit(ReturnStatement *returnStmt) {
  print("|-ReturnStatement");
  
  if (auto exp = returnStmt->getExpression()) {
    indent();
    exp->accept(this);
    resetIndent();
  }
}

void PrinterASTVisitor::visit(BreakStatement *breakStmt) {
  print("|-BreakStatement");
}

void PrinterASTVisitor::visit(ContinueStatement *continueStmt) {
  print("|-ContinueStatement");
}

void PrinterASTVisitor::visit(DiscardStatement *discardStmt) {
  print("|-DiscardStatement");
}

void PrinterASTVisitor::visit(FunctionDeclaration *funcDecl) {
  print("|-FunctionDeclaration: name=" + funcDecl->getName());
  indent();

  for (auto &arg : funcDecl->getParams()) {
    print("Arg: name=" + arg->getName());
  }

  funcDecl->getBody()->accept(this);
  resetIndent();
}

void PrinterASTVisitor::visit(DefaultLabel *defaultLabel) {
  print("|-DefaultLabel");
}

void PrinterASTVisitor::visit(CaseLabel *caseLabel) {
  print("|-CaseLabel");

  indent();
  caseLabel->getExpression()->accept(this);
  resetIndent();
}

} // namespace ast
} // namespace shaderpulse
