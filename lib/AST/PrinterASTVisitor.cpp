#include "AST/AST.h"
#include "AST/Util.h"
#include "AST/PrinterASTVisitor.h"
#include <iostream>
#include "../../utils/include/magic_enum.hpp"

namespace shaderpulse {

namespace ast {

void PrinterASTVisitor::print(const std::string &text) {
  for (int i = 0; i < indentationLevel; i++) {
    if (levels.find(i) != levels.end())
      std::cout << "|";

    std::cout << " ";
  }

  std::cout << text << std::endl;
}

void PrinterASTVisitor::indent() {
  indentationLevel+=2;
  levels.insert(indentationLevel);
}

void PrinterASTVisitor::resetIndent() {
  indentationLevel-=2;
  levels.erase(std::prev(levels.end()));
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
  print("|-VariableDeclarationList: type=" + std::string(magic_enum::enum_name(varDeclList->getType()->getKind())));
  indent();

  for (auto &var : varDeclList->getDeclarations()) {
    var->accept(this);
  }

  resetIndent();
}

void PrinterASTVisitor::visit(VariableDeclaration *varDecl) {
  auto typeName = varDecl->getType() != nullptr ? std::string(magic_enum::enum_name(varDecl->getType()->getKind())) : "";
  print("|-VariableDeclaration: name=" + varDecl->getIdentifierName() + ((typeName != "") ? (", type=" + typeName) : ""));

  indent();
  if (auto exp = varDecl->getInitialzerExpression()) {
    exp->accept(this);
  }
  resetIndent();
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
  print("|-AssignmentExpression");
  indent();
  assignmentExp->getUnaryExpression()->accept(this);
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

void PrinterASTVisitor::visit(ForStatement *forStmt) {
  print("|-ForStatement");
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

void PrinterASTVisitor::visit(ConditionalExpression *condExp) {
  // TODO: ConditionalExpression
}

void PrinterASTVisitor::visit(CallExpression *callee) {
  print("|-CallExpression: name=" + callee->getFunctionName());

  indent();

  for (auto &arg : callee->getArguments()) {
    arg->accept(this);
  }

  resetIndent();
}

void PrinterASTVisitor::visit(ConstructorExpression *constExp) {
  print("|-ConstructorExpression");

  indent();
  for (auto &exp : constExp->getArguments()) {
    exp->accept(this);
  }
  resetIndent();
}

void PrinterASTVisitor::visit(VariableExpression *varExp) {
  print("|-VariableExpression: name=" + varExp->getName());
}

void PrinterASTVisitor::visit(IntegerConstantExpression *intExp) { 
  print("|-IntegerConstantExpression: value=" + std::to_string(intExp->getVal()));
}

void PrinterASTVisitor::visit(StructDeclaration *structDecl) {
  print("|-StructDeclaration: name=" + structDecl->getName());

  indent();

  for (auto &member : structDecl->getMembers()) {
    member->accept(this);
  }

  resetIndent();
}

void PrinterASTVisitor::visit(UnsignedIntegerConstantExpression *uintExp) {
  print("|-UnsignedIntegerConstantExpression: value=" + std::to_string(uintExp->getVal()));
}

void PrinterASTVisitor::visit(FloatConstantExpression *floatExp) {
  print("|-FloatConstantExpression: value=" + std::to_string(floatExp->getVal()));
}

void PrinterASTVisitor::visit(DoubleConstantExpression *doubleExp) {
  print("|-DoubleConstantExpression: value=" + std::to_string(doubleExp->getVal()));
}

void PrinterASTVisitor::visit(BoolConstantExpression *boolExp) {
  print("|-BoolConstantExpression: value=" + std::to_string(boolExp->getVal()));
}

void PrinterASTVisitor::visit(MemberAccessExpression *memberExp) {
  print("|-MemberAccessExpression");

  indent();
  memberExp->getBaseComposite()->accept(this);

  int savedIndentation = indentationLevel;
  
  for (auto &member : memberExp->getMembers()) {
    indent();
    member->accept(this);
  }
  indentationLevel = savedIndentation;

  resetIndent();
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
  print("|-FunctionDeclaration: name=" + funcDecl->getName() + ", return type=" + std::string(magic_enum::enum_name(funcDecl->getReturnType()->getKind())));
  indent();

  print("|-Args:");
  indent();

  for (auto &arg : funcDecl->getParams()) {
    print("name=" + arg->getName() + ", type=" + std::string(magic_enum::enum_name(arg->getType()->getKind())));
  }

  resetIndent();

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
