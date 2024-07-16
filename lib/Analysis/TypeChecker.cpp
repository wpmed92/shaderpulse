#include "Analysis/TypeChecker.h"
#include <iostream>
#include "../../utils/include/magic_enum.hpp"

namespace shaderpulse {

using namespace ast;

namespace analysis {

void TypeChecker::visit(TranslationUnit *unit) {
    std::cout << "Typechecking translation unit" << std::endl;
    for (auto &extDecl : unit->getExternalDeclarations()) {
        extDecl->accept(this);
        std::cout << "Accepting..." << std::endl;
    }
}

void TypeChecker::visit(VariableDeclarationList *varDeclList) {
}

void TypeChecker::visit(VariableDeclaration *varDecl) {

    std::cout << "Typechecking vardecl" << std::endl;
    auto entry = SymbolTableEntry();
    entry.type = varDecl->getType();
    scopeManager.putSymbol(varDecl->getIdentifierName(), entry);
    std::cout << "Entry added to current scope's symbol table: " << varDecl->getIdentifierName() << std::endl;
    auto testFound = scopeManager.findSymbol(varDecl->getIdentifierName());
    if (testFound) {
        std::cout << "Entry properly saved and found in symbol table" << std::endl;
    }
}

void TypeChecker::visit(SwitchStatement *switchStmt) {

}

void TypeChecker::visit(WhileStatement *whileStmt) {

}

void TypeChecker::visit(DoStatement *doStmt) {

}

void TypeChecker::visit(IfStatement *ifStmt) {

}

void TypeChecker::visit(AssignmentExpression *assignmentExp) {

}

void TypeChecker::visit(StatementList *stmtList) { 

}

void TypeChecker::visit(ForStatement *forStmt) {

}

void TypeChecker::visit(UnaryExpression *unExp) {

}

void TypeChecker::visit(BinaryExpression *binExp) {

}

void TypeChecker::visit(ConditionalExpression *condExp) {
  // TODO: ConditionalExpression
}

void TypeChecker::visit(CallExpression *callee) {

}

void TypeChecker::visit(ConstructorExpression *constExp) {

}

void TypeChecker::visit(InitializerExpression *initExp) {

}

void TypeChecker::visit(VariableExpression *varExp) {

}

void TypeChecker::visit(IntegerConstantExpression *intExp) { 

}

void TypeChecker::visit(StructDeclaration *structDecl) {

}

void TypeChecker::visit(UnsignedIntegerConstantExpression *uintExp) {

}

void TypeChecker::visit(FloatConstantExpression *floatExp) {

}

void TypeChecker::visit(DoubleConstantExpression *doubleExp) {

}

void TypeChecker::visit(BoolConstantExpression *boolExp) {

}

void TypeChecker::visit(MemberAccessExpression *memberExp) {

}

void TypeChecker::visit(ArrayAccessExpression *arrayAccess) {

}

void TypeChecker::visit(ReturnStatement *returnStmt) {

}

void TypeChecker::visit(BreakStatement *breakStmt) {

}

void TypeChecker::visit(ContinueStatement *continueStmt) {

}

void TypeChecker::visit(DiscardStatement *discardStmt) {

}

void TypeChecker::visit(FunctionDeclaration *funcDecl) {

}

void TypeChecker::visit(DefaultLabel *defaultLabel) {

}

void TypeChecker::visit(CaseLabel *caseLabel) {

}

} // namespace analysis
} // namespace shaderpulse
