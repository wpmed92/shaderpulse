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

void TypeChecker::visit(FunctionDeclaration *funcDecl) {
    std::cout << "Putting function" << std::endl;
    auto entry = SymbolTableEntry();
    entry.type = funcDecl->getReturnType();
    entry.id = funcDecl->getName();
    entry.isFunction = true;
    entry.isGlobal = true;

    for (auto &param : funcDecl->getParams()) {
        std::cout << "Saving argument type information..." << std::endl;
        entry.argumentTypes.push_back(param.get()->getType());
    }

    std::cout << "Saved function..." << std::endl;
    scopeManager.putSymbol(funcDecl->getName(), entry);

    auto testFound = scopeManager.findSymbol(funcDecl->getName());

    if (testFound) {
        std::cout << "Function properly saved and found in symbol table: " << testFound->id << ", " << testFound->type->getKind() << ", isFunction: " << testFound->isFunction << std::endl;
    }

    // Type check function body
    std::cout << "Accepting function body..." << std::endl;
    funcDecl->getBody()->accept(this);
}

void TypeChecker::visit(VariableDeclarationList *varDeclList) {
    std::cout << "Typechecking vardecl list" << std::endl;
}

void TypeChecker::visit(VariableDeclaration *varDecl) {
    std::cout << "Typechecking vardecl" << std::endl;
    auto entry = SymbolTableEntry();
    entry.type = varDecl->getType();
    entry.id = varDecl->getIdentifierName();
    scopeManager.putSymbol(varDecl->getIdentifierName(), entry);
    std::cout << "Entry added to current scope's symbol table: " << varDecl->getIdentifierName() << std::endl;
    auto testFound = scopeManager.findSymbol(varDecl->getIdentifierName());
    if (testFound) {
        std::cout << "Entry properly saved and found in symbol table: " << testFound->id << ", " << testFound->type->getKind() <<  std::endl;
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
    for (auto &stmt : stmtList->getStatements()) {
        stmt->accept(this);
    }
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

void TypeChecker::visit(DefaultLabel *defaultLabel) {

}

void TypeChecker::visit(CaseLabel *caseLabel) {

}

} // namespace analysis
} // namespace shaderpulse
