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

    std::cout << "End" << std::endl;
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

    if (varDecl->getInitialzerExpression() != nullptr) {
        varDecl->getInitialzerExpression()->accept(this);
        std::cout << "Found initializer expression" << std::endl;

        if (typeStack.size() > 0) {
          auto typeToAssign = typeStack.back();
          typeStack.pop_back();

          if (!matchTypes(entry.type, typeToAssign)) {
            std::cout << "Cannot assign initializer expression's type to " << entry.id << " variable." << std::endl;
          }
        }
    }

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
  std::cout << "Visiting assignment" << std::endl;
}

void TypeChecker::visit(StatementList *stmtList) { 
    for (auto &stmt : stmtList->getStatements()) {
        stmt->accept(this);
    }
}

void TypeChecker::visit(ForStatement *forStmt) {
  std::cout << "Visiting for stmt" << std::endl;
}

void TypeChecker::visit(UnaryExpression *unExp) {

}

void TypeChecker::visit(BinaryExpression *binExp) {
  std::cout << "Type checking binexp" << std::endl;
  binExp->getLhs()->accept(this);
  binExp->getRhs()->accept(this);

  Type* rhsType = typeStack.back();
  typeStack.pop_back();
  Type* lhsType = typeStack.back();
  typeStack.pop_back();

  if (matchTypes(lhsType, rhsType)) {
    typeStack.push_back(lhsType);
    std::cout << "Binary operation allowed." << std::endl;
  } else {
    std::cout << "Binary operation not supported on the provided types. " << std::endl;
  }
}

bool TypeChecker::matchTypes(Type* a, Type* b) {
  if (a->isScalar() && b->isScalar() && a->getKind() == b->getKind()) {
    return true;
  } else if (a->isVector() && b->isVector()) {
    auto aVec = dynamic_cast<VectorType*>(a);
    auto bVec = dynamic_cast<VectorType*>(b);

    return aVec->getElementType()->getKind() == bVec->getElementType()->getKind() && aVec->getLength() == bVec->getLength();
  } else {
    return false;
  }
}

void TypeChecker::visit(ConditionalExpression *condExp) {

}

void TypeChecker::visit(CallExpression *callee) {
  auto entry = scopeManager.findSymbol(callee->getFunctionName());
  typeStack.push_back(entry->type);
}

void TypeChecker::visit(ConstructorExpression *constExp) {
  std::cout << "Visiting constructor expression" << std::endl;
  typeStack.push_back(constExp->getType());
}

void TypeChecker::visit(InitializerExpression *initExp) {

}

void TypeChecker::visit(VariableExpression *varExp) {
  auto entry = scopeManager.findSymbol(varExp->getName());
  typeStack.push_back(entry->type);
}

void TypeChecker::visit(IntegerConstantExpression *intExp) { 
  typeStack.push_back(intExp->getType());
}

void TypeChecker::visit(StructDeclaration *structDecl) {
  std::cout << "Visiting struct declaration" << std::endl;
}

void TypeChecker::visit(UnsignedIntegerConstantExpression *uintExp) {
  typeStack.push_back(uintExp->getType());
}

void TypeChecker::visit(FloatConstantExpression *floatExp) {
  typeStack.push_back(floatExp->getType());
}

void TypeChecker::visit(DoubleConstantExpression *doubleExp) {
  typeStack.push_back(doubleExp->getType());
}

void TypeChecker::visit(BoolConstantExpression *boolExp) {
  typeStack.push_back(boolExp->getType());
}

void TypeChecker::visit(MemberAccessExpression *memberExp) {
 // Figure out the type of the accessed property, then push to type stack
}

void TypeChecker::visit(ArrayAccessExpression *arrayAccess) {
  arrayAccess->getArray()->accept(this);
}

void TypeChecker::visit(ReturnStatement *returnStmt) {
  std::cout << "Visiting return stmt" << std::endl;
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
