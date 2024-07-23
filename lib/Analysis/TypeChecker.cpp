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

    // This is our current type context
    typeStack.push_back(entry.type);
    typeContext = entry.type;

    if (varDecl->getInitialzerExpression() != nullptr) {
        varDecl->getInitialzerExpression()->accept(this);
        std::cout << "Found initializer expression" << std::endl;
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

/*
When performing implicit conversion for binary operators, there may be multiple data types to which the two operands can be converted. 
For example, when adding an int value to a uint value, both values can be implicitly converted to uint, float, and double.
In such cases, a floating-point type is chosen if either operand has a floating-point type. 
Otherwise, an unsigned integer type is chosen if either operand has an unsigned integer type. 
Otherwise, a signed integer type is chosen. 
-------------------------------------------
int + float -> float
int + uint -> uint

Type of expression Can be implicitly converted to
int     uint
int    
uint   float
int uint float
double
ivec2 uvec2
ivec3 uvec3
ivec4 uvec4
ivec2 uvec2
vec2
ivec3 uvec3
vec3
ivec4 uvec4
vec4
ivec2 uvec2 vec2
dvec2
ivec3 uvec3 vec3
dvec3
ivec4 uvec4 vec4
dvec4
mat2   dmat2
mat3   dmat3
mat4   dmat4
mat2x3 dmat2x3
mat2x4 dmat2x4
mat3x2 dmat3x2
mat3x4 dmat3x4
mat4x2 dmat4x2
mat4x3 dmat4x3
*/
void TypeChecker::visit(BinaryExpression *binExp) {
  std::cout << "Type checking binexp" << std::endl;
  binExp->getLhs()->accept(this);
  binExp->getRhs()->accept(this);

  Type* rhsType = typeStack.back();
  typeStack.pop_back();
  Type* lhsType = typeStack.back();
  typeStack.pop_back();

  if (binopAllowed(lhsType, rhsType)) {
    typeStack.push_back(lhsType);
    std::cout << "Binary operation allowed." << std::endl;
  } else {
    std::cout << "Binary operation not supported on the provided types. " << std::endl;
  }
}

bool TypeChecker::binopAllowed(Type* a, Type* b) {
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
  // TODO: ConditionalExpression
}

void TypeChecker::visit(CallExpression *callee) {

}

void TypeChecker::visit(ConstructorExpression *constExp) {
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
