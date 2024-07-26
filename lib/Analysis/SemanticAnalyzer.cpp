#include "Analysis/SemanticAnalyzer.h"
#include <iostream>
#include "../../utils/include/magic_enum.hpp"

namespace shaderpulse {

using namespace ast;

namespace analysis {

void SemanticAnalyzer::visit(TranslationUnit *unit) {
  for (auto &extDecl : unit->getExternalDeclarations()) {
    extDecl->accept(this);
  }
}

void SemanticAnalyzer::visit(FunctionDeclaration *funcDecl) {
  auto entry = SymbolTableEntry();
  entry.type = funcDecl->getReturnType();
  entry.id = funcDecl->getName();
  entry.isFunction = true;
  entry.isGlobal = true;
  currentFunctionReturnType = entry.type;

  for (auto &param : funcDecl->getParams()) {
    entry.argumentTypes.push_back(param.get()->getType());
  }

  scopeManager.putSymbol(funcDecl->getName(), entry);

  funcDecl->getBody()->accept(this);
}

void SemanticAnalyzer::visit(VariableDeclarationList *varDeclList) {
  std::cout << "Typechecking vardecl list" << std::endl;
}

void SemanticAnalyzer::visit(VariableDeclaration *varDecl) {
  auto entry = SymbolTableEntry();
  entry.type = varDecl->getType();
  entry.id = varDecl->getIdentifierName();
  scopeManager.putSymbol(varDecl->getIdentifierName(), entry);

  if (varDecl->getInitialzerExpression() != nullptr) {
    varDecl->getInitialzerExpression()->accept(this);

    if (typeStack.size() > 0) {
      auto typeToAssign = typeStack.back();
      typeStack.pop_back();

      if (!matchTypes(entry.type, typeToAssign)) {
        std::cout << "Cannot assign initializer expression's type to " << entry.id << " variable." << std::endl;
      }
    }
  }
}

void SemanticAnalyzer::visit(SwitchStatement *switchStmt) {

}

void SemanticAnalyzer::visit(WhileStatement *whileStmt) {
  whileStmt->getCondition()->accept(this);

  Type* condition = typeStack.back();
  typeStack.pop_back();

  if (condition->getKind() != TypeKind::Bool) {
    std::cout << "boolean expression expected in while condition." << std::endl;
  } else {
    std::cout << "while loop condition type correct." << std::endl;
  }


  if (whileStmt->getBody() != nullptr) {
    scopeManager.newScope(ScopeType::Loop);
    whileStmt->getBody()->accept(this);
    scopeManager.exitScope();

    scopeManager.printScopes();
  }
}

// TODO: merge with WhileStatement type check as these are the same
void SemanticAnalyzer::visit(DoStatement *doStmt) {
  doStmt->getCondition()->accept(this);

  Type* condition = typeStack.back();
  typeStack.pop_back();

  if (condition->getKind() != TypeKind::Bool) {
    std::cout << "boolean expression expected in while condition." << std::endl;
  } else {
    std::cout << "while loop condition type correct." << std::endl;
  }

  if (doStmt->getBody() != nullptr) {
    scopeManager.newScope(ScopeType::Loop);
    doStmt->getBody()->accept(this);
    scopeManager.exitScope();
  }
}

void SemanticAnalyzer::visit(IfStatement *ifStmt) {
  ifStmt->getCondition()->accept(this);

  Type* condition = typeStack.back();
  typeStack.pop_back();

  if (condition->getKind() != TypeKind::Bool) {
    std::cout << "boolean expression expected in if condition." << std::endl;
  } else {
    std::cout << "if condition type correct." << std::endl;
  }

  if (ifStmt->getTruePart() != nullptr) {
    scopeManager.newScope(ScopeType::Conditional);
    ifStmt->getTruePart()->accept(this);
    scopeManager.exitScope();
  } else if (ifStmt->getFalsePart() != nullptr) {
    scopeManager.newScope(ScopeType::Conditional);
    ifStmt->getFalsePart()->accept(this);
    scopeManager.exitScope();
  }
}

void SemanticAnalyzer::visit(AssignmentExpression *assignmentExp) {
  assignmentExp->getUnaryExpression()->accept(this);
  assignmentExp->getExpression()->accept(this);

  Type* rhsType = typeStack.back();
  typeStack.pop_back();
  Type* lhsType = typeStack.back();
  typeStack.pop_back();

  if (!matchTypes(lhsType, rhsType)) {
    std::cout << "Cannot assign the provided type to the variable." << std::endl;
  } else {
    std::cout << "Assignment expression OK" << std::endl;
  }
}

void SemanticAnalyzer::visit(StatementList *stmtList) { 
    for (auto &stmt : stmtList->getStatements()) {
        stmt->accept(this);
    }
}

void SemanticAnalyzer::visit(ForStatement *forStmt) {
  // TODO: analyse for statement
}

void SemanticAnalyzer::visit(UnaryExpression *unExp) {

}

void SemanticAnalyzer::visit(BinaryExpression *binExp) {
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
    std::cout << "Binary operation not supported on the provided types." << std::endl;
  }
}

void SemanticAnalyzer::visit(ConditionalExpression *condExp) {

}

void SemanticAnalyzer::visit(CallExpression *callee) {
  auto entry = scopeManager.findSymbol(callee->getFunctionName());
  typeStack.push_back(entry->type);
}

void SemanticAnalyzer::visit(ConstructorExpression *constExp) {
  typeStack.push_back(constExp->getType());
}

void SemanticAnalyzer::visit(InitializerExpression *initExp) {

}

void SemanticAnalyzer::visit(VariableExpression *varExp) {
  auto entry = scopeManager.findSymbol(varExp->getName());
  typeStack.push_back(entry->type);
}

void SemanticAnalyzer::visit(IntegerConstantExpression *intExp) { 
  typeStack.push_back(intExp->getType());
}

void SemanticAnalyzer::visit(StructDeclaration *structDecl) {
  // TODO: handle structs
}

void SemanticAnalyzer::visit(UnsignedIntegerConstantExpression *uintExp) {
  typeStack.push_back(uintExp->getType());
}

void SemanticAnalyzer::visit(FloatConstantExpression *floatExp) {
  typeStack.push_back(floatExp->getType());
}

void SemanticAnalyzer::visit(DoubleConstantExpression *doubleExp) {
  typeStack.push_back(doubleExp->getType());
}

void SemanticAnalyzer::visit(BoolConstantExpression *boolExp) {
  typeStack.push_back(boolExp->getType());
}

void SemanticAnalyzer::visit(MemberAccessExpression *memberExp) {
 // TODO: Figure out the type of the accessed property, then push to type stack
}

void SemanticAnalyzer::visit(ArrayAccessExpression *arrayAccess) {
  arrayAccess->getArray()->accept(this);
}

void SemanticAnalyzer::visit(ReturnStatement *returnStmt) {
  if (returnStmt->getExpression() != nullptr) {
    returnStmt->getExpression()->accept(this);
    Type* expressionType = typeStack.back();
    typeStack.pop_back();

    if (!matchTypes(expressionType, currentFunctionReturnType)) {
      std::cout << "'return' value type does not match the function type." << std::endl;
    } else {
      std::cout << "Return type check OK" << std::endl;
    }
  } else {
    if (currentFunctionReturnType->getKind() != TypeKind::Void) {
      std::cout << "'return' with no value, in function returning non-void." << std::endl;
    }
  }
}

void SemanticAnalyzer::visit(BreakStatement *breakStmt) {
  if (!scopeManager.hasParentScopeOf(ScopeType::Loop) && !scopeManager.hasParentScopeOf(ScopeType::Switch) ) {
    std::cout << "break statement only allowed in loops and switch statements" << std::endl;
  } else {
    std::cout << "break correct." << std::endl;
  }
}

void SemanticAnalyzer::visit(ContinueStatement *continueStmt) {
  if (!scopeManager.hasParentScopeOf(ScopeType::Loop)) {
    std::cout << "continue statement only allowed in loops" << std::endl;
  } else {
    std::cout << "continue correct." << std::endl;
  }
}

void SemanticAnalyzer::visit(DiscardStatement *discardStmt) {

}

void SemanticAnalyzer::visit(DefaultLabel *defaultLabel) {
  // TODO: Check if inside switch
}

void SemanticAnalyzer::visit(CaseLabel *caseLabel) {
  // TODO: Check if inside switch
}

bool SemanticAnalyzer::matchTypes(Type* a, Type* b) {
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

} // namespace analysis
} // namespace shaderpulse
