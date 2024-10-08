#include "Analysis/SemanticAnalyzer.h"
#include <iostream>

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

  // Save function in global scope
  if (!scopeManager.putSymbol(entry.id, entry)) {
    std::cout << "Redeclaration of symbol '" << entry.id << "'" << std::endl;
  }

  // New scope for function body
  scopeManager.newScope(ScopeType::Function);

  for (auto &param : funcDecl->getParams()) {
    // Save function params in the function scope's symbol table
    auto paramEntry = SymbolTableEntry();
    paramEntry.type = param->getType();
    paramEntry.id = param->getName();
    paramEntry.isFunction = false;
    paramEntry.isGlobal = false;
    scopeManager.putSymbol(param->getName(), paramEntry);
  }

  funcDecl->getBody()->accept(this);

  scopeManager.exitScope();
}

void SemanticAnalyzer::visit(VariableDeclarationList *varDeclList) {
  std::cout << "Typechecking vardecl list" << std::endl;
}

void SemanticAnalyzer::visit(VariableDeclaration *varDecl) {
  auto entry = SymbolTableEntry();
  entry.type = varDecl->getType();
  entry.id = varDecl->getIdentifierName();
  
  if (!scopeManager.putSymbol(varDecl->getIdentifierName(), entry)) {
    std::cout << "Redeclaration of symbol '" << varDecl->getIdentifierName() << "'" << std::endl;
  }

  if (varDecl->getInitialzerExpression() != nullptr) {
    varDecl->getInitialzerExpression()->accept(this);

    if (typeStack.size() > 0) {
      auto typeToAssign = typeStack.back();
      typeStack.pop_back();

      if (!entry.type->isEqual(*typeToAssign)) {
        std::cout << "Cannot convert '" << typeToAssign->toString() << "' to '" << entry.type->toString() << "'." << std::endl;
      }
    }
  }
}

void SemanticAnalyzer::visit(SwitchStatement *switchStmt) {
  switchStmt->getExpression()->accept(this);

  Type* initExpType = typeStack.back();
  typeStack.pop_back();

  if (initExpType->getKind() != TypeKind::Integer) {
    std::cout <<  "init-expression in a switch statement must be a scalar integer" << std::endl;
  }

  if (switchStmt->getBody() != nullptr) {
    scopeManager.newScope(ScopeType::Switch);
    switchStmt->getBody()->accept(this);
    scopeManager.exitScope();
  }
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

  if (!lhsType->isEqual(*rhsType)) {
    std::cout << "Cannot convert '" << rhsType->toString() << "' to '" << lhsType->toString() << "'." << std::endl;
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

  if (lhsType->isEqual(*rhsType)) {
    typeStack.push_back(lhsType);
    std::cout << "Binary operation allowed." << std::endl;
  } else {
    std::cout << "Binary operation not supported on the provided types." << std::endl;
  }
}

void SemanticAnalyzer::visit(ConditionalExpression *condExp) {
  condExp->getCondition()->accept(this);

  Type* conditionType = typeStack.back();
  typeStack.pop_back();

  if (conditionType->getKind() != TypeKind::Bool) {
    std::cout << "Boolean expression expected" << std::endl;
  }

  condExp->getTruePart()->accept(this);
  condExp->getFalsePart()->accept(this);

  Type* falseType = typeStack.back();
  typeStack.pop_back();

  Type* trueType = typeStack.back();
  typeStack.pop_back();

  if (!trueType->isEqual(*falseType)) {
    std::cout << "the types of true and false parts of conditional expression do not match" << std::endl;
  }

  typeStack.push_back(trueType);
}

void SemanticAnalyzer::visit(CallExpression *callee) {
  auto entry = scopeManager.findSymbol(callee->getFunctionName());
  typeStack.push_back(entry->type);

  // Type check call expression parameters
  for (int i = 0; i < callee->getArguments().size(); i++) {
    auto &arg = callee->getArguments()[i];
    Type* argType = entry->argumentTypes[i];
    arg->accept(this);

    Type* argExpressionType = typeStack.back();
    typeStack.pop_back();

    if (!argType->isEqual(*argExpressionType)) {
      std::cout << "Argument type mismatch" << std::endl;
    }
  }
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

    if (!expressionType->isEqual(*currentFunctionReturnType)) {
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
  // TODO: assert in fragment shader
}

void SemanticAnalyzer::visit(DefaultLabel *defaultLabel) {
  if (!scopeManager.hasParentScopeOf(ScopeType::Switch)) {
    std::cout << "default labels need to be inside switch statements" << std::endl;
  } else {
    std::cout << "default label correct" << std::endl;
  }
}

void SemanticAnalyzer::visit(CaseLabel *caseLabel) {
  if (!scopeManager.hasParentScopeOf(ScopeType::Switch)) {
    std::cout << "case labels need to be inside switch statements" << std::endl;
  } else {
    std::cout << "case label correct" << std::endl;
  }

  caseLabel->getExpression()->accept(this);

  Type* caseLabelType = typeStack.back();
  typeStack.pop_back();

  if (caseLabelType->getKind() != TypeKind::Integer) {
    std::cout << "Case label must be a scalar integer" << std::endl;
  }
}

} // namespace analysis
} // namespace shaderpulse
