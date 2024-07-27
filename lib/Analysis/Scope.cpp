#include "Analysis/Scope.h"
#include <iostream>

namespace shaderpulse {

namespace analysis {

Scope::Scope(ScopeType type) : table(std::make_unique<SymbolTable>()), type(type) {
}

SymbolTable* Scope::getSymbolTable() {
    return table.get();
}

Scope* Scope::getParent() {
    return parent;
}

ScopeType Scope::getType() {
    return type;
}

void Scope::makeNew(ScopeType type) {
    auto newScope = std::make_unique<Scope>(type);
    newScope->parent = this;
    children.push_back(std::move(newScope));
}

std::vector<std::unique_ptr<Scope>>& Scope::getChildren() {
    return children;
}

ScopeManager::ScopeManager() {
    scopeChain = std::make_unique<Scope>();
    currentScope = scopeChain.get();
}

void ScopeManager::newScope(ScopeType type) {
    currentScope->makeNew(type);
    currentScope = currentScope->getChildren().back().get();
}

Scope* ScopeManager::getCurrentScope() {
    return currentScope;
}

bool ScopeManager::hasParentScopeOf(ScopeType type) {
    Scope* scope = currentScope;

    while (scope->getParent() != nullptr) {
        if (scope->getType() == type) {
            return true;
        }

        scope = scope->getParent();
    }

    return false;
}

void ScopeManager::enterScope() {
    // currentScope = currentScope->children[0];
}

void ScopeManager::exitScope() {
    currentScope = currentScope->getParent();
}

void ScopeManager::printScopes(Scope* scope) {
    if (scope == nullptr) {
        scope = scopeChain.get();
    }

    scope->getSymbolTable()->printEntries();

    for (auto &childScope : scope->getChildren()) {
        printScopes(childScope.get());
    }
}

SymbolTableEntry* ScopeManager::findSymbol(std::string identifier) {
    Scope* scope = currentScope;

    while (scope != nullptr) {
        if (auto entry = scope->getSymbolTable()->find(identifier)) {
            return entry;
        }

        scope = scope->getParent();
    }

    return nullptr;
}

bool ScopeManager::putSymbol(std::string identifier, SymbolTableEntry entry) {
    if (currentScope != nullptr) {
        return currentScope->getSymbolTable()->put(identifier, entry);
    }

    return false;
}

}

}
