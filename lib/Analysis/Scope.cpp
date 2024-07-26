#include "Analysis/Scope.h"
#include <iostream>

namespace shaderpulse {

namespace analysis {

Scope::Scope() {
    table = std::make_unique<SymbolTable>();
}

SymbolTable* Scope::getSymbolTable() {
    return table.get();
}

Scope* Scope::getParent() {
    return parent;
}

void Scope::push() {
    auto newScope = std::make_unique<Scope>();
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

// TODO: consistent naming between ScopeManager and Scope
void ScopeManager::newScope() {
    currentScope->push();
    currentScope = currentScope->getChildren().back().get();
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
    if (currentScope != nullptr) {
        return currentScope->getSymbolTable()->find(identifier);
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
