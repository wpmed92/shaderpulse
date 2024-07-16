#pragma once
#include "../AST/Types.h"
#include "SymbolTable.h"
#include <memory>

namespace shaderpulse {

namespace analysis {

class Scope {

public:
    Scope();
    SymbolTable* getSymbolTable();
    std::vector<std::unique_ptr<Scope>>& getChildren();
    void push();
    Scope* getParent();

private:
    std::unique_ptr<SymbolTable> table;
    std::vector<std::unique_ptr<Scope>> children;
    Scope* parent;
};

class ScopeManager {

public:
    ScopeManager();
    void newScope();
    void enterScope();
    void exitScope();
    SymbolTableEntry* findSymbol(std::string identifier);
    bool putSymbol(std::string identifier, SymbolTableEntry entry);

private:
    std::unique_ptr<Scope> scopeChain;
    Scope* currentScope;

};

} // namespace analysis

} // namespace shaderpulse

