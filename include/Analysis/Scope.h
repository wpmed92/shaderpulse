#pragma once
#include "../AST/Types.h"
#include "SymbolTable.h"
#include <memory>

namespace shaderpulse {

namespace analysis {

enum ScopeType {
    Global,
    Function,
    Loop,
    Conditional,
    Switch,
    Block
};

class Scope {

public:
    Scope(ScopeType type = ScopeType::Global);
    SymbolTable* getSymbolTable();
    std::vector<std::unique_ptr<Scope>>& getChildren();
    void makeNew(ScopeType type);
    Scope* getParent();
    ScopeType getType();

private:
    std::unique_ptr<SymbolTable> table;
    std::vector<std::unique_ptr<Scope>> children;
    Scope* parent;
    ScopeType type;
};

class ScopeManager {

public:
    ScopeManager();
    void newScope(ScopeType type);
    void enterScope();
    void exitScope();
    void printScopes(Scope* scope = nullptr);
    SymbolTableEntry* findSymbol(std::string identifier);
    bool putSymbol(std::string identifier, SymbolTableEntry entry);
    Scope* getCurrentScope();
    bool hasParentScopeOf(ScopeType type);

private:
    std::unique_ptr<Scope> scopeChain;
    Scope* currentScope;

};

} // namespace analysis

} // namespace shaderpulse
