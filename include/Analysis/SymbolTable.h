#pragma once
#include "../AST/Types.h"
#include "SymbolTable.h"
#include <map>

namespace shaderpulse {

namespace analysis {

struct SymbolTableEntry {
    std::string id;
    Type* type;
    bool isFunction;
    bool isGlobal;
    std::vector<Type*> argumentTypes;
};

class SymbolTable {

public:
    SymbolTableEntry* find(std::string);
    bool put(std::string, SymbolTableEntry);

private:
    std::map<std::string, SymbolTableEntry> table;
};

}

}