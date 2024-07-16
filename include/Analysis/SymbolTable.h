#pragma once
#include "../AST/Types.h"
#include "SymbolTable.h"
#include <map>

namespace shaderpulse {

namespace analysis {

class SymbolTableEntry {

public:
    Type* type;

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