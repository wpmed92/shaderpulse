#include "Analysis/SymbolTable.h"
#include <iostream>

namespace shaderpulse {

namespace analysis {

SymbolTableEntry* SymbolTable::find(std::string identifier) {
    if(table.find(identifier) != table.end()) {
        return &table[identifier];
    }

    return nullptr;
}

bool SymbolTable::put(std::string identifier, SymbolTableEntry entry) {
    auto result = table.insert(std::make_pair(identifier, entry));
    
    return result.second;
}

}

}