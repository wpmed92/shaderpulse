#include "AST/AST.h"
#include <iostream>

namespace shaderpulse {

namespace ast {

namespace util {

void printExpression(int depth, Expression *exp) {
  if (exp == nullptr) {
    return;
  }

  for (int i = 0; i < depth; i++) {
    std::cout << " ";
  }

  if (auto binExp = dynamic_cast<BinaryExpression *>(exp)) {
    depth++;
    std::cout << "BinOp: " << binExp->getOp() << std::endl;
    printExpression(depth, binExp->getLhs());
    printExpression(depth, binExp->getRhs());
  } else if (auto unExp = dynamic_cast<UnaryExpression *>(exp)) {
    depth++;
    std::cout << "UnOp: " << unExp->getOp() << std::endl;
    printExpression(depth, unExp->getExpression());
  } else if (auto intConst = dynamic_cast<IntegerConstantExpression *>(exp)) {
    std::cout << "Leaf: " << intConst->getVal() << std::endl;
  } else if (auto intConst =
                 dynamic_cast<UnsignedIntegerConstantExpression *>(exp)) {
    std::cout << "Leaf: " << intConst->getVal() << std::endl;
  } else if (auto intConst = dynamic_cast<FloatConstantExpression *>(exp)) {
    std::cout << "Leaf: " << intConst->getVal() << std::endl;
  } else if (auto intConst = dynamic_cast<DoubleConstantExpression *>(exp)) {
    std::cout << "Leaf: " << intConst->getVal() << std::endl;
  } else if (auto intConst = dynamic_cast<BoolConstantExpression *>(exp)) {
    std::cout << "Leaf: " << intConst->getVal() << std::endl;
  } else if (auto var = dynamic_cast<VariableExpression *>(exp)) {
    std::cout << "Leaf: " << var->getName() << std::endl;
  } else if (auto callExp = dynamic_cast<CallExpression *>(exp)) {
    std::cout << "Leaf: " << callExp->getFunctionName() << std::endl;
    depth++;
    for (auto &arg : callExp->getArguments()) {
      printExpression(depth, arg.get());
    }
  }
}

}; // namespace util

}; // namespace ast

}; // namespace shaderpulse