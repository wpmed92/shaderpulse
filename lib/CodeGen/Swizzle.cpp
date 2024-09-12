#include "CodeGen/Swizzle.h"
#include <vector>
#include <iostream>

namespace shaderpulse {

namespace codegen {

std::unordered_map<char, int> swizzleMap = {
      {'x', 0},
      {'y', 1},
      {'z', 2},
      {'w', 3},
      {'r', 0},
      {'g', 1},
      {'b', 2},
      {'a', 3}
};

mlir::Value swizzle(mlir::OpBuilder &builder, mlir::Value composite, ast::MemberAccessExpression* memberAccess, int startIndex) {
    mlir::Value currentComposite = composite;

    for (int i = startIndex; i < memberAccess->getMembers().size(); i++) {
        auto &member = memberAccess->getMembers()[i];
        if (auto var = dynamic_cast<ast::VariableExpression*>(member.get())) {
            std::vector<int> indices;
            auto swizzle = var->getName();
            std::cerr << "Swizzle: " << swizzle << std::endl;

            if (swizzle.length() == 1) {
                indices.push_back(swizzleMap.find(swizzle[0])->second);
                return builder.create<mlir::spirv::CompositeExtractOp>(builder.getUnknownLoc(), currentComposite, indices);
            } else {
                for (auto c : swizzle) {
                    indices.push_back(swizzleMap.find(c)->second);
                }

                int64_t len = static_cast<int64_t>(swizzle.length());
                std::cerr << "Swizz length " << len << std::endl;
                llvm::ArrayRef<int64_t> shape(&len, 1);
                mlir::Type elementType = currentComposite.getType().dyn_cast<mlir::VectorType>().getElementType();
                mlir::VectorType shuffleType = mlir::VectorType::get(shape, elementType);
                currentComposite = builder.create<mlir::spirv::VectorShuffleOp>(builder.getUnknownLoc(), shuffleType, currentComposite, currentComposite, builder.getI32ArrayAttr(indices));
            }
        }
    }

    return currentComposite;
}

};

};
