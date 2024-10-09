void main() {
    // Hidden break/continue control vars

    // CHECK: %0 = spirv.Variable : !spirv.ptr<i1, Function>
    // CHECK-NEXT: %1 = spirv.Variable : !spirv.ptr<i1, Function>
    while (true) {
        // CHECK: %cst1_si32 = spirv.Constant 1 : si32
        // CHECK-NEXT: %2 = spirv.Variable : !spirv.ptr<si32, Function>
        // CHECK-NEXT: spirv.Store "Function" %2, %cst1_si32 : si32
        int someVarBefore = 1;

        // CHECK: ^bb1:  // pred: ^bb0
        // CHECK-NEXT: %true_4 = spirv.Constant true
        // CHECK-NEXT: spirv.Store "Function" %1, %true_4 : i1
        if (true) {
            continue;
        // CHECK: ^bb2:  // pred: ^bb0
        // CHECK-NEXT: %true_5 = spirv.Constant true
        // CHECK-NEXT: spirv.Store "Function" %0, %true_5 : i1
        } else {
            break;
        }

        // CHECK: spirv.mlir.merge
        // CHECK-NEXT: }
        // CHECK-NEXT: %3 = spirv.Load "Function" %1 : i1
        // CHECK-NEXT: spirv.BranchConditional %3, ^bb5, ^bb3
        // CHECK-NEXT: ^bb3:  // pred: ^bb2
        // CHECK-NEXT: %4 = spirv.Load "Function" %0 : i1
        // CHECK-NEXT: spirv.BranchConditional %4, ^bb6, ^bb4

        // CHECK: ^bb4:  // pred: ^bb3
        // CHECK-NEXT: %cst1_si32_2 = spirv.Constant 1 : si32
        // CHECK-NEXT: %5 = spirv.Variable : !spirv.ptr<si32, Function>
        int someVarAfter = 1;
    }

    // CHECK: ^bb6:  // 2 preds: ^bb1, ^bb3
    // CHECK-NEXT: spirv.mlir.merge
    // CHECK-NEXT: }
}
