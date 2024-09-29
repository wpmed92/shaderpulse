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
        // CHECK-NEXT: %true_2 = spirv.Constant true
        // CHECK-NEXT: spirv.Store "Function" %0, %true_2 : i1
        if (true) {
            break;
        }

        // CHECK: spirv.mlir.merge
        // CHECK-NEXT: }
        // CHECK-NEXT: %3 = spirv.Load "Function" %0 : i1
        // CHECK-NEXT: spirv.BranchConditional %3, ^bb5, ^bb3

        // CHECK: ^bb3:  // pred: ^bb2
        // CHECK-NEXT: %false = spirv.Constant false

        // Reset break control var

        // CHECK-NEXT: spirv.Store "Function" %0, %false : i1
        // CHECK-NEXT: %cst1_si32_1 = spirv.Constant 1 : si32
        // CHECK-NEXT: %4 = spirv.Variable : !spirv.ptr<si32, Function>
        int someVarAfter = 1;
    }

    // CHECK: ^bb5:  // 2 preds: ^bb1, ^bb2
    // CHECK-NEXT:  spirv.mlir.merge
    // CHECK-NEXT: }
}
