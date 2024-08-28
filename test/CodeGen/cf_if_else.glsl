void main() {
    bool a = true;
    int b;

    // CHECK: spirv.mlir.selection {
    // CHECK-NEXT:  spirv.BranchConditional %2, ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:  // pred: ^bb0
    if (a) {
        b = 0;
    // CHECK: spirv.Branch ^bb3
    // CHECK-NEXT: ^bb2:  // pred: ^bb0
    } else {
        b = 1;
        // CHECK:  spirv.Branch ^bb3
    }

    // CHECK-NEXT: ^bb3:  // 2 preds: ^bb1, ^bb2
    // CHECK-NEXT:  spirv.mlir.merge
    // CHECK-NEXT: }
}
