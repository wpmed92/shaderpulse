void main() {
    bool a = true;
    int b;

    // CHECK:  spirv.mlir.loop {
    // CHECK-NEXT: spirv.Branch ^bb1
    // CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb2
    // CHECK-NEXT:  spirv.BranchConditional %2, ^bb2, ^bb3
    // CHECK-NEXT:  ^bb2:  // pred: ^bb1
    while (a) {
        int c = 2;
        int d = 3;

        // CHECK: spirv.Store "Function" %1, %7 : si32
        // CHECK-NEXT: spirv.Branch ^bb1
        b = c + d;
    }

    // CHECK: ^bb3:  // pred: ^bb1
    // CHECK-NEXT:  spirv.mlir.merge
    // CHECK-NEXT: }
}
