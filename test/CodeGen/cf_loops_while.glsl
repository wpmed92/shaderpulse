void main() {
    bool a = true;
    int b;

    // CHECK:  spirv.mlir.loop {
    // CHECK-NEXT: spirv.Branch ^bb1
    // CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb3
    // CHECK-NEXT: %4 = spirv.Load "Function" %0 : i1
    // CHECK-NEXT: spirv.BranchConditional %4, ^bb2, ^bb4
    // CHECK-NEXT: ^bb2:  // pred: ^bb1
    while (a) {
        int c = 2;
        int d = 3;

        // CHECK: spirv.Store "Function" %1, %9 : si32
        // CHECK-NEXT: spirv.Branch ^bb3
        b = c + d;
    }

    // CHECK-NEXT: ^bb3:  // pred: ^bb2
    // CHECK-NEXT:  %false_1 = spirv.Constant false
    // CHECK-NEXT:  spirv.Store "Function" %3, %false_1 : i1
    // CHECK-NEXT:  spirv.Branch ^bb1
    // CHECK-NEXT: ^bb4:  // pred: ^bb1
    // CHECK-NEXT: spirv.mlir.merge
    // CHECK-NEXT: }
}
