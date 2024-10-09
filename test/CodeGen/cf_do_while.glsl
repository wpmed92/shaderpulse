void main() {
    bool a = true;
    int b;

    // Header jumps unconditionally to loop body

    // CHECK: spirv.mlir.loop {
    // CHECK-NEXT: spirv.Branch ^bb1
    // CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb3
    // CHECK-NEXT: spirv.Branch ^bb2
    do {
        int c = 2;
        int d = 3;
        b = c + d;

    // The condition check happens in the continue block

    // CHECK: ^bb3:  // pred: ^bb2
    // CHECK-NEXT: %false_1 = spirv.Constant false
    // CHECK-NEXT: spirv.Store "Function" %3, %false_1 : i1
    // CHECK-NEXT: %10 = spirv.Load "Function" %0 : i1
    // CHECK-NEXT: spirv.BranchConditional %10, ^bb1, ^bb4
    // CHECK-NEXT: ^bb4:  // pred: ^bb3
    // CHECK-NEXT: spirv.mlir.merge
    // CHECK-NEXT: }
    } while(a);

    int someVarAfter = 12;
}
