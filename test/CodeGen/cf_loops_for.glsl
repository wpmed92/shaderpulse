void main() {
    // CHECK: %2 = spirv.Variable : !spirv.ptr<si32, Function>
    // CHECK: spirv.mlir.loop {
    // CHECK-NEXT:  spirv.Branch ^bb1
    // CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb3
    // CHECK-NEXT:  %cst10_si32 = spirv.Constant 10 : si32
    // CHECK-NEXT:  %3 = spirv.Load "Function" %2 : si32
    // CHECK-NEXT:  %4 = spirv.SLessThan %3, %cst10_si32 : si32
    // CHECK-NEXT:  spirv.BranchConditional %4, ^bb2, ^bb4
    // CHECK-NEXT:^bb2:  // pred: ^bb1
    // CHECK-NEXT:  %cst1_si32 = spirv.Constant 1 : si32
    // CHECK-NEXT:  %5 = spirv.Load "Function" %2 : si32
    // CHECK-NEXT:  %6 = spirv.IAdd %5, %cst1_si32 : si32
    // CHECK-NEXT:  %7 = spirv.Variable : !spirv.ptr<si32, Function>
    // CHECK-NEXT:  spirv.Store "Function" %7, %6 : si32
    // CHECK-NEXT:  spirv.Branch ^bb3
    // CHECK-NEXT:^bb3:  // pred: ^bb2
    // CHECK-NEXT:  %8 = spirv.Load "Function" %2 : si32
    // CHECK-NEXT:  %cst1_si32_0 = spirv.Constant 1 : si32
    // CHECK-NEXT:  %9 = spirv.IAdd %8, %cst1_si32_0 : si32
    // CHECK-NEXT:  spirv.Store "Function" %2, %9 : si32
    // CHECK-NEXT:  spirv.Branch ^bb1
    // CHECK-NEXT:^bb4:  // pred: ^bb1
    // CHECK-NEXT:  spirv.mlir.merge
    // CHECK-NEXT:}
    for (int i = 0; i < 10; ++i) {
        int a = i + 1;
    }
}