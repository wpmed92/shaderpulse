void main() {
    // CHECK: %cst0_si32 = spirv.Constant 0 : si32
    // CHECK-NEXT: %0 = spirv.Variable : !spirv.ptr<si32, Function>
    // CHECK-NEXT: spirv.Store "Function" %0, %cst0_si32 : si32
    // CHECK-NEXT: spirv.mlir.loop {
    // CHECK-NEXT:   spirv.Branch ^bb1
    // CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb2
    // CHECK-NEXT:  %cst10_si32 = spirv.Constant 10 : si32
    // CHECK-NEXT:  %2 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT:  %3 = spirv.SLessThan %2, %cst10_si32 : si32
    // CHECK-NEXT:  spirv.BranchConditional %3, ^bb2, ^bb3
    // CHECK-NEXT: ^bb2:  // pred: ^bb1
    // CHECK-NEXT:  %cst1_si32 = spirv.Constant 1 : si32
    // CHECK-NEXT:  %4 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT:  %5 = spirv.IAdd %4, %cst1_si32 : si32
    // CHECK-NEXT:  %6 = spirv.Variable : !spirv.ptr<si32, Function>
    // CHECK-NEXT:  spirv.Store "Function" %6, %5 : si32
    // CHECK-NEXT:  %7 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT:  %cst1_si32_1 = spirv.Constant 1 : si32
    // CHECK-NEXT:  %8 = spirv.IAdd %7, %cst1_si32_1 : si32
    // CHECK-NEXT:  spirv.Store "Function" %0, %8 : si32
    // CHECK-NEXT:  spirv.Branch ^bb1
    // CHECK-NEXT: ^bb3:  // pred: ^bb1
    // CHECK-NEXT:  spirv.mlir.merge
    // CHECK-NEXT: }
    for (int i = 0; i < 10; ++i) {
        int a = i + 1;
    }

    // TODO: file check embedded loops
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 20; ++j) {
            int a = i + j;
        }
    }
}