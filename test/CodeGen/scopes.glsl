void main() {
    // CHECK: %0 = spirv.Variable : !spirv.ptr<si32, Function>
    int a;

    /*
     * 
     * test new scope for 'if' and 'else' parts
     *
     */

    // CHECK: %1 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT: %2 = spirv.IEqual %1, %cst1_si32 : si32
    if (a == 1) {
        // CHECK: %7 = spirv.Variable : !spirv.ptr<si32, Function>
        int a;

        // CHECK: %cst2_si32_2 = spirv.Constant 2 : si32
        // CHECK-NEXT: spirv.Store "Function" %7, %cst2_si32_2 : si32
        a = 2;
    } else {
        // CHECK: %8 = spirv.Variable : !spirv.ptr<si32, Function>
        int a;

        // CHECK: %cst3_si32 = spirv.Constant 3 : si32
        // CHECK-NEXT: spirv.Store "Function" %8, %cst3_si32 : si32
        a = 3;
    }

    // CHECK: %cst2_si32 = spirv.Constant 2 : si32
    // CHECK-NEXT: spirv.Store "Function" %0, %cst2_si32 : si32
    a = 2;

    /*
     * 
     * test new scope for loop body
     *
     */

    // CHECK: %3 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT: %4 = spirv.IEqual %3, %cst1_si32_0 : si32
    while (a == 1) {
        // CHECK: %7 = spirv.Variable : !spirv.ptr<si32, Function>
        int a;

        // CHECK: %cst5_si32 = spirv.Constant 5 : si32
        // CHECK-NEXT: spirv.Store "Function" %7, %cst5_si32 : si32
        a = 5;
    }

    // CHECK: %cst4_si32 = spirv.Constant 4 : si32
    // CHECK-NEXT: spirv.Store "Function" %0, %cst4_si32 : si32
    a = 4;

    /*
     * 
     * test nested scopes
     *
     */

    // CHECK: %5 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT: %6 = spirv.IEqual %5, %cst1_si32_1 : si32
    if (a == 1) {
        // CHECK: %7 = spirv.Variable : !spirv.ptr<si32, Function>
        // CHECK-NEXT: %cst1_si32_2 = spirv.Constant 1 : si32
        // CHECK-NEXT: spirv.Store "Function" %7, %cst1_si32_2 : si32
        int a;
        a = 1;

        // CHECK: %8 = spirv.Load "Function" %7 : si32
        // CHECK-NEXT: %9 = spirv.IEqual %8, %cst2_si32_3 : si32
        if (a == 2) {
            // CHECK: %10 = spirv.Variable : !spirv.ptr<si32, Function>
            // CHECK-NEXT: %cst2_si32_4 = spirv.Constant 2 : si32
            // CHECK-NEXT: spirv.Store "Function" %10, %cst2_si32_4 : si32
            int a;
            a = 2;
        }
    }
}
