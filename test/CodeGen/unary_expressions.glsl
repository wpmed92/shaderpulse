void main() {
    // CHECK: %0 = spirv.Variable : !spirv.ptr<si32, Function>
    int a = 0;

    // CHECK: %1 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT: %cst1_si32 = spirv.Constant 1 : si32
    // CHECK-NEXT: %2 = spirv.IAdd %1, %cst1_si32 : si32
    // CHECK-NEXT: spirv.Store "Function" %0, %2 : si32
    // CHECK-NEXT: %3 = spirv.Variable : !spirv.ptr<si32, Function>
    // CHECK-NEXT: spirv.Store "Function" %3, %2 : si32
    int b = ++a;
    
    // CHECK: %4 = spirv.Variable : !spirv.ptr<f32, Function>
    float c = 1.0;

    // CHECK: %5 = spirv.Load "Function" %4 : f32
    // CHECK-NEXT: %cst_f32_0 = spirv.Constant 1.000000e+00 : f32
    // CHECK-NEXT: %6 = spirv.FAdd %5, %cst_f32_0 : f32
    // CHECK-NEXT: spirv.Store "Function" %4, %6 : f32
    float d = ++c;

    // CHECK: %cst1_si32_1 = spirv.Constant 1 : si32
    // CHECK-NEXT: %9 = spirv.ISub %8, %cst1_si32_1 : si32
    // CHECK-NEXT: spirv.Store "Function" %0, %9 : si32
    b = --a;

    // CHECK: %cst_f32_2 = spirv.Constant 1.000000e+00 : f32
    // CHECK-NEXT: %11 = spirv.FSub %10, %cst_f32_2 : f32
    // CHECK-NEXT: spirv.Store "Function" %4, %11 : f32
    d = --c;

    // CHECK: %12 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT: %13 = spirv.SNegate %12 : si32
    int e = -a;

    // CHECK: %15 = spirv.Load "Function" %4 : f32
    // CHECK-NEXT: %16 = spirv.FNegate %15 : f32
    c = -c;

    // CHECK: %17 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT: spirv.Store "Function" %14, %17 : si32
    e = +a;

    // CHECK: %18 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT: %19 = spirv.Not %18 : si32
    e = ~a;

    bool f = true;

    // CHECK: %21 = spirv.Load "Function" %20 : i1
    // CHECK-NEXT: %22 = spirv.LogicalNot %21 : i1
    f = !f;
}
