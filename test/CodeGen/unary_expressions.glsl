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

    b = --a;
    d = --c;
}