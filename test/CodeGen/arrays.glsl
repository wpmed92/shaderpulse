void main() {
    // CHECK: %0 = spirv.CompositeConstruct %cst_f32, %cst_f32_0, %cst_f32_1 : (f32, f32, f32) -> !spirv.array<3 x f32>
    float[3] myArray = float[3](0.1, 0.2, 0.3);

    // CHECK: %cst0_si32 = spirv.Constant 0 : si32
    // CHECK-NEXT: %2 = spirv.AccessChain %1[%cst0_si32] : !spirv.ptr<!spirv.array<3 x f32>, Function>, si32
    float a = myArray[0];

    // CHECK: %cst1_si32 = spirv.Constant 1 : si32
    // CHECK-NEXT: %5 = spirv.AccessChain %1[%cst1_si32] : !spirv.ptr<!spirv.array<3 x f32>, Function>, si32
    float b = myArray[1];

    // CHECK: %cst2_si32 = spirv.Constant 2 : si32
    // CHECK-NEXT: %8 = spirv.AccessChain %1[%cst2_si32] : !spirv.ptr<!spirv.array<3 x f32>, Function>, si32
    float c = myArray[2];

    // CHECK: %cst0_si32_2 = spirv.Constant 0 : si32
    // CHECK-NEXT: %11 = spirv.AccessChain %1[%cst0_si32_2] : !spirv.ptr<!spirv.array<3 x f32>, Function>, si32
    // CHECK-NEXT: %cst_f32_3 = spirv.Constant 0.00999999977 : f32
    // CHECK-NEXT: spirv.Store "Function" %11, %cst_f32_3 : f32
    myArray[0] = 0.01;

    // CHECK: %cst1_si32_4 = spirv.Constant 1 : si32
    // CHECK-NEXT: %12 = spirv.Variable : !spirv.ptr<si32, Function>
    // CHECK-NEXT: spirv.Store "Function" %12, %cst1_si32_4 : si32
    int varIdx = 1;

    // CHECK: %13 = spirv.Load "Function" %12 : si32
    // CHECK-NEXT: %14 = spirv.AccessChain %1[%13] : !spirv.ptr<!spirv.array<3 x f32>, Function>, si32
    // CHECK-NEXT: %cst_f32_5 = spirv.Constant 2.000000e-02 : f32
    // CHECK-NEXT: spirv.Store "Function" %14, %cst_f32_5 : f32
    myArray[varIdx] = 0.02;

    return;
}
