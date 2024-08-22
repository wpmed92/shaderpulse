void main() {
    // CHECK: %0 = spirv.IAdd %cst1_si32, %cst2_si32 : si3
    int a = 1 + 2;

    // CHECK: %2 = spirv.IAdd %cst1_ui32, %cst2_ui32 : ui32
    uint c = 1u + 2u;

    // CHECK: %4 = spirv.FAdd %cst_f32, %cst_f32_0 : f32
    float b = 1.0f + 2.0f;

    // CHECK: %6 = spirv.FAdd %cst_f64, %cst_f64_1 : f64
    double d = 1.0lf + 2.0lf;

    return;
}
