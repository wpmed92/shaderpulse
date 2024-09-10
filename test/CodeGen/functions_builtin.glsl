// CHECK:  spirv.func @main() "None" {
void main() {
    float a = sqrt(2.0);

    // CHECK:  %cst_f32_0 = spirv.Constant 2.000000e+00 : f32
    // CHECK-NEXT: %2 = spirv.GL.InverseSqrt %cst_f32_0 : f32
    a = inversesqrt(2.0);

    // CHECK: %cst_f32_1 = spirv.Constant 1.500000e+00 : f32
    // CHECK-NEXT: %3 = spirv.GL.Sin %cst_f32_1 : f32
    a = sin(1.5);

    // CHECK: %cst_f32_2 = spirv.Constant 3.140000e+00 : f32
    // CHECK-NEXT: %4 = spirv.GL.Cos %cst_f32_2 : f32
    a = cos(3.14);

    // CHECK: %cst_f32_3 = spirv.Constant 1.000000e+00 : f32
    // CHECK-NEXT: %5 = spirv.GL.Tan %cst_f32_3 : f32
    a = tan(1.0);
}
