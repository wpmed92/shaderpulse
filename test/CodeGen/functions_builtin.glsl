// CHECK:  spirv.func @main() "None" {
void main() {
    // CHECK: %cst_f32 = spirv.Constant 2.000000e+00 : f32
    // CHECK-NEXT: %0 = spirv.GL.Sqrt %cst_f32 : f32
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

    // CHECK: %cst_f32_4 = spirv.Constant 5.000000e-01 : f32
    // CHECK-NEXT: %6 = spirv.GL.Asin %cst_f32_4 : f32
    a = asin(0.5);

    // CHECK: %cst_f32_5 = spirv.Constant 0.866024971 : f32
    // CHECK-NEXT: %7 = spirv.GL.Acos %cst_f32_5 : f32
    a = acos(0.866025);

    // CHECK: %cst_f32_6 = spirv.Constant 7.853980e-01 : f32
    // CHECK-NEXT: %8 = spirv.GL.Atan %cst_f32_6 : f32
    a = atan(0.785398);

    // CHECK: %cst_f32_7 = spirv.Constant 5.000000e-01 : f32
    // CHECK-NEXT: %9 = spirv.GL.Exp %cst_f32_7 : f32
    a = exp(0.5);

    // CHECK: %cst_f32_8 = spirv.Constant 4.605170e+00 : f32
    // CHECK-NEXT: %10 = spirv.GL.Log %cst_f32_8 : f32
    a = log(4.605170);

    // CHECK: %cst_f32_9 = spirv.Constant 8.000000e+00 : f32
    // CHECK: %cst_f32_10 = spirv.Constant 2.000000e+00 : f32
    // CHECK-NEXT: %11 = spirv.GL.Pow %cst_f32_10, %cst_f32_9 : f32
    a = exp2(8.0);

    // CHECK: %cst_f32_11 = spirv.Constant 2.000000e+00 : f32
    // CHECK-NEXT: %12 = spirv.GL.Sqrt %cst_f32_11 : f32
    a = sqrt(2.0);

    // CHECK: %cst_f32_12 = spirv.Constant 8.000000e-01 : f32
    // CHECK-NEXT: %13 = spirv.GL.FAbs %cst_f32_12 : f32
    a = fabs(0.8);

    // CHECK: %cst_f32_13 = spirv.Constant 1.500000e+00 : f32
    // CHECK-NEXT: %14 = spirv.GL.Ceil %cst_f32_13 : f32
    a = ceil(1.5);

    // CHECK: %cst_f32_14 = spirv.Constant 2.700000e+00 : f32
    // CHECK-NEXT: %15 = spirv.GL.Floor %cst_f32_14 : f32
    a = floor(2.7);
}
