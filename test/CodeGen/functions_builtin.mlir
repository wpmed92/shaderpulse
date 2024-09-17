spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @main() "None" {
    %cst_f32 = spirv.Constant 2.000000e+00 : f32
    %0 = spirv.GL.Sqrt %cst_f32 : f32
    %1 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %1, %0 : f32
    %cst_f32_0 = spirv.Constant 2.000000e+00 : f32
    %2 = spirv.GL.InverseSqrt %cst_f32_0 : f32
    spirv.Store "Function" %1, %2 : f32
    %cst_f32_1 = spirv.Constant 1.500000e+00 : f32
    %3 = spirv.GL.Sin %cst_f32_1 : f32
    spirv.Store "Function" %1, %3 : f32
    %cst_f32_2 = spirv.Constant 3.140000e+00 : f32
    %4 = spirv.GL.Cos %cst_f32_2 : f32
    spirv.Store "Function" %1, %4 : f32
    %cst_f32_3 = spirv.Constant 1.000000e+00 : f32
    %5 = spirv.GL.Tan %cst_f32_3 : f32
    spirv.Store "Function" %1, %5 : f32
    %cst_f32_4 = spirv.Constant 5.000000e-01 : f32
    %6 = spirv.GL.Asin %cst_f32_4 : f32
    spirv.Store "Function" %1, %6 : f32
    %cst_f32_5 = spirv.Constant 0.866024971 : f32
    %7 = spirv.GL.Acos %cst_f32_5 : f32
    spirv.Store "Function" %1, %7 : f32
    %cst_f32_6 = spirv.Constant 7.853980e-01 : f32
    %8 = spirv.GL.Atan %cst_f32_6 : f32
    spirv.Store "Function" %1, %8 : f32
    %cst_f32_7 = spirv.Constant 5.000000e-01 : f32
    %9 = spirv.GL.Exp %cst_f32_7 : f32
    spirv.Store "Function" %1, %9 : f32
    %cst_f32_8 = spirv.Constant 4.605170e+00 : f32
    %10 = spirv.GL.Log %cst_f32_8 : f32
    spirv.Store "Function" %1, %10 : f32
    %cst_f32_9 = spirv.Constant 8.000000e+00 : f32
    %cst_f32_10 = spirv.Constant 2.000000e+00 : f32
    %11 = spirv.GL.Pow %cst_f32_10, %cst_f32_9 : f32
    spirv.Store "Function" %1, %11 : f32
    %cst_f32_11 = spirv.Constant 2.000000e+00 : f32
    %12 = spirv.GL.Sqrt %cst_f32_11 : f32
    spirv.Store "Function" %1, %12 : f32
    %cst_f32_12 = spirv.Constant 8.000000e-01 : f32
    %13 = spirv.GL.FAbs %cst_f32_12 : f32
    spirv.Store "Function" %1, %13 : f32
    %cst1_si32 = spirv.Constant 1 : si32
    %14 = spirv.GL.SAbs %cst1_si32 : si32
    %15 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %15, %14 : si32
    %cst_f32_13 = spirv.Constant 1.500000e+00 : f32
    %16 = spirv.GL.Ceil %cst_f32_13 : f32
    spirv.Store "Function" %1, %16 : f32
    %cst_f32_14 = spirv.Constant 2.700000e+00 : f32
    %17 = spirv.GL.Floor %cst_f32_14 : f32
    spirv.Store "Function" %1, %17 : f32
    %cst_f32_15 = spirv.Constant 1.200000e+00 : f32
    %cst_f32_16 = spirv.Constant 1.000000e-01 : f32
    %cst_f32_17 = spirv.Constant 1.000000e+00 : f32
    %18 = spirv.GL.FClamp %cst_f32_15, %cst_f32_16, %cst_f32_17 : f32
    spirv.Store "Function" %1, %18 : f32
    %cst10_si32 = spirv.Constant 10 : si32
    %cst2_si32 = spirv.Constant 2 : si32
    %cst8_si32 = spirv.Constant 8 : si32
    %19 = spirv.GL.SClamp %cst10_si32, %cst2_si32, %cst8_si32 : si32
    spirv.Store "Function" %15, %19 : si32
    %cst10_ui32 = spirv.Constant 10 : ui32
    %cst2_ui32 = spirv.Constant 2 : ui32
    %cst8_ui32 = spirv.Constant 8 : ui32
    %20 = spirv.GL.UClamp %cst10_ui32, %cst2_ui32, %cst8_ui32 : ui32
    %21 = spirv.Variable : !spirv.ptr<ui32, Function>
    spirv.Store "Function" %21, %20 : ui32
    %cst_f32_18 = spirv.Constant 1.000000e-01 : f32
    %cst_f32_19 = spirv.Constant 1.100000e+00 : f32
    %22 = spirv.GL.FMax %cst_f32_18, %cst_f32_19 : f32
    spirv.Store "Function" %1, %22 : f32
    %cst1_si32_20 = spirv.Constant 1 : si32
    %23 = spirv.SNegate %cst1_si32_20 : si32
    %cst10_si32_21 = spirv.Constant 10 : si32
    %24 = spirv.GL.SMax %23, %cst10_si32_21 : si32
    spirv.Store "Function" %15, %24 : si32
    %cst1_ui32 = spirv.Constant 1 : ui32
    %cst10_ui32_22 = spirv.Constant 10 : ui32
    %25 = spirv.GL.UMax %cst1_ui32, %cst10_ui32_22 : ui32
    spirv.Store "Function" %21, %25 : ui32
    %cst_f32_23 = spirv.Constant 1.000000e-01 : f32
    %cst_f32_24 = spirv.Constant 1.100000e+00 : f32
    %26 = spirv.GL.FMin %cst_f32_23, %cst_f32_24 : f32
    spirv.Store "Function" %1, %26 : f32
    %cst1_si32_25 = spirv.Constant 1 : si32
    %27 = spirv.SNegate %cst1_si32_25 : si32
    %cst10_si32_26 = spirv.Constant 10 : si32
    %28 = spirv.GL.SMin %27, %cst10_si32_26 : si32
    spirv.Store "Function" %15, %28 : si32
    %cst1_ui32_27 = spirv.Constant 1 : ui32
    %cst10_ui32_28 = spirv.Constant 10 : ui32
    %29 = spirv.GL.UMin %cst1_ui32_27, %cst10_ui32_28 : ui32
    spirv.Store "Function" %21, %29 : ui32
    %cst_f32_29 = spirv.Constant 2.100000e+00 : f32
    %cst_f32_30 = spirv.Constant 3.800000e+00 : f32
    %cst_f32_31 = spirv.Constant 1.000000e-01 : f32
    %30 = spirv.GL.FMix %cst_f32_29 : f32, %cst_f32_30 : f32, %cst_f32_31 : f32 -> f32
    spirv.Store "Function" %1, %30 : f32
    %cst_f32_32 = spirv.Constant 1.100000e+00 : f32
    %31 = spirv.FNegate %cst_f32_32 : f32
    %32 = spirv.GL.FSign %31 : f32
    spirv.Store "Function" %1, %32 : f32
    %cst1_si32_33 = spirv.Constant 1 : si32
    %33 = spirv.SNegate %cst1_si32_33 : si32
    %34 = spirv.GL.SSign %33 : si32
    spirv.Store "Function" %15, %34 : si32
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
