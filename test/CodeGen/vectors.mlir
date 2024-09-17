spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @main() "None" {
    %cst_f32 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_0 = spirv.Constant 0.000000e+00 : f32
    %0 = spirv.CompositeConstruct %cst_f32, %cst_f32_0 : (f32, f32) -> vector<2xf32>
    %1 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %1, %0 : vector<2xf32>
    %cst_f32_1 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_2 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_3 = spirv.Constant 1.000000e+00 : f32
    %2 = spirv.CompositeConstruct %cst_f32_1, %cst_f32_2, %cst_f32_3 : (f32, f32, f32) -> vector<3xf32>
    %3 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %3, %2 : vector<3xf32>
    %4 = spirv.Load "Function" %1 : vector<2xf32>
    %cst_f32_4 = spirv.Constant 1.000000e+00 : f32
    %5 = spirv.CompositeConstruct %4, %cst_f32_4 : (vector<2xf32>, f32) -> vector<3xf32>
    %6 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %6, %5 : vector<3xf32>
    %cst_f32_5 = spirv.Constant 1.000000e+00 : f32
    %7 = spirv.Load "Function" %1 : vector<2xf32>
    %8 = spirv.CompositeConstruct %cst_f32_5, %7 : (f32, vector<2xf32>) -> vector<3xf32>
    %9 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %9, %8 : vector<3xf32>
    %cst_f32_6 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_7 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_8 = spirv.Constant 1.000000e+00 : f32
    %10 = spirv.CompositeConstruct %cst_f32_6, %cst_f32_7, %cst_f32_8 : (f32, f32, f32) -> vector<3xf32>
    %11 = spirv.CompositeConstruct %10 : (vector<3xf32>) -> vector<3xf32>
    %12 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %12, %11 : vector<3xf32>
    %13 = spirv.Load "Function" %1 : vector<2xf32>
    %14 = spirv.Load "Function" %1 : vector<2xf32>
    %15 = spirv.CompositeConstruct %13, %14 : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
    %16 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %16, %15 : vector<4xf32>
    %17 = spirv.Load "Function" %3 : vector<3xf32>
    %cst_f32_9 = spirv.Constant 1.000000e+00 : f32
    %18 = spirv.CompositeConstruct %17, %cst_f32_9 : (vector<3xf32>, f32) -> vector<4xf32>
    %19 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %19, %18 : vector<4xf32>
    %cst_f32_10 = spirv.Constant 1.000000e+00 : f32
    %20 = spirv.Load "Function" %3 : vector<3xf32>
    %21 = spirv.CompositeConstruct %cst_f32_10, %20 : (f32, vector<3xf32>) -> vector<4xf32>
    %22 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %22, %21 : vector<4xf32>
    %cst_f32_11 = spirv.Constant 1.000000e+00 : f32
    %23 = spirv.Load "Function" %1 : vector<2xf32>
    %cst_f32_12 = spirv.Constant 1.000000e+00 : f32
    %24 = spirv.CompositeConstruct %cst_f32_11, %23, %cst_f32_12 : (f32, vector<2xf32>, f32) -> vector<4xf32>
    %25 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %25, %24 : vector<4xf32>
    %cst_f32_13 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_14 = spirv.Constant 1.000000e+00 : f32
    %26 = spirv.Load "Function" %1 : vector<2xf32>
    %27 = spirv.CompositeConstruct %cst_f32_13, %cst_f32_14, %26 : (f32, f32, vector<2xf32>) -> vector<4xf32>
    %28 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %28, %27 : vector<4xf32>
    %29 = spirv.Load "Function" %1 : vector<2xf32>
    %cst_f32_15 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_16 = spirv.Constant 1.000000e+00 : f32
    %30 = spirv.CompositeConstruct %29, %cst_f32_15, %cst_f32_16 : (vector<2xf32>, f32, f32) -> vector<4xf32>
    %31 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %31, %30 : vector<4xf32>
    %cst_f32_17 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_18 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_19 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_20 = spirv.Constant 1.000000e+00 : f32
    %32 = spirv.CompositeConstruct %cst_f32_17, %cst_f32_18, %cst_f32_19, %cst_f32_20 : (f32, f32, f32, f32) -> vector<4xf32>
    %33 = spirv.CompositeConstruct %32 : (vector<4xf32>) -> vector<4xf32>
    %34 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %34, %33 : vector<4xf32>
    %cst_f32_21 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_22 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_23 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_24 = spirv.Constant 1.000000e+00 : f32
    %35 = spirv.CompositeConstruct %cst_f32_21, %cst_f32_22, %cst_f32_23, %cst_f32_24 : (f32, f32, f32, f32) -> vector<4xf32>
    %36 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %36, %35 : vector<4xf32>
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
