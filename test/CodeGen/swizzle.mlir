spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @main() "None" {
    %cst_f32 = spirv.Constant 1.000000e-01 : f32
    %cst_f32_0 = spirv.Constant 2.000000e-01 : f32
    %0 = spirv.CompositeConstruct %cst_f32, %cst_f32_0 : (f32, f32) -> vector<2xf32>
    %1 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %1, %0 : vector<2xf32>
    %cst_f32_1 = spirv.Constant 1.100000e+00 : f32
    %cst_f32_2 = spirv.Constant 1.200000e+00 : f32
    %cst_f32_3 = spirv.Constant 1.300000e+00 : f32
    %2 = spirv.CompositeConstruct %cst_f32_1, %cst_f32_2, %cst_f32_3 : (f32, f32, f32) -> vector<3xf32>
    %3 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %3, %2 : vector<3xf32>
    %cst_f32_4 = spirv.Constant 3.400000e+00 : f32
    %cst_f32_5 = spirv.Constant 4.800000e+00 : f32
    %cst_f32_6 = spirv.Constant 5.600000e+00 : f32
    %cst_f32_7 = spirv.Constant 6.6999998 : f32
    %4 = spirv.CompositeConstruct %cst_f32_4, %cst_f32_5, %cst_f32_6, %cst_f32_7 : (f32, f32, f32, f32) -> vector<4xf32>
    %5 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %5, %4 : vector<4xf32>
    %6 = spirv.Load "Function" %1 : vector<2xf32>
    %7 = spirv.VectorShuffle [0 : i32, 1 : i32] %6 : vector<2xf32>, %6 : vector<2xf32> -> vector<2xf32>
    %8 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %8, %7 : vector<2xf32>
    %9 = spirv.Load "Function" %3 : vector<3xf32>
    %10 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32] %9 : vector<3xf32>, %9 : vector<3xf32> -> vector<3xf32>
    %11 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %11, %10 : vector<3xf32>
    %12 = spirv.Load "Function" %5 : vector<4xf32>
    %13 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32] %12 : vector<4xf32>, %12 : vector<4xf32> -> vector<4xf32>
    %14 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %14, %13 : vector<4xf32>
    %15 = spirv.Load "Function" %5 : vector<4xf32>
    %16 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32] %15 : vector<4xf32>, %15 : vector<4xf32> -> vector<4xf32>
    %17 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32] %16 : vector<4xf32>, %16 : vector<4xf32> -> vector<3xf32>
    %18 = spirv.VectorShuffle [0 : i32, 1 : i32] %17 : vector<3xf32>, %17 : vector<3xf32> -> vector<2xf32>
    %19 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %19, %18 : vector<2xf32>
    %20 = spirv.Load "Function" %3 : vector<3xf32>
    %21 = spirv.CompositeExtract %20[0 : i32] : vector<3xf32>
    %22 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %22, %21 : f32
    %23 = spirv.Load "Function" %3 : vector<3xf32>
    %24 = spirv.CompositeExtract %23[1 : i32] : vector<3xf32>
    spirv.Store "Function" %22, %24 : f32
    %25 = spirv.Load "Function" %3 : vector<3xf32>
    %26 = spirv.CompositeExtract %25[2 : i32] : vector<3xf32>
    spirv.Store "Function" %22, %26 : f32
    %27 = spirv.Load "Function" %5 : vector<4xf32>
    %28 = spirv.CompositeExtract %27[3 : i32] : vector<4xf32>
    spirv.Store "Function" %22, %28 : f32
    %29 = spirv.Load "Function" %5 : vector<4xf32>
    %30 = spirv.VectorShuffle [2 : i32, 1 : i32, 0 : i32] %29 : vector<4xf32>, %29 : vector<4xf32> -> vector<3xf32>
    spirv.Store "Function" %11, %30 : vector<3xf32>
    %cst_f32_8 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_9 = spirv.Constant 0.000000e+00 : f32
    %cst_f32_10 = spirv.Constant 0.000000e+00 : f32
    %31 = spirv.CompositeConstruct %cst_f32_8, %cst_f32_9, %cst_f32_10 : (f32, f32, f32) -> vector<3xf32>
    %cst_f32_11 = spirv.Constant 5.000000e-01 : f32
    %cst_f32_12 = spirv.Constant 0.000000e+00 : f32
    %cst_f32_13 = spirv.Constant 1.000000e+00 : f32
    %32 = spirv.CompositeConstruct %cst_f32_11, %cst_f32_12, %cst_f32_13 : (f32, f32, f32) -> vector<3xf32>
    %33 = spirv.CompositeConstruct %31, %32 : (vector<3xf32>, vector<3xf32>) -> !spirv.struct<(vector<3xf32>, vector<3xf32>)>
    %34 = spirv.Variable : !spirv.ptr<!spirv.struct<(vector<3xf32>, vector<3xf32>)>, Function>
    spirv.Store "Function" %34, %33 : !spirv.struct<(vector<3xf32>, vector<3xf32>)>
    %cst0_i32 = spirv.Constant 0 : i32
    %35 = spirv.AccessChain %34[%cst0_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32>, vector<3xf32>)>, Function>, i32
    %36 = spirv.Load "Function" %35 : vector<3xf32>
    %37 = spirv.VectorShuffle [1 : i32, 0 : i32] %36 : vector<3xf32>, %36 : vector<3xf32> -> vector<2xf32>
    spirv.Store "Function" %19, %37 : vector<2xf32>
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
