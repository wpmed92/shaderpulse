spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @main() "None" {
    %cst_f32 = spirv.Constant 1.000000e-01 : f32
    %cst_f32_0 = spirv.Constant 2.000000e-01 : f32
    %cst_f32_1 = spirv.Constant 3.000000e-01 : f32
    %0 = spirv.CompositeConstruct %cst_f32, %cst_f32_0, %cst_f32_1 : (f32, f32, f32) -> !spirv.array<3 x f32>
    %1 = spirv.Variable : !spirv.ptr<!spirv.array<3 x f32>, Function>
    spirv.Store "Function" %1, %0 : !spirv.array<3 x f32>
    %cst0_si32 = spirv.Constant 0 : si32
    %2 = spirv.AccessChain %1[%cst0_si32] : !spirv.ptr<!spirv.array<3 x f32>, Function>, si32
    %3 = spirv.Load "Function" %2 : f32
    %4 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %4, %3 : f32
    %cst1_si32 = spirv.Constant 1 : si32
    %5 = spirv.AccessChain %1[%cst1_si32] : !spirv.ptr<!spirv.array<3 x f32>, Function>, si32
    %6 = spirv.Load "Function" %5 : f32
    %7 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %7, %6 : f32
    %cst2_si32 = spirv.Constant 2 : si32
    %8 = spirv.AccessChain %1[%cst2_si32] : !spirv.ptr<!spirv.array<3 x f32>, Function>, si32
    %9 = spirv.Load "Function" %8 : f32
    %10 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %10, %9 : f32
    %cst0_si32_2 = spirv.Constant 0 : si32
    %11 = spirv.AccessChain %1[%cst0_si32_2] : !spirv.ptr<!spirv.array<3 x f32>, Function>, si32
    %cst_f32_3 = spirv.Constant 0.00999999977 : f32
    spirv.Store "Function" %11, %cst_f32_3 : f32
    %cst1_si32_4 = spirv.Constant 1 : si32
    %12 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %12, %cst1_si32_4 : si32
    %13 = spirv.Load "Function" %12 : si32
    %14 = spirv.AccessChain %1[%13] : !spirv.ptr<!spirv.array<3 x f32>, Function>, si32
    %cst_f32_5 = spirv.Constant 2.000000e-02 : f32
    spirv.Store "Function" %14, %cst_f32_5 : f32
    %cst_f32_6 = spirv.Constant 1.000000e-01 : f32
    %cst_f32_7 = spirv.Constant 2.000000e-01 : f32
    %cst_f32_8 = spirv.Constant 3.000000e-01 : f32
    %15 = spirv.CompositeConstruct %cst_f32_6, %cst_f32_7, %cst_f32_8 : (f32, f32, f32) -> !spirv.array<3 x f32>
    %cst_f32_9 = spirv.Constant 4.000000e-01 : f32
    %cst_f32_10 = spirv.Constant 5.000000e-01 : f32
    %cst_f32_11 = spirv.Constant 6.000000e-01 : f32
    %16 = spirv.CompositeConstruct %cst_f32_9, %cst_f32_10, %cst_f32_11 : (f32, f32, f32) -> !spirv.array<3 x f32>
    %17 = spirv.CompositeConstruct %15, %16 : (!spirv.array<3 x f32>, !spirv.array<3 x f32>) -> !spirv.array<2 x !spirv.array<3 x f32>>
    %18 = spirv.Variable : !spirv.ptr<!spirv.array<2 x !spirv.array<3 x f32>>, Function>
    spirv.Store "Function" %18, %17 : !spirv.array<2 x !spirv.array<3 x f32>>
    %cst0_si32_12 = spirv.Constant 0 : si32
    %cst1_si32_13 = spirv.Constant 1 : si32
    %19 = spirv.AccessChain %18[%cst0_si32_12, %cst1_si32_13] : !spirv.ptr<!spirv.array<2 x !spirv.array<3 x f32>>, Function>, si32, si32
    %20 = spirv.Load "Function" %19 : f32
    %21 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %21, %20 : f32
    %cst0_si32_14 = spirv.Constant 0 : si32
    %cst1_si32_15 = spirv.Constant 1 : si32
    %22 = spirv.AccessChain %18[%cst0_si32_14, %cst1_si32_15] : !spirv.ptr<!spirv.array<2 x !spirv.array<3 x f32>>, Function>, si32, si32
    %cst_f32_16 = spirv.Constant 1.000000e+00 : f32
    spirv.Store "Function" %22, %cst_f32_16 : f32
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
