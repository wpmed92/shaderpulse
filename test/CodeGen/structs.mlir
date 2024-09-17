spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @main() "None" {
    %cst_f32 = spirv.Constant 1.000000e-01 : f32
    %cst2_si32 = spirv.Constant 2 : si32
    %cst3_ui32 = spirv.Constant 3 : ui32
    %true = spirv.Constant true
    %0 = spirv.CompositeConstruct %cst_f32, %cst2_si32, %cst3_ui32, %true : (f32, si32, ui32, i1) -> !spirv.struct<(f32, si32, ui32, i1)>
    %1 = spirv.Variable : !spirv.ptr<!spirv.struct<(f32, si32, ui32, i1)>, Function>
    spirv.Store "Function" %1, %0 : !spirv.struct<(f32, si32, ui32, i1)>
    %cst0_i32 = spirv.Constant 0 : i32
    %2 = spirv.AccessChain %1[%cst0_i32] : !spirv.ptr<!spirv.struct<(f32, si32, ui32, i1)>, Function>, i32
    %3 = spirv.Load "Function" %2 : f32
    %4 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %4, %3 : f32
    %cst1_i32 = spirv.Constant 1 : i32
    %5 = spirv.AccessChain %1[%cst1_i32] : !spirv.ptr<!spirv.struct<(f32, si32, ui32, i1)>, Function>, i32
    %6 = spirv.Load "Function" %5 : si32
    %7 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %7, %6 : si32
    %cst2_i32 = spirv.Constant 2 : i32
    %8 = spirv.AccessChain %1[%cst2_i32] : !spirv.ptr<!spirv.struct<(f32, si32, ui32, i1)>, Function>, i32
    %9 = spirv.Load "Function" %8 : ui32
    %10 = spirv.Variable : !spirv.ptr<ui32, Function>
    spirv.Store "Function" %10, %9 : ui32
    %cst3_i32 = spirv.Constant 3 : i32
    %11 = spirv.AccessChain %1[%cst3_i32] : !spirv.ptr<!spirv.struct<(f32, si32, ui32, i1)>, Function>, i32
    %12 = spirv.Load "Function" %11 : i1
    %13 = spirv.Variable : !spirv.ptr<i1, Function>
    spirv.Store "Function" %13, %12 : i1
    %cst_f32_0 = spirv.Constant 1.000000e-01 : f32
    %cst2_si32_1 = spirv.Constant 2 : si32
    %cst3_ui32_2 = spirv.Constant 3 : ui32
    %true_3 = spirv.Constant true
    %14 = spirv.CompositeConstruct %cst_f32_0, %cst2_si32_1, %cst3_ui32_2, %true_3 : (f32, si32, ui32, i1) -> !spirv.struct<(f32, si32, ui32, i1)>
    %cst1_si32 = spirv.Constant 1 : si32
    %15 = spirv.CompositeConstruct %14, %cst1_si32 : (!spirv.struct<(f32, si32, ui32, i1)>, si32) -> !spirv.struct<(!spirv.struct<(f32, si32, ui32, i1)>, si32)>
    %16 = spirv.Variable : !spirv.ptr<!spirv.struct<(!spirv.struct<(f32, si32, ui32, i1)>, si32)>, Function>
    spirv.Store "Function" %16, %15 : !spirv.struct<(!spirv.struct<(f32, si32, ui32, i1)>, si32)>
    %cst0_i32_4 = spirv.Constant 0 : i32
    %cst3_i32_5 = spirv.Constant 3 : i32
    %17 = spirv.AccessChain %16[%cst0_i32_4, %cst3_i32_5] : !spirv.ptr<!spirv.struct<(!spirv.struct<(f32, si32, ui32, i1)>, si32)>, Function>, i32, i32
    %18 = spirv.Load "Function" %17 : i1
    spirv.Store "Function" %13, %18 : i1
    %cst1_si32_6 = spirv.Constant 1 : si32
    %cst2_si32_7 = spirv.Constant 2 : si32
    %cst3_si32 = spirv.Constant 3 : si32
    %cst4_si32 = spirv.Constant 4 : si32
    %19 = spirv.CompositeConstruct %cst1_si32_6, %cst2_si32_7, %cst3_si32, %cst4_si32 : (si32, si32, si32, si32) -> !spirv.array<4 x si32>
    %20 = spirv.CompositeConstruct %19 : (!spirv.array<4 x si32>) -> !spirv.struct<(!spirv.array<4 x si32>)>
    %21 = spirv.Variable : !spirv.ptr<!spirv.struct<(!spirv.array<4 x si32>)>, Function>
    spirv.Store "Function" %21, %20 : !spirv.struct<(!spirv.array<4 x si32>)>
    %cst0_i32_8 = spirv.Constant 0 : i32
    %cst2_si32_9 = spirv.Constant 2 : si32
    %22 = spirv.AccessChain %21[%cst0_i32_8, %cst2_si32_9] : !spirv.ptr<!spirv.struct<(!spirv.array<4 x si32>)>, Function>, i32, si32
    %23 = spirv.Load "Function" %22 : si32
    %24 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %24, %23 : si32
    %cst1_si32_10 = spirv.Constant 1 : si32
    %cst2_si32_11 = spirv.Constant 2 : si32
    %25 = spirv.CompositeConstruct %cst1_si32_10, %cst2_si32_11 : (si32, si32) -> !spirv.array<2 x si32>
    %26 = spirv.Variable : !spirv.ptr<!spirv.array<2 x si32>, Function>
    spirv.Store "Function" %26, %25 : !spirv.array<2 x si32>
    %cst0_si32 = spirv.Constant 0 : si32
    %cst1_si32_12 = spirv.Constant 1 : si32
    %27 = spirv.CompositeConstruct %cst0_si32, %cst1_si32_12 : (si32, si32) -> !spirv.struct<(si32, si32)>
    %28 = spirv.Variable : !spirv.ptr<!spirv.struct<(si32, si32)>, Function>
    spirv.Store "Function" %28, %27 : !spirv.struct<(si32, si32)>
    %cst1_i32_13 = spirv.Constant 1 : i32
    %29 = spirv.AccessChain %28[%cst1_i32_13] : !spirv.ptr<!spirv.struct<(si32, si32)>, Function>, i32
    %30 = spirv.Load "Function" %29 : si32
    %31 = spirv.AccessChain %26[%30] : !spirv.ptr<!spirv.array<2 x si32>, Function>, si32
    %cst24_si32 = spirv.Constant 24 : si32
    spirv.Store "Function" %31, %cst24_si32 : si32
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
