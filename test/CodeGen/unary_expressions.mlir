spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @main() "None" {
    %cst0_si32 = spirv.Constant 0 : si32
    %0 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %0, %cst0_si32 : si32
    %1 = spirv.Load "Function" %0 : si32
    %cst1_si32 = spirv.Constant 1 : si32
    %2 = spirv.IAdd %1, %cst1_si32 : si32
    spirv.Store "Function" %0, %2 : si32
    %3 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %3, %2 : si32
    %cst_f32 = spirv.Constant 1.000000e+00 : f32
    %4 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %4, %cst_f32 : f32
    %5 = spirv.Load "Function" %4 : f32
    %cst_f32_0 = spirv.Constant 1.000000e+00 : f32
    %6 = spirv.FAdd %5, %cst_f32_0 : f32
    spirv.Store "Function" %4, %6 : f32
    %7 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %7, %6 : f32
    %8 = spirv.Load "Function" %0 : si32
    %cst1_si32_1 = spirv.Constant 1 : si32
    %9 = spirv.ISub %8, %cst1_si32_1 : si32
    spirv.Store "Function" %0, %9 : si32
    spirv.Store "Function" %3, %9 : si32
    %10 = spirv.Load "Function" %4 : f32
    %cst_f32_2 = spirv.Constant 1.000000e+00 : f32
    %11 = spirv.FSub %10, %cst_f32_2 : f32
    spirv.Store "Function" %4, %11 : f32
    spirv.Store "Function" %7, %11 : f32
    %12 = spirv.Load "Function" %0 : si32
    %13 = spirv.SNegate %12 : si32
    %14 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %14, %13 : si32
    %15 = spirv.Load "Function" %4 : f32
    %16 = spirv.FNegate %15 : f32
    spirv.Store "Function" %4, %16 : f32
    %17 = spirv.Load "Function" %0 : si32
    spirv.Store "Function" %14, %17 : si32
    %18 = spirv.Load "Function" %0 : si32
    %19 = spirv.Not %18 : si32
    spirv.Store "Function" %14, %19 : si32
    %true = spirv.Constant true
    %20 = spirv.Variable : !spirv.ptr<i1, Function>
    spirv.Store "Function" %20, %true : i1
    %21 = spirv.Load "Function" %20 : i1
    %22 = spirv.LogicalNot %21 : i1
    spirv.Store "Function" %20, %22 : i1
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
