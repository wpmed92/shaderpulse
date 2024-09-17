spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @add(%arg0: si32, %arg1: si32) -> si32 "None" {
    %0 = spirv.IAdd %arg0, %arg1 : si32
    spirv.ReturnValue %0 : si32
  }
  spirv.func @main() "None" {
    %cst1_si32 = spirv.Constant 1 : si32
    %0 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %0, %cst1_si32 : si32
    %cst2_si32 = spirv.Constant 2 : si32
    %1 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %1, %cst2_si32 : si32
    %2 = spirv.Load "Function" %0 : si32
    %3 = spirv.Load "Function" %1 : si32
    %4 = spirv.FunctionCall @add(%2, %3) : (si32, si32) -> si32
    %5 = spirv.Load "Function" %0 : si32
    %6 = spirv.Load "Function" %1 : si32
    %7 = spirv.FunctionCall @add(%5, %6) : (si32, si32) -> si32
    %8 = spirv.IMul %4, %7 : si32
    %9 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %9, %8 : si32
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
