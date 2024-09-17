spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @main() "None" {
    %true = spirv.Constant true
    %0 = spirv.Variable : !spirv.ptr<i1, Function>
    spirv.Store "Function" %0, %true : i1
    %1 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.mlir.loop {
      spirv.Branch ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      %2 = spirv.Load "Function" %0 : i1
      spirv.BranchConditional %2, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %cst2_si32 = spirv.Constant 2 : si32
      %3 = spirv.Variable : !spirv.ptr<si32, Function>
      spirv.Store "Function" %3, %cst2_si32 : si32
      %cst3_si32 = spirv.Constant 3 : si32
      %4 = spirv.Variable : !spirv.ptr<si32, Function>
      spirv.Store "Function" %4, %cst3_si32 : si32
      %5 = spirv.Load "Function" %4 : si32
      %6 = spirv.Load "Function" %3 : si32
      %7 = spirv.IAdd %6, %5 : si32
      spirv.Store "Function" %1, %7 : si32
      spirv.Branch ^bb1
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
