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
    %2 = spirv.Load "Function" %0 : i1
    spirv.mlir.selection {
      spirv.BranchConditional %2, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %cst0_si32 = spirv.Constant 0 : si32
      spirv.Store "Function" %1, %cst0_si32 : si32
      spirv.Branch ^bb3
    ^bb2:  // pred: ^bb0
      %cst1_si32 = spirv.Constant 1 : si32
      spirv.Store "Function" %1, %cst1_si32 : si32
      spirv.Branch ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
