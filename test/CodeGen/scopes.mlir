spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @main() "None" {
    %0 = spirv.Variable : !spirv.ptr<si32, Function>
    %cst1_si32 = spirv.Constant 1 : si32
    %1 = spirv.Load "Function" %0 : si32
    %2 = spirv.IEqual %1, %cst1_si32 : si32
    spirv.mlir.selection {
      spirv.BranchConditional %2, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %5 = spirv.Variable : !spirv.ptr<si32, Function>
      %cst2_si32_1 = spirv.Constant 2 : si32
      spirv.Store "Function" %5, %cst2_si32_1 : si32
      spirv.Branch ^bb3
    ^bb2:  // pred: ^bb0
      %6 = spirv.Variable : !spirv.ptr<si32, Function>
      %cst3_si32 = spirv.Constant 3 : si32
      spirv.Store "Function" %6, %cst3_si32 : si32
      spirv.Branch ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      spirv.mlir.merge
    }
    %cst2_si32 = spirv.Constant 2 : si32
    spirv.Store "Function" %0, %cst2_si32 : si32
    spirv.mlir.loop {
      spirv.Branch ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      %cst1_si32_1 = spirv.Constant 1 : si32
      %5 = spirv.Load "Function" %0 : si32
      %6 = spirv.IEqual %5, %cst1_si32_1 : si32
      spirv.BranchConditional %6, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %7 = spirv.Variable : !spirv.ptr<si32, Function>
      %cst5_si32 = spirv.Constant 5 : si32
      spirv.Store "Function" %7, %cst5_si32 : si32
      spirv.Branch ^bb1
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    %cst4_si32 = spirv.Constant 4 : si32
    spirv.Store "Function" %0, %cst4_si32 : si32
    %cst1_si32_0 = spirv.Constant 1 : si32
    %3 = spirv.Load "Function" %0 : si32
    %4 = spirv.IEqual %3, %cst1_si32_0 : si32
    spirv.mlir.selection {
      spirv.BranchConditional %4, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %5 = spirv.Variable : !spirv.ptr<si32, Function>
      %cst1_si32_1 = spirv.Constant 1 : si32
      spirv.Store "Function" %5, %cst1_si32_1 : si32
      %cst2_si32_2 = spirv.Constant 2 : si32
      %6 = spirv.Load "Function" %5 : si32
      %7 = spirv.IEqual %6, %cst2_si32_2 : si32
      spirv.mlir.selection {
        spirv.BranchConditional %7, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %8 = spirv.Variable : !spirv.ptr<si32, Function>
        %cst2_si32_3 = spirv.Constant 2 : si32
        spirv.Store "Function" %8, %cst2_si32_3 : si32
        spirv.Branch ^bb3
      ^bb2:  // pred: ^bb0
        spirv.Branch ^bb3
      ^bb3:  // 2 preds: ^bb1, ^bb2
        spirv.mlir.merge
      }
      spirv.Branch ^bb3
    ^bb2:  // pred: ^bb0
      spirv.Branch ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
