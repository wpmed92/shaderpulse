spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @data {binding = 0 : i32} : !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>
  spirv.GlobalVariable @result {binding = 1 : i32} : !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>
  spirv.func @main() "None" {
    %gl_GlobalInvocationID_addr = spirv.mlir.addressof @gl_GlobalInvocationID : !spirv.ptr<vector<3xui32>, Input>
    %0 = spirv.Load "Input" %gl_GlobalInvocationID_addr : vector<3xui32>
    %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xui32>
    %2 = spirv.Variable : !spirv.ptr<ui32, Function>
    spirv.Store "Function" %2, %1 : ui32
    %result_addr = spirv.mlir.addressof @result : !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>
    %3 = spirv.Load "Function" %2 : ui32
    %4 = spirv.AccessChain %result_addr[%3] : !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>, ui32
    %data_addr = spirv.mlir.addressof @data : !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>
    %5 = spirv.Load "Function" %2 : ui32
    %6 = spirv.AccessChain %data_addr[%5] : !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>, ui32
    %cst_f32 = spirv.Constant 2.000000e+00 : f32
    %7 = spirv.Load "StorageBuffer" %6 : f32
    %8 = spirv.FMul %7, %cst_f32 : f32
    spirv.Store "StorageBuffer" %4, %8 : f32
    spirv.Return
  }
  spirv.ExecutionMode @main "LocalSize", 16, 8, 4
  spirv.EntryPoint "GLCompute" @main, @gl_GlobalInvocationID, @result, @data
}
