spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @imageData {binding = 0 : i32} : !spirv.ptr<!spirv.rtarray<vector<4xf32>>, StorageBuffer>
  spirv.func @main() "None" {
    %gl_GlobalInvocationID_addr = spirv.mlir.addressof @gl_GlobalInvocationID : !spirv.ptr<vector<3xui32>, Input>
    %0 = spirv.Load "Input" %gl_GlobalInvocationID_addr : vector<3xui32>
    %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xui32>
    %2 = spirv.ConvertUToF %1 : ui32 to f32
    %cst3200_si32 = spirv.Constant 3200 : si32
    %3 = spirv.ConvertSToF %cst3200_si32 : si32 to f32
    %4 = spirv.FDiv %2, %3 : f32
    %5 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %5, %4 : f32
    %gl_GlobalInvocationID_addr_0 = spirv.mlir.addressof @gl_GlobalInvocationID : !spirv.ptr<vector<3xui32>, Input>
    %6 = spirv.Load "Input" %gl_GlobalInvocationID_addr_0 : vector<3xui32>
    %7 = spirv.CompositeExtract %6[1 : i32] : vector<3xui32>
    %8 = spirv.ConvertUToF %7 : ui32 to f32
    %cst2400_si32 = spirv.Constant 2400 : si32
    %9 = spirv.ConvertSToF %cst2400_si32 : si32 to f32
    %10 = spirv.FDiv %8, %9 : f32
    %11 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %11, %10 : f32
    %12 = spirv.Load "Function" %5 : f32
    %13 = spirv.Load "Function" %11 : f32
    %14 = spirv.CompositeConstruct %12, %13 : (f32, f32) -> vector<2xf32>
    %15 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %15, %14 : vector<2xf32>
    %cst_f32 = spirv.Constant 0.000000e+00 : f32
    %16 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %16, %cst_f32 : f32
    %cst_f32_1 = spirv.Constant 2.000000e+00 : f32
    %cst_f32_2 = spirv.Constant 1.700000e+00 : f32
    %cst_f32_3 = spirv.Constant 2.000000e-01 : f32
    %17 = spirv.FMul %cst_f32_2, %cst_f32_3 : f32
    %18 = spirv.FAdd %cst_f32_1, %17 : f32
    %19 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %19, %18 : f32
    %cst_f32_4 = spirv.Constant 4.450000e-01 : f32
    %20 = spirv.FNegate %cst_f32_4 : f32
    %cst_f32_5 = spirv.Constant 0.000000e+00 : f32
    %21 = spirv.CompositeConstruct %20, %cst_f32_5 : (f32, f32) -> vector<2xf32>
    %cst_f32_6 = spirv.Constant 5.000000e-01 : f32
    %cst_f32_7 = spirv.Constant 5.000000e-01 : f32
    %22 = spirv.CompositeConstruct %cst_f32_6, %cst_f32_7 : (f32, f32) -> vector<2xf32>
    %23 = spirv.Load "Function" %15 : vector<2xf32>
    %24 = spirv.FSub %23, %22 : vector<2xf32>
    %25 = spirv.Load "Function" %19 : f32
    %26 = spirv.Load "Function" %19 : f32
    %27 = spirv.CompositeConstruct %25, %26 : (f32, f32) -> vector<2xf32>
    %28 = spirv.FMul %24, %27 : vector<2xf32>
    %29 = spirv.FAdd %21, %28 : vector<2xf32>
    %30 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %30, %29 : vector<2xf32>
    %cst_f32_8 = spirv.Constant 0.000000e+00 : f32
    %cst_f32_9 = spirv.Constant 0.000000e+00 : f32
    %31 = spirv.CompositeConstruct %cst_f32_8, %cst_f32_9 : (f32, f32) -> vector<2xf32>
    %32 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %32, %31 : vector<2xf32>
    %cst128_si32 = spirv.Constant 128 : si32
    %33 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %33, %cst128_si32 : si32
    %34 = spirv.Variable : !spirv.ptr<i1, Function>
    %35 = spirv.Variable : !spirv.ptr<i1, Function>
    %cst0_si32 = spirv.Constant 0 : si32
    %36 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %36, %cst0_si32 : si32
    spirv.mlir.loop {
      spirv.Branch ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb4
      %77 = spirv.Load "Function" %33 : si32
      %78 = spirv.Load "Function" %36 : si32
      %79 = spirv.SLessThan %78, %77 : si32
      spirv.BranchConditional %79, ^bb2, ^bb5
    ^bb2:  // pred: ^bb1
      %80 = spirv.Load "Function" %32 : vector<2xf32>
      %81 = spirv.CompositeExtract %80[0 : i32] : vector<2xf32>
      %82 = spirv.Load "Function" %32 : vector<2xf32>
      %83 = spirv.CompositeExtract %82[0 : i32] : vector<2xf32>
      %84 = spirv.FMul %81, %83 : f32
      %85 = spirv.Load "Function" %32 : vector<2xf32>
      %86 = spirv.CompositeExtract %85[1 : i32] : vector<2xf32>
      %87 = spirv.Load "Function" %32 : vector<2xf32>
      %88 = spirv.CompositeExtract %87[1 : i32] : vector<2xf32>
      %89 = spirv.FMul %86, %88 : f32
      %90 = spirv.FSub %84, %89 : f32
      %cst_f32_29 = spirv.Constant 2.000000e+00 : f32
      %91 = spirv.Load "Function" %32 : vector<2xf32>
      %92 = spirv.CompositeExtract %91[0 : i32] : vector<2xf32>
      %93 = spirv.FMul %cst_f32_29, %92 : f32
      %94 = spirv.Load "Function" %32 : vector<2xf32>
      %95 = spirv.CompositeExtract %94[1 : i32] : vector<2xf32>
      %96 = spirv.FMul %93, %95 : f32
      %97 = spirv.CompositeConstruct %90, %96 : (f32, f32) -> vector<2xf32>
      %98 = spirv.Load "Function" %30 : vector<2xf32>
      %99 = spirv.FAdd %97, %98 : vector<2xf32>
      spirv.Store "Function" %32, %99 : vector<2xf32>
      %100 = spirv.Load "Function" %32 : vector<2xf32>
      %101 = spirv.Load "Function" %32 : vector<2xf32>
      %102 = spirv.Dot %100, %101 : vector<2xf32> -> f32
      %103 = spirv.ConvertFToS %102 : f32 to si32
      %cst2_si32 = spirv.Constant 2 : si32
      %104 = spirv.SGreaterThan %103, %cst2_si32 : si32
      spirv.mlir.selection {
        spirv.BranchConditional %104, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %true = spirv.Constant true
        spirv.Store "Function" %34, %true : i1
        spirv.Branch ^bb3
      ^bb2:  // pred: ^bb0
        spirv.Branch ^bb3
      ^bb3:  // 2 preds: ^bb1, ^bb2
        spirv.mlir.merge
      }
      %105 = spirv.Load "Function" %34 : i1
      spirv.BranchConditional %105, ^bb5, ^bb3
    ^bb3:  // pred: ^bb2
      %false = spirv.Constant false
      spirv.Store "Function" %34, %false : i1
      %cst_f32_30 = spirv.Constant 1.000000e+00 : f32
      %106 = spirv.Load "Function" %16 : f32
      %107 = spirv.FAdd %106, %cst_f32_30 : f32
      spirv.Store "Function" %16, %107 : f32
      spirv.Branch ^bb4
    ^bb4:  // pred: ^bb3
      %108 = spirv.Load "Function" %36 : si32
      %cst1_si32 = spirv.Constant 1 : si32
      %109 = spirv.IAdd %108, %cst1_si32 : si32
      spirv.Store "Function" %36, %109 : si32
      spirv.Branch ^bb1
    ^bb5:  // 2 preds: ^bb1, ^bb2
      spirv.mlir.merge
    }
    %37 = spirv.Load "Function" %16 : f32
    %cst128_si32_10 = spirv.Constant 128 : si32
    %38 = spirv.ConvertSToF %cst128_si32_10 : si32 to f32
    %39 = spirv.FDiv %37, %38 : f32
    %40 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %40, %39 : f32
    %cst_f32_11 = spirv.Constant 3.000000e-01 : f32
    %cst_f32_12 = spirv.Constant 3.000000e-01 : f32
    %cst_f32_13 = spirv.Constant 5.000000e-01 : f32
    %41 = spirv.CompositeConstruct %cst_f32_11, %cst_f32_12, %cst_f32_13 : (f32, f32, f32) -> vector<3xf32>
    %42 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %42, %41 : vector<3xf32>
    %cst_f32_14 = spirv.Constant 2.000000e-01 : f32
    %43 = spirv.FNegate %cst_f32_14 : f32
    %cst_f32_15 = spirv.Constant 3.000000e-01 : f32
    %44 = spirv.FNegate %cst_f32_15 : f32
    %cst_f32_16 = spirv.Constant 5.000000e-01 : f32
    %45 = spirv.FNegate %cst_f32_16 : f32
    %46 = spirv.CompositeConstruct %43, %44, %45 : (f32, f32, f32) -> vector<3xf32>
    %47 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %47, %46 : vector<3xf32>
    %cst_f32_17 = spirv.Constant 2.100000e+00 : f32
    %cst_f32_18 = spirv.Constant 2.000000e+00 : f32
    %cst_f32_19 = spirv.Constant 3.000000e+00 : f32
    %48 = spirv.CompositeConstruct %cst_f32_17, %cst_f32_18, %cst_f32_19 : (f32, f32, f32) -> vector<3xf32>
    %49 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %49, %48 : vector<3xf32>
    %cst_f32_20 = spirv.Constant 0.000000e+00 : f32
    %cst_f32_21 = spirv.Constant 1.000000e-01 : f32
    %cst_f32_22 = spirv.Constant 0.000000e+00 : f32
    %50 = spirv.CompositeConstruct %cst_f32_20, %cst_f32_21, %cst_f32_22 : (f32, f32, f32) -> vector<3xf32>
    %51 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %51, %50 : vector<3xf32>
    %cst_f32_23 = spirv.Constant 6.283180e+00 : f32
    %cst_f32_24 = spirv.Constant 6.283180e+00 : f32
    %cst_f32_25 = spirv.Constant 6.283180e+00 : f32
    %52 = spirv.CompositeConstruct %cst_f32_23, %cst_f32_24, %cst_f32_25 : (f32, f32, f32) -> vector<3xf32>
    %53 = spirv.Load "Function" %40 : f32
    %54 = spirv.Load "Function" %40 : f32
    %55 = spirv.Load "Function" %40 : f32
    %56 = spirv.CompositeConstruct %53, %54, %55 : (f32, f32, f32) -> vector<3xf32>
    %57 = spirv.Load "Function" %49 : vector<3xf32>
    %58 = spirv.FMul %57, %56 : vector<3xf32>
    %59 = spirv.Load "Function" %51 : vector<3xf32>
    %60 = spirv.FAdd %58, %59 : vector<3xf32>
    %61 = spirv.FMul %52, %60 : vector<3xf32>
    %62 = spirv.GL.Cos %61 : vector<3xf32>
    %63 = spirv.Load "Function" %47 : vector<3xf32>
    %64 = spirv.FMul %63, %62 : vector<3xf32>
    %65 = spirv.Load "Function" %42 : vector<3xf32>
    %66 = spirv.FAdd %65, %64 : vector<3xf32>
    %cst_f32_26 = spirv.Constant 1.000000e+00 : f32
    %67 = spirv.CompositeConstruct %66, %cst_f32_26 : (vector<3xf32>, f32) -> vector<4xf32>
    %68 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %68, %67 : vector<4xf32>
    %imageData_addr = spirv.mlir.addressof @imageData : !spirv.ptr<!spirv.rtarray<vector<4xf32>>, StorageBuffer>
    %cst3200_ui32 = spirv.Constant 3200 : ui32
    %gl_GlobalInvocationID_addr_27 = spirv.mlir.addressof @gl_GlobalInvocationID : !spirv.ptr<vector<3xui32>, Input>
    %69 = spirv.Load "Input" %gl_GlobalInvocationID_addr_27 : vector<3xui32>
    %70 = spirv.CompositeExtract %69[1 : i32] : vector<3xui32>
    %71 = spirv.IMul %cst3200_ui32, %70 : ui32
    %gl_GlobalInvocationID_addr_28 = spirv.mlir.addressof @gl_GlobalInvocationID : !spirv.ptr<vector<3xui32>, Input>
    %72 = spirv.Load "Input" %gl_GlobalInvocationID_addr_28 : vector<3xui32>
    %73 = spirv.CompositeExtract %72[0 : i32] : vector<3xui32>
    %74 = spirv.IAdd %71, %73 : ui32
    %75 = spirv.AccessChain %imageData_addr[%74] : !spirv.ptr<!spirv.rtarray<vector<4xf32>>, StorageBuffer>, ui32
    %76 = spirv.Load "Function" %68 : vector<4xf32>
    spirv.Store "StorageBuffer" %75, %76 : vector<4xf32>
    spirv.Return
  }
  spirv.ExecutionMode @main "LocalSize", 32, 32, 1
  spirv.EntryPoint "GLCompute" @main, @gl_GlobalInvocationID, @imageData
}
