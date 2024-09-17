spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @main() "None" {
    %cst1_si32 = spirv.Constant 1 : si32
    %0 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %0, %cst1_si32 : si32
    %cst2_ui32 = spirv.Constant 2 : ui32
    %1 = spirv.Variable : !spirv.ptr<ui32, Function>
    spirv.Store "Function" %1, %cst2_ui32 : ui32
    %cst_f32 = spirv.Constant 1.000000e+00 : f32
    %2 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %2, %cst_f32 : f32
    %cst_f64 = spirv.Constant 1.000000e+00 : f64
    %3 = spirv.Variable : !spirv.ptr<f64, Function>
    spirv.Store "Function" %3, %cst_f64 : f64
    %true = spirv.Constant true
    %4 = spirv.Variable : !spirv.ptr<i1, Function>
    spirv.Store "Function" %4, %true : i1
    %5 = spirv.Load "Function" %1 : ui32
    %6 = spirv.Bitcast %5 : ui32 to si32
    %7 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %7, %6 : si32
    %8 = spirv.Load "Function" %4 : i1
    %cst1_si32_0 = spirv.Constant 1 : si32
    %cst0_si32 = spirv.Constant 0 : si32
    %9 = spirv.Select %8, %cst1_si32_0, %cst0_si32 : i1, si32
    %10 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %10, %9 : si32
    %11 = spirv.Load "Function" %2 : f32
    %12 = spirv.ConvertFToS %11 : f32 to si32
    %13 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %13, %12 : si32
    %14 = spirv.Load "Function" %3 : f64
    %15 = spirv.ConvertFToS %14 : f64 to si32
    %16 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %16, %15 : si32
    %17 = spirv.Load "Function" %0 : si32
    %18 = spirv.Bitcast %17 : si32 to ui32
    %19 = spirv.Variable : !spirv.ptr<ui32, Function>
    spirv.Store "Function" %19, %18 : ui32
    %20 = spirv.Load "Function" %4 : i1
    %cst1_ui32 = spirv.Constant 1 : ui32
    %cst0_ui32 = spirv.Constant 0 : ui32
    %21 = spirv.Select %20, %cst1_ui32, %cst0_ui32 : i1, ui32
    %22 = spirv.Variable : !spirv.ptr<ui32, Function>
    spirv.Store "Function" %22, %21 : ui32
    %23 = spirv.Load "Function" %2 : f32
    %24 = spirv.ConvertFToU %23 : f32 to ui32
    %25 = spirv.Variable : !spirv.ptr<ui32, Function>
    spirv.Store "Function" %25, %24 : ui32
    %26 = spirv.Load "Function" %3 : f64
    %27 = spirv.ConvertFToU %26 : f64 to ui32
    %28 = spirv.Variable : !spirv.ptr<ui32, Function>
    spirv.Store "Function" %28, %27 : ui32
    %29 = spirv.Load "Function" %0 : si32
    %30 = spirv.ConvertSToF %29 : si32 to f32
    %31 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %31, %30 : f32
    %32 = spirv.Load "Function" %1 : ui32
    %33 = spirv.ConvertUToF %32 : ui32 to f32
    %34 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %34, %33 : f32
    %35 = spirv.Load "Function" %4 : i1
    %cst_f32_1 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_2 = spirv.Constant 0.000000e+00 : f32
    %36 = spirv.Select %35, %cst_f32_1, %cst_f32_2 : i1, f32
    %37 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %37, %36 : f32
    %38 = spirv.Load "Function" %3 : f64
    %39 = spirv.FConvert %38 : f64 to f32
    %40 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %40, %39 : f32
    %41 = spirv.Load "Function" %0 : si32
    %42 = spirv.ConvertSToF %41 : si32 to f64
    %43 = spirv.Variable : !spirv.ptr<f64, Function>
    spirv.Store "Function" %43, %42 : f64
    %44 = spirv.Load "Function" %1 : ui32
    %45 = spirv.ConvertUToF %44 : ui32 to f64
    %46 = spirv.Variable : !spirv.ptr<f64, Function>
    spirv.Store "Function" %46, %45 : f64
    %47 = spirv.Load "Function" %4 : i1
    %cst_f64_3 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_4 = spirv.Constant 0.000000e+00 : f64
    %48 = spirv.Select %47, %cst_f64_3, %cst_f64_4 : i1, f64
    %49 = spirv.Variable : !spirv.ptr<f64, Function>
    spirv.Store "Function" %49, %48 : f64
    %50 = spirv.Load "Function" %2 : f32
    %51 = spirv.FConvert %50 : f32 to f64
    %52 = spirv.Variable : !spirv.ptr<f64, Function>
    spirv.Store "Function" %52, %51 : f64
    %53 = spirv.Load "Function" %0 : si32
    %cst0_si32_5 = spirv.Constant 0 : si32
    %54 = spirv.INotEqual %53, %cst0_si32_5 : si32
    %55 = spirv.Variable : !spirv.ptr<i1, Function>
    spirv.Store "Function" %55, %54 : i1
    %56 = spirv.Load "Function" %1 : ui32
    %cst0_ui32_6 = spirv.Constant 0 : ui32
    %57 = spirv.INotEqual %56, %cst0_ui32_6 : ui32
    %58 = spirv.Variable : !spirv.ptr<i1, Function>
    spirv.Store "Function" %58, %57 : i1
    %59 = spirv.Load "Function" %2 : f32
    %cst_f32_7 = spirv.Constant 0.000000e+00 : f32
    %60 = spirv.FOrdNotEqual %59, %cst_f32_7 : f32
    %61 = spirv.Variable : !spirv.ptr<i1, Function>
    spirv.Store "Function" %61, %60 : i1
    %62 = spirv.Load "Function" %3 : f64
    %cst_f64_8 = spirv.Constant 0.000000e+00 : f64
    %63 = spirv.FOrdNotEqual %62, %cst_f64_8 : f64
    %64 = spirv.Variable : !spirv.ptr<i1, Function>
    spirv.Store "Function" %64, %63 : i1
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
