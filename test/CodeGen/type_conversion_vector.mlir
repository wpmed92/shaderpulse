spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @main() "None" {
    %cst1_si32 = spirv.Constant 1 : si32
    %cst2_si32 = spirv.Constant 2 : si32
    %0 = spirv.CompositeConstruct %cst1_si32, %cst2_si32 : (si32, si32) -> vector<2xsi32>
    %1 = spirv.Variable : !spirv.ptr<vector<2xsi32>, Function>
    spirv.Store "Function" %1, %0 : vector<2xsi32>
    %cst3_ui32 = spirv.Constant 3 : ui32
    %cst4_ui32 = spirv.Constant 4 : ui32
    %2 = spirv.CompositeConstruct %cst3_ui32, %cst4_ui32 : (ui32, ui32) -> vector<2xui32>
    %3 = spirv.Variable : !spirv.ptr<vector<2xui32>, Function>
    spirv.Store "Function" %3, %2 : vector<2xui32>
    %cst_f32 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_0 = spirv.Constant 2.000000e+00 : f32
    %4 = spirv.CompositeConstruct %cst_f32, %cst_f32_0 : (f32, f32) -> vector<2xf32>
    %5 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %5, %4 : vector<2xf32>
    %cst_f64 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_1 = spirv.Constant 2.000000e+00 : f64
    %6 = spirv.CompositeConstruct %cst_f64, %cst_f64_1 : (f64, f64) -> vector<2xf64>
    %7 = spirv.Variable : !spirv.ptr<vector<2xf64>, Function>
    spirv.Store "Function" %7, %6 : vector<2xf64>
    %true = spirv.Constant true
    %false = spirv.Constant false
    %8 = spirv.CompositeConstruct %true, %false : (i1, i1) -> vector<2xi1>
    %9 = spirv.Variable : !spirv.ptr<vector<2xi1>, Function>
    spirv.Store "Function" %9, %8 : vector<2xi1>
    %cst1_si32_2 = spirv.Constant 1 : si32
    %cst2_si32_3 = spirv.Constant 2 : si32
    %cst3_si32 = spirv.Constant 3 : si32
    %10 = spirv.CompositeConstruct %cst1_si32_2, %cst2_si32_3, %cst3_si32 : (si32, si32, si32) -> vector<3xsi32>
    %11 = spirv.Variable : !spirv.ptr<vector<3xsi32>, Function>
    spirv.Store "Function" %11, %10 : vector<3xsi32>
    %cst3_ui32_4 = spirv.Constant 3 : ui32
    %cst4_ui32_5 = spirv.Constant 4 : ui32
    %cst5_ui32 = spirv.Constant 5 : ui32
    %12 = spirv.CompositeConstruct %cst3_ui32_4, %cst4_ui32_5, %cst5_ui32 : (ui32, ui32, ui32) -> vector<3xui32>
    %13 = spirv.Variable : !spirv.ptr<vector<3xui32>, Function>
    spirv.Store "Function" %13, %12 : vector<3xui32>
    %cst_f32_6 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_7 = spirv.Constant 2.000000e+00 : f32
    %cst_f32_8 = spirv.Constant 3.000000e+00 : f32
    %14 = spirv.CompositeConstruct %cst_f32_6, %cst_f32_7, %cst_f32_8 : (f32, f32, f32) -> vector<3xf32>
    %15 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %15, %14 : vector<3xf32>
    %cst_f64_9 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_10 = spirv.Constant 2.000000e+00 : f64
    %cst_f64_11 = spirv.Constant 3.000000e+00 : f64
    %16 = spirv.CompositeConstruct %cst_f64_9, %cst_f64_10, %cst_f64_11 : (f64, f64, f64) -> vector<3xf64>
    %17 = spirv.Variable : !spirv.ptr<vector<3xf64>, Function>
    spirv.Store "Function" %17, %16 : vector<3xf64>
    %true_12 = spirv.Constant true
    %false_13 = spirv.Constant false
    %true_14 = spirv.Constant true
    %18 = spirv.CompositeConstruct %true_12, %false_13, %true_14 : (i1, i1, i1) -> vector<3xi1>
    %19 = spirv.Variable : !spirv.ptr<vector<3xi1>, Function>
    spirv.Store "Function" %19, %18 : vector<3xi1>
    %cst1_si32_15 = spirv.Constant 1 : si32
    %cst2_si32_16 = spirv.Constant 2 : si32
    %cst3_si32_17 = spirv.Constant 3 : si32
    %cst4_si32 = spirv.Constant 4 : si32
    %20 = spirv.CompositeConstruct %cst1_si32_15, %cst2_si32_16, %cst3_si32_17, %cst4_si32 : (si32, si32, si32, si32) -> vector<4xsi32>
    %21 = spirv.Variable : !spirv.ptr<vector<4xsi32>, Function>
    spirv.Store "Function" %21, %20 : vector<4xsi32>
    %cst3_ui32_18 = spirv.Constant 3 : ui32
    %cst4_ui32_19 = spirv.Constant 4 : ui32
    %cst5_ui32_20 = spirv.Constant 5 : ui32
    %cst6_ui32 = spirv.Constant 6 : ui32
    %22 = spirv.CompositeConstruct %cst3_ui32_18, %cst4_ui32_19, %cst5_ui32_20, %cst6_ui32 : (ui32, ui32, ui32, ui32) -> vector<4xui32>
    %23 = spirv.Variable : !spirv.ptr<vector<4xui32>, Function>
    spirv.Store "Function" %23, %22 : vector<4xui32>
    %cst_f32_21 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_22 = spirv.Constant 2.000000e+00 : f32
    %cst_f32_23 = spirv.Constant 3.000000e+00 : f32
    %cst_f32_24 = spirv.Constant 4.000000e+00 : f32
    %24 = spirv.CompositeConstruct %cst_f32_21, %cst_f32_22, %cst_f32_23, %cst_f32_24 : (f32, f32, f32, f32) -> vector<4xf32>
    %25 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %25, %24 : vector<4xf32>
    %cst_f64_25 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_26 = spirv.Constant 2.000000e+00 : f64
    %cst_f64_27 = spirv.Constant 3.000000e+00 : f64
    %cst_f64_28 = spirv.Constant 4.000000e+00 : f64
    %26 = spirv.CompositeConstruct %cst_f64_25, %cst_f64_26, %cst_f64_27, %cst_f64_28 : (f64, f64, f64, f64) -> vector<4xf64>
    %27 = spirv.Variable : !spirv.ptr<vector<4xf64>, Function>
    spirv.Store "Function" %27, %26 : vector<4xf64>
    %true_29 = spirv.Constant true
    %false_30 = spirv.Constant false
    %true_31 = spirv.Constant true
    %false_32 = spirv.Constant false
    %28 = spirv.CompositeConstruct %true_29, %false_30, %true_31, %false_32 : (i1, i1, i1, i1) -> vector<4xi1>
    %29 = spirv.Variable : !spirv.ptr<vector<4xi1>, Function>
    spirv.Store "Function" %29, %28 : vector<4xi1>
    %30 = spirv.Load "Function" %3 : vector<2xui32>
    %31 = spirv.Bitcast %30 : vector<2xui32> to vector<2xsi32>
    %32 = spirv.Variable : !spirv.ptr<vector<2xsi32>, Function>
    spirv.Store "Function" %32, %31 : vector<2xsi32>
    %33 = spirv.Load "Function" %9 : vector<2xi1>
    %cst1_si32_33 = spirv.Constant 1 : si32
    %cst0_si32 = spirv.Constant 0 : si32
    %34 = spirv.CompositeConstruct %cst0_si32, %cst0_si32 : (si32, si32) -> vector<2xsi32>
    %35 = spirv.CompositeConstruct %cst1_si32_33, %cst1_si32_33 : (si32, si32) -> vector<2xsi32>
    %36 = spirv.Select %33, %35, %34 : vector<2xi1>, vector<2xsi32>
    %37 = spirv.Variable : !spirv.ptr<vector<2xsi32>, Function>
    spirv.Store "Function" %37, %36 : vector<2xsi32>
    %38 = spirv.Load "Function" %5 : vector<2xf32>
    %39 = spirv.ConvertFToS %38 : vector<2xf32> to vector<2xsi32>
    %40 = spirv.Variable : !spirv.ptr<vector<2xsi32>, Function>
    spirv.Store "Function" %40, %39 : vector<2xsi32>
    %41 = spirv.Load "Function" %7 : vector<2xf64>
    %42 = spirv.ConvertFToS %41 : vector<2xf64> to vector<2xsi32>
    %43 = spirv.Variable : !spirv.ptr<vector<2xsi32>, Function>
    spirv.Store "Function" %43, %42 : vector<2xsi32>
    %44 = spirv.Load "Function" %1 : vector<2xsi32>
    %45 = spirv.Bitcast %44 : vector<2xsi32> to vector<2xui32>
    %46 = spirv.Variable : !spirv.ptr<vector<2xui32>, Function>
    spirv.Store "Function" %46, %45 : vector<2xui32>
    %47 = spirv.Load "Function" %9 : vector<2xi1>
    %cst1_ui32 = spirv.Constant 1 : ui32
    %cst0_ui32 = spirv.Constant 0 : ui32
    %48 = spirv.CompositeConstruct %cst0_ui32, %cst0_ui32 : (ui32, ui32) -> vector<2xui32>
    %49 = spirv.CompositeConstruct %cst1_ui32, %cst1_ui32 : (ui32, ui32) -> vector<2xui32>
    %50 = spirv.Select %47, %49, %48 : vector<2xi1>, vector<2xui32>
    %51 = spirv.Variable : !spirv.ptr<vector<2xui32>, Function>
    spirv.Store "Function" %51, %50 : vector<2xui32>
    %52 = spirv.Load "Function" %5 : vector<2xf32>
    %53 = spirv.ConvertFToU %52 : vector<2xf32> to vector<2xui32>
    %54 = spirv.Variable : !spirv.ptr<vector<2xui32>, Function>
    spirv.Store "Function" %54, %53 : vector<2xui32>
    %55 = spirv.Load "Function" %7 : vector<2xf64>
    %56 = spirv.ConvertFToU %55 : vector<2xf64> to vector<2xui32>
    %57 = spirv.Variable : !spirv.ptr<vector<2xui32>, Function>
    spirv.Store "Function" %57, %56 : vector<2xui32>
    %58 = spirv.Load "Function" %1 : vector<2xsi32>
    %59 = spirv.ConvertSToF %58 : vector<2xsi32> to vector<2xf32>
    %60 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %60, %59 : vector<2xf32>
    %61 = spirv.Load "Function" %3 : vector<2xui32>
    %62 = spirv.ConvertUToF %61 : vector<2xui32> to vector<2xf32>
    %63 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %63, %62 : vector<2xf32>
    %64 = spirv.Load "Function" %9 : vector<2xi1>
    %cst_f32_34 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_35 = spirv.Constant 0.000000e+00 : f32
    %65 = spirv.CompositeConstruct %cst_f32_35, %cst_f32_35 : (f32, f32) -> vector<2xf32>
    %66 = spirv.CompositeConstruct %cst_f32_34, %cst_f32_34 : (f32, f32) -> vector<2xf32>
    %67 = spirv.Select %64, %66, %65 : vector<2xi1>, vector<2xf32>
    %68 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %68, %67 : vector<2xf32>
    %69 = spirv.Load "Function" %7 : vector<2xf64>
    %70 = spirv.FConvert %69 : vector<2xf64> to vector<2xf32>
    %71 = spirv.Variable : !spirv.ptr<vector<2xf32>, Function>
    spirv.Store "Function" %71, %70 : vector<2xf32>
    %72 = spirv.Load "Function" %1 : vector<2xsi32>
    %73 = spirv.ConvertSToF %72 : vector<2xsi32> to vector<2xf64>
    %74 = spirv.Variable : !spirv.ptr<vector<2xf64>, Function>
    spirv.Store "Function" %74, %73 : vector<2xf64>
    %75 = spirv.Load "Function" %3 : vector<2xui32>
    %76 = spirv.ConvertUToF %75 : vector<2xui32> to vector<2xf64>
    %77 = spirv.Variable : !spirv.ptr<vector<2xf64>, Function>
    spirv.Store "Function" %77, %76 : vector<2xf64>
    %78 = spirv.Load "Function" %9 : vector<2xi1>
    %cst_f64_36 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_37 = spirv.Constant 0.000000e+00 : f64
    %79 = spirv.CompositeConstruct %cst_f64_37, %cst_f64_37 : (f64, f64) -> vector<2xf64>
    %80 = spirv.CompositeConstruct %cst_f64_36, %cst_f64_36 : (f64, f64) -> vector<2xf64>
    %81 = spirv.Select %78, %80, %79 : vector<2xi1>, vector<2xf64>
    %82 = spirv.Variable : !spirv.ptr<vector<2xf64>, Function>
    spirv.Store "Function" %82, %81 : vector<2xf64>
    %83 = spirv.Load "Function" %5 : vector<2xf32>
    %84 = spirv.FConvert %83 : vector<2xf32> to vector<2xf64>
    %85 = spirv.Variable : !spirv.ptr<vector<2xf64>, Function>
    spirv.Store "Function" %85, %84 : vector<2xf64>
    %86 = spirv.Load "Function" %1 : vector<2xsi32>
    %cst0_si32_38 = spirv.Constant 0 : si32
    %87 = spirv.CompositeConstruct %cst0_si32_38, %cst0_si32_38 : (si32, si32) -> vector<2xsi32>
    %88 = spirv.INotEqual %86, %87 : vector<2xsi32>
    %89 = spirv.Variable : !spirv.ptr<vector<2xi1>, Function>
    spirv.Store "Function" %89, %88 : vector<2xi1>
    %90 = spirv.Load "Function" %3 : vector<2xui32>
    %cst0_ui32_39 = spirv.Constant 0 : ui32
    %91 = spirv.CompositeConstruct %cst0_ui32_39, %cst0_ui32_39 : (ui32, ui32) -> vector<2xui32>
    %92 = spirv.INotEqual %90, %91 : vector<2xui32>
    %93 = spirv.Variable : !spirv.ptr<vector<2xi1>, Function>
    spirv.Store "Function" %93, %92 : vector<2xi1>
    %94 = spirv.Load "Function" %5 : vector<2xf32>
    %cst_f32_40 = spirv.Constant 0.000000e+00 : f32
    %95 = spirv.CompositeConstruct %cst_f32_40, %cst_f32_40 : (f32, f32) -> vector<2xf32>
    %96 = spirv.FOrdNotEqual %94, %95 : vector<2xf32>
    %97 = spirv.Variable : !spirv.ptr<vector<2xi1>, Function>
    spirv.Store "Function" %97, %96 : vector<2xi1>
    %98 = spirv.Load "Function" %7 : vector<2xf64>
    %cst_f64_41 = spirv.Constant 0.000000e+00 : f64
    %99 = spirv.CompositeConstruct %cst_f64_41, %cst_f64_41 : (f64, f64) -> vector<2xf64>
    %100 = spirv.FOrdNotEqual %98, %99 : vector<2xf64>
    %101 = spirv.Variable : !spirv.ptr<vector<2xi1>, Function>
    spirv.Store "Function" %101, %100 : vector<2xi1>
    %102 = spirv.Load "Function" %13 : vector<3xui32>
    %103 = spirv.Bitcast %102 : vector<3xui32> to vector<3xsi32>
    %104 = spirv.Variable : !spirv.ptr<vector<3xsi32>, Function>
    spirv.Store "Function" %104, %103 : vector<3xsi32>
    %105 = spirv.Load "Function" %19 : vector<3xi1>
    %cst1_si32_42 = spirv.Constant 1 : si32
    %cst0_si32_43 = spirv.Constant 0 : si32
    %106 = spirv.CompositeConstruct %cst0_si32_43, %cst0_si32_43, %cst0_si32_43 : (si32, si32, si32) -> vector<3xsi32>
    %107 = spirv.CompositeConstruct %cst1_si32_42, %cst1_si32_42, %cst1_si32_42 : (si32, si32, si32) -> vector<3xsi32>
    %108 = spirv.Select %105, %107, %106 : vector<3xi1>, vector<3xsi32>
    %109 = spirv.Variable : !spirv.ptr<vector<3xsi32>, Function>
    spirv.Store "Function" %109, %108 : vector<3xsi32>
    %110 = spirv.Load "Function" %15 : vector<3xf32>
    %111 = spirv.ConvertFToS %110 : vector<3xf32> to vector<3xsi32>
    %112 = spirv.Variable : !spirv.ptr<vector<3xsi32>, Function>
    spirv.Store "Function" %112, %111 : vector<3xsi32>
    %113 = spirv.Load "Function" %17 : vector<3xf64>
    %114 = spirv.ConvertFToS %113 : vector<3xf64> to vector<3xsi32>
    %115 = spirv.Variable : !spirv.ptr<vector<3xsi32>, Function>
    spirv.Store "Function" %115, %114 : vector<3xsi32>
    %116 = spirv.Load "Function" %11 : vector<3xsi32>
    %117 = spirv.Bitcast %116 : vector<3xsi32> to vector<3xui32>
    %118 = spirv.Variable : !spirv.ptr<vector<3xui32>, Function>
    spirv.Store "Function" %118, %117 : vector<3xui32>
    %119 = spirv.Load "Function" %19 : vector<3xi1>
    %cst1_ui32_44 = spirv.Constant 1 : ui32
    %cst0_ui32_45 = spirv.Constant 0 : ui32
    %120 = spirv.CompositeConstruct %cst0_ui32_45, %cst0_ui32_45, %cst0_ui32_45 : (ui32, ui32, ui32) -> vector<3xui32>
    %121 = spirv.CompositeConstruct %cst1_ui32_44, %cst1_ui32_44, %cst1_ui32_44 : (ui32, ui32, ui32) -> vector<3xui32>
    %122 = spirv.Select %119, %121, %120 : vector<3xi1>, vector<3xui32>
    %123 = spirv.Variable : !spirv.ptr<vector<3xui32>, Function>
    spirv.Store "Function" %123, %122 : vector<3xui32>
    %124 = spirv.Load "Function" %15 : vector<3xf32>
    %125 = spirv.ConvertFToU %124 : vector<3xf32> to vector<3xui32>
    %126 = spirv.Variable : !spirv.ptr<vector<3xui32>, Function>
    spirv.Store "Function" %126, %125 : vector<3xui32>
    %127 = spirv.Load "Function" %17 : vector<3xf64>
    %128 = spirv.ConvertFToU %127 : vector<3xf64> to vector<3xui32>
    %129 = spirv.Variable : !spirv.ptr<vector<3xui32>, Function>
    spirv.Store "Function" %129, %128 : vector<3xui32>
    %130 = spirv.Load "Function" %11 : vector<3xsi32>
    %131 = spirv.ConvertSToF %130 : vector<3xsi32> to vector<3xf32>
    %132 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %132, %131 : vector<3xf32>
    %133 = spirv.Load "Function" %13 : vector<3xui32>
    %134 = spirv.ConvertUToF %133 : vector<3xui32> to vector<3xf32>
    %135 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %135, %134 : vector<3xf32>
    %136 = spirv.Load "Function" %19 : vector<3xi1>
    %cst_f32_46 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_47 = spirv.Constant 0.000000e+00 : f32
    %137 = spirv.CompositeConstruct %cst_f32_47, %cst_f32_47, %cst_f32_47 : (f32, f32, f32) -> vector<3xf32>
    %138 = spirv.CompositeConstruct %cst_f32_46, %cst_f32_46, %cst_f32_46 : (f32, f32, f32) -> vector<3xf32>
    %139 = spirv.Select %136, %138, %137 : vector<3xi1>, vector<3xf32>
    %140 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %140, %139 : vector<3xf32>
    %141 = spirv.Load "Function" %17 : vector<3xf64>
    %142 = spirv.FConvert %141 : vector<3xf64> to vector<3xf32>
    %143 = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>
    spirv.Store "Function" %143, %142 : vector<3xf32>
    %144 = spirv.Load "Function" %11 : vector<3xsi32>
    %145 = spirv.ConvertSToF %144 : vector<3xsi32> to vector<3xf64>
    %146 = spirv.Variable : !spirv.ptr<vector<3xf64>, Function>
    spirv.Store "Function" %146, %145 : vector<3xf64>
    %147 = spirv.Load "Function" %13 : vector<3xui32>
    %148 = spirv.ConvertUToF %147 : vector<3xui32> to vector<3xf64>
    %149 = spirv.Variable : !spirv.ptr<vector<3xf64>, Function>
    spirv.Store "Function" %149, %148 : vector<3xf64>
    %150 = spirv.Load "Function" %19 : vector<3xi1>
    %cst_f64_48 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_49 = spirv.Constant 0.000000e+00 : f64
    %151 = spirv.CompositeConstruct %cst_f64_49, %cst_f64_49, %cst_f64_49 : (f64, f64, f64) -> vector<3xf64>
    %152 = spirv.CompositeConstruct %cst_f64_48, %cst_f64_48, %cst_f64_48 : (f64, f64, f64) -> vector<3xf64>
    %153 = spirv.Select %150, %152, %151 : vector<3xi1>, vector<3xf64>
    %154 = spirv.Variable : !spirv.ptr<vector<3xf64>, Function>
    spirv.Store "Function" %154, %153 : vector<3xf64>
    %155 = spirv.Load "Function" %15 : vector<3xf32>
    %156 = spirv.FConvert %155 : vector<3xf32> to vector<3xf64>
    %157 = spirv.Variable : !spirv.ptr<vector<3xf64>, Function>
    spirv.Store "Function" %157, %156 : vector<3xf64>
    %158 = spirv.Load "Function" %11 : vector<3xsi32>
    %cst0_si32_50 = spirv.Constant 0 : si32
    %159 = spirv.CompositeConstruct %cst0_si32_50, %cst0_si32_50, %cst0_si32_50 : (si32, si32, si32) -> vector<3xsi32>
    %160 = spirv.INotEqual %158, %159 : vector<3xsi32>
    %161 = spirv.Variable : !spirv.ptr<vector<3xi1>, Function>
    spirv.Store "Function" %161, %160 : vector<3xi1>
    %162 = spirv.Load "Function" %13 : vector<3xui32>
    %cst0_ui32_51 = spirv.Constant 0 : ui32
    %163 = spirv.CompositeConstruct %cst0_ui32_51, %cst0_ui32_51, %cst0_ui32_51 : (ui32, ui32, ui32) -> vector<3xui32>
    %164 = spirv.INotEqual %162, %163 : vector<3xui32>
    %165 = spirv.Variable : !spirv.ptr<vector<3xi1>, Function>
    spirv.Store "Function" %165, %164 : vector<3xi1>
    %166 = spirv.Load "Function" %15 : vector<3xf32>
    %cst_f32_52 = spirv.Constant 0.000000e+00 : f32
    %167 = spirv.CompositeConstruct %cst_f32_52, %cst_f32_52, %cst_f32_52 : (f32, f32, f32) -> vector<3xf32>
    %168 = spirv.FOrdNotEqual %166, %167 : vector<3xf32>
    %169 = spirv.Variable : !spirv.ptr<vector<3xi1>, Function>
    spirv.Store "Function" %169, %168 : vector<3xi1>
    %170 = spirv.Load "Function" %17 : vector<3xf64>
    %cst_f64_53 = spirv.Constant 0.000000e+00 : f64
    %171 = spirv.CompositeConstruct %cst_f64_53, %cst_f64_53, %cst_f64_53 : (f64, f64, f64) -> vector<3xf64>
    %172 = spirv.FOrdNotEqual %170, %171 : vector<3xf64>
    %173 = spirv.Variable : !spirv.ptr<vector<3xi1>, Function>
    spirv.Store "Function" %173, %172 : vector<3xi1>
    %174 = spirv.Load "Function" %23 : vector<4xui32>
    %175 = spirv.Bitcast %174 : vector<4xui32> to vector<4xsi32>
    %176 = spirv.Variable : !spirv.ptr<vector<4xsi32>, Function>
    spirv.Store "Function" %176, %175 : vector<4xsi32>
    %177 = spirv.Load "Function" %29 : vector<4xi1>
    %cst1_si32_54 = spirv.Constant 1 : si32
    %cst0_si32_55 = spirv.Constant 0 : si32
    %178 = spirv.CompositeConstruct %cst0_si32_55, %cst0_si32_55, %cst0_si32_55, %cst0_si32_55 : (si32, si32, si32, si32) -> vector<4xsi32>
    %179 = spirv.CompositeConstruct %cst1_si32_54, %cst1_si32_54, %cst1_si32_54, %cst1_si32_54 : (si32, si32, si32, si32) -> vector<4xsi32>
    %180 = spirv.Select %177, %179, %178 : vector<4xi1>, vector<4xsi32>
    %181 = spirv.Variable : !spirv.ptr<vector<4xsi32>, Function>
    spirv.Store "Function" %181, %180 : vector<4xsi32>
    %182 = spirv.Load "Function" %25 : vector<4xf32>
    %183 = spirv.ConvertFToS %182 : vector<4xf32> to vector<4xsi32>
    %184 = spirv.Variable : !spirv.ptr<vector<4xsi32>, Function>
    spirv.Store "Function" %184, %183 : vector<4xsi32>
    %185 = spirv.Load "Function" %27 : vector<4xf64>
    %186 = spirv.ConvertFToS %185 : vector<4xf64> to vector<4xsi32>
    %187 = spirv.Variable : !spirv.ptr<vector<4xsi32>, Function>
    spirv.Store "Function" %187, %186 : vector<4xsi32>
    %188 = spirv.Load "Function" %21 : vector<4xsi32>
    %189 = spirv.Bitcast %188 : vector<4xsi32> to vector<4xui32>
    %190 = spirv.Variable : !spirv.ptr<vector<4xui32>, Function>
    spirv.Store "Function" %190, %189 : vector<4xui32>
    %191 = spirv.Load "Function" %29 : vector<4xi1>
    %cst1_ui32_56 = spirv.Constant 1 : ui32
    %cst0_ui32_57 = spirv.Constant 0 : ui32
    %192 = spirv.CompositeConstruct %cst0_ui32_57, %cst0_ui32_57, %cst0_ui32_57, %cst0_ui32_57 : (ui32, ui32, ui32, ui32) -> vector<4xui32>
    %193 = spirv.CompositeConstruct %cst1_ui32_56, %cst1_ui32_56, %cst1_ui32_56, %cst1_ui32_56 : (ui32, ui32, ui32, ui32) -> vector<4xui32>
    %194 = spirv.Select %191, %193, %192 : vector<4xi1>, vector<4xui32>
    %195 = spirv.Variable : !spirv.ptr<vector<4xui32>, Function>
    spirv.Store "Function" %195, %194 : vector<4xui32>
    %196 = spirv.Load "Function" %25 : vector<4xf32>
    %197 = spirv.ConvertFToU %196 : vector<4xf32> to vector<4xui32>
    %198 = spirv.Variable : !spirv.ptr<vector<4xui32>, Function>
    spirv.Store "Function" %198, %197 : vector<4xui32>
    %199 = spirv.Load "Function" %27 : vector<4xf64>
    %200 = spirv.ConvertFToU %199 : vector<4xf64> to vector<4xui32>
    %201 = spirv.Variable : !spirv.ptr<vector<4xui32>, Function>
    spirv.Store "Function" %201, %200 : vector<4xui32>
    %202 = spirv.Load "Function" %21 : vector<4xsi32>
    %203 = spirv.ConvertSToF %202 : vector<4xsi32> to vector<4xf32>
    %204 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %204, %203 : vector<4xf32>
    %205 = spirv.Load "Function" %23 : vector<4xui32>
    %206 = spirv.ConvertUToF %205 : vector<4xui32> to vector<4xf32>
    %207 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %207, %206 : vector<4xf32>
    %208 = spirv.Load "Function" %29 : vector<4xi1>
    %cst_f32_58 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_59 = spirv.Constant 0.000000e+00 : f32
    %209 = spirv.CompositeConstruct %cst_f32_59, %cst_f32_59, %cst_f32_59, %cst_f32_59 : (f32, f32, f32, f32) -> vector<4xf32>
    %210 = spirv.CompositeConstruct %cst_f32_58, %cst_f32_58, %cst_f32_58, %cst_f32_58 : (f32, f32, f32, f32) -> vector<4xf32>
    %211 = spirv.Select %208, %210, %209 : vector<4xi1>, vector<4xf32>
    %212 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %212, %211 : vector<4xf32>
    %213 = spirv.Load "Function" %27 : vector<4xf64>
    %214 = spirv.FConvert %213 : vector<4xf64> to vector<4xf32>
    %215 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    spirv.Store "Function" %215, %214 : vector<4xf32>
    %216 = spirv.Load "Function" %21 : vector<4xsi32>
    %217 = spirv.ConvertSToF %216 : vector<4xsi32> to vector<4xf64>
    %218 = spirv.Variable : !spirv.ptr<vector<4xf64>, Function>
    spirv.Store "Function" %218, %217 : vector<4xf64>
    %219 = spirv.Load "Function" %23 : vector<4xui32>
    %220 = spirv.ConvertUToF %219 : vector<4xui32> to vector<4xf64>
    %221 = spirv.Variable : !spirv.ptr<vector<4xf64>, Function>
    spirv.Store "Function" %221, %220 : vector<4xf64>
    %222 = spirv.Load "Function" %29 : vector<4xi1>
    %cst_f64_60 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_61 = spirv.Constant 0.000000e+00 : f64
    %223 = spirv.CompositeConstruct %cst_f64_61, %cst_f64_61, %cst_f64_61, %cst_f64_61 : (f64, f64, f64, f64) -> vector<4xf64>
    %224 = spirv.CompositeConstruct %cst_f64_60, %cst_f64_60, %cst_f64_60, %cst_f64_60 : (f64, f64, f64, f64) -> vector<4xf64>
    %225 = spirv.Select %222, %224, %223 : vector<4xi1>, vector<4xf64>
    %226 = spirv.Variable : !spirv.ptr<vector<4xf64>, Function>
    spirv.Store "Function" %226, %225 : vector<4xf64>
    %227 = spirv.Load "Function" %25 : vector<4xf32>
    %228 = spirv.FConvert %227 : vector<4xf32> to vector<4xf64>
    %229 = spirv.Variable : !spirv.ptr<vector<4xf64>, Function>
    spirv.Store "Function" %229, %228 : vector<4xf64>
    %230 = spirv.Load "Function" %21 : vector<4xsi32>
    %cst0_si32_62 = spirv.Constant 0 : si32
    %231 = spirv.CompositeConstruct %cst0_si32_62, %cst0_si32_62, %cst0_si32_62, %cst0_si32_62 : (si32, si32, si32, si32) -> vector<4xsi32>
    %232 = spirv.INotEqual %230, %231 : vector<4xsi32>
    %233 = spirv.Variable : !spirv.ptr<vector<4xi1>, Function>
    spirv.Store "Function" %233, %232 : vector<4xi1>
    %234 = spirv.Load "Function" %23 : vector<4xui32>
    %cst0_ui32_63 = spirv.Constant 0 : ui32
    %235 = spirv.CompositeConstruct %cst0_ui32_63, %cst0_ui32_63, %cst0_ui32_63, %cst0_ui32_63 : (ui32, ui32, ui32, ui32) -> vector<4xui32>
    %236 = spirv.INotEqual %234, %235 : vector<4xui32>
    %237 = spirv.Variable : !spirv.ptr<vector<4xi1>, Function>
    spirv.Store "Function" %237, %236 : vector<4xi1>
    %238 = spirv.Load "Function" %25 : vector<4xf32>
    %cst_f32_64 = spirv.Constant 0.000000e+00 : f32
    %239 = spirv.CompositeConstruct %cst_f32_64, %cst_f32_64, %cst_f32_64, %cst_f32_64 : (f32, f32, f32, f32) -> vector<4xf32>
    %240 = spirv.FOrdNotEqual %238, %239 : vector<4xf32>
    %241 = spirv.Variable : !spirv.ptr<vector<4xi1>, Function>
    spirv.Store "Function" %241, %240 : vector<4xi1>
    %242 = spirv.Load "Function" %27 : vector<4xf64>
    %cst_f64_65 = spirv.Constant 0.000000e+00 : f64
    %243 = spirv.CompositeConstruct %cst_f64_65, %cst_f64_65, %cst_f64_65, %cst_f64_65 : (f64, f64, f64, f64) -> vector<4xf64>
    %244 = spirv.FOrdNotEqual %242, %243 : vector<4xf64>
    %245 = spirv.Variable : !spirv.ptr<vector<4xi1>, Function>
    spirv.Store "Function" %245, %244 : vector<4xi1>
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
