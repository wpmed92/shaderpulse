spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
  spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
  spirv.func @main() "None" {
    %cst1_si32 = spirv.Constant 1 : si32
    %cst2_si32 = spirv.Constant 2 : si32
    %0 = spirv.IAdd %cst1_si32, %cst2_si32 : si32
    %1 = spirv.Variable : !spirv.ptr<si32, Function>
    spirv.Store "Function" %1, %0 : si32
    %cst1_ui32 = spirv.Constant 1 : ui32
    %cst2_ui32 = spirv.Constant 2 : ui32
    %2 = spirv.IAdd %cst1_ui32, %cst2_ui32 : ui32
    %3 = spirv.Variable : !spirv.ptr<ui32, Function>
    spirv.Store "Function" %3, %2 : ui32
    %cst_f32 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_0 = spirv.Constant 2.000000e+00 : f32
    %4 = spirv.FAdd %cst_f32, %cst_f32_0 : f32
    %5 = spirv.Variable : !spirv.ptr<f32, Function>
    spirv.Store "Function" %5, %4 : f32
    %cst_f64 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_1 = spirv.Constant 2.000000e+00 : f64
    %6 = spirv.FAdd %cst_f64, %cst_f64_1 : f64
    %7 = spirv.Variable : !spirv.ptr<f64, Function>
    spirv.Store "Function" %7, %6 : f64
    %cst1_si32_2 = spirv.Constant 1 : si32
    %cst2_si32_3 = spirv.Constant 2 : si32
    %8 = spirv.ISub %cst1_si32_2, %cst2_si32_3 : si32
    spirv.Store "Function" %1, %8 : si32
    %cst2_ui32_4 = spirv.Constant 2 : ui32
    %cst1_ui32_5 = spirv.Constant 1 : ui32
    %9 = spirv.ISub %cst2_ui32_4, %cst1_ui32_5 : ui32
    spirv.Store "Function" %3, %9 : ui32
    %cst_f32_6 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_7 = spirv.Constant 2.000000e+00 : f32
    %10 = spirv.FSub %cst_f32_6, %cst_f32_7 : f32
    spirv.Store "Function" %5, %10 : f32
    %cst_f64_8 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_9 = spirv.Constant 2.000000e+00 : f64
    %11 = spirv.FSub %cst_f64_8, %cst_f64_9 : f64
    spirv.Store "Function" %7, %11 : f64
    %cst1_si32_10 = spirv.Constant 1 : si32
    %cst2_si32_11 = spirv.Constant 2 : si32
    %12 = spirv.IMul %cst1_si32_10, %cst2_si32_11 : si32
    spirv.Store "Function" %1, %12 : si32
    %cst2_ui32_12 = spirv.Constant 2 : ui32
    %cst1_ui32_13 = spirv.Constant 1 : ui32
    %13 = spirv.IMul %cst2_ui32_12, %cst1_ui32_13 : ui32
    spirv.Store "Function" %3, %13 : ui32
    %cst_f32_14 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_15 = spirv.Constant 2.000000e+00 : f32
    %14 = spirv.FMul %cst_f32_14, %cst_f32_15 : f32
    spirv.Store "Function" %5, %14 : f32
    %cst_f64_16 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_17 = spirv.Constant 2.000000e+00 : f64
    %15 = spirv.FMul %cst_f64_16, %cst_f64_17 : f64
    spirv.Store "Function" %7, %15 : f64
    %cst1_si32_18 = spirv.Constant 1 : si32
    %cst2_si32_19 = spirv.Constant 2 : si32
    %16 = spirv.SDiv %cst1_si32_18, %cst2_si32_19 : si32
    spirv.Store "Function" %1, %16 : si32
    %cst2_ui32_20 = spirv.Constant 2 : ui32
    %cst1_ui32_21 = spirv.Constant 1 : ui32
    %17 = spirv.UDiv %cst2_ui32_20, %cst1_ui32_21 : ui32
    spirv.Store "Function" %3, %17 : ui32
    %cst_f32_22 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_23 = spirv.Constant 2.000000e+00 : f32
    %18 = spirv.FDiv %cst_f32_22, %cst_f32_23 : f32
    spirv.Store "Function" %5, %18 : f32
    %cst_f64_24 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_25 = spirv.Constant 2.000000e+00 : f64
    %19 = spirv.FDiv %cst_f64_24, %cst_f64_25 : f64
    spirv.Store "Function" %7, %19 : f64
    %cst1_si32_26 = spirv.Constant 1 : si32
    %cst2_si32_27 = spirv.Constant 2 : si32
    %20 = spirv.SRem %cst1_si32_26, %cst2_si32_27 : si32
    spirv.Store "Function" %1, %20 : si32
    %cst2_ui32_28 = spirv.Constant 2 : ui32
    %cst1_ui32_29 = spirv.Constant 1 : ui32
    %21 = spirv.SRem %cst2_ui32_28, %cst1_ui32_29 : ui32
    spirv.Store "Function" %3, %21 : ui32
    %cst_f32_30 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_31 = spirv.Constant 2.000000e+00 : f32
    %22 = spirv.FRem %cst_f32_30, %cst_f32_31 : f32
    spirv.Store "Function" %5, %22 : f32
    %cst_f64_32 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_33 = spirv.Constant 2.000000e+00 : f64
    %23 = spirv.FRem %cst_f64_32, %cst_f64_33 : f64
    spirv.Store "Function" %7, %23 : f64
    %cst1_si32_34 = spirv.Constant 1 : si32
    %cst1_si32_35 = spirv.Constant 1 : si32
    %24 = spirv.IEqual %cst1_si32_34, %cst1_si32_35 : si32
    %25 = spirv.Variable : !spirv.ptr<i1, Function>
    spirv.Store "Function" %25, %24 : i1
    %cst1_ui32_36 = spirv.Constant 1 : ui32
    %cst1_ui32_37 = spirv.Constant 1 : ui32
    %26 = spirv.IEqual %cst1_ui32_36, %cst1_ui32_37 : ui32
    spirv.Store "Function" %25, %26 : i1
    %cst_f32_38 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_39 = spirv.Constant 1.000000e+00 : f32
    %27 = spirv.FOrdEqual %cst_f32_38, %cst_f32_39 : f32
    spirv.Store "Function" %25, %27 : i1
    %cst_f64_40 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_41 = spirv.Constant 1.000000e+00 : f64
    %28 = spirv.FOrdEqual %cst_f64_40, %cst_f64_41 : f64
    spirv.Store "Function" %25, %28 : i1
    %cst1_si32_42 = spirv.Constant 1 : si32
    %cst1_si32_43 = spirv.Constant 1 : si32
    %29 = spirv.INotEqual %cst1_si32_42, %cst1_si32_43 : si32
    spirv.Store "Function" %25, %29 : i1
    %cst1_ui32_44 = spirv.Constant 1 : ui32
    %cst1_ui32_45 = spirv.Constant 1 : ui32
    %30 = spirv.INotEqual %cst1_ui32_44, %cst1_ui32_45 : ui32
    spirv.Store "Function" %25, %30 : i1
    %cst_f32_46 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_47 = spirv.Constant 1.000000e+00 : f32
    %31 = spirv.FOrdNotEqual %cst_f32_46, %cst_f32_47 : f32
    spirv.Store "Function" %25, %31 : i1
    %cst_f64_48 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_49 = spirv.Constant 1.000000e+00 : f64
    %32 = spirv.FOrdNotEqual %cst_f64_48, %cst_f64_49 : f64
    spirv.Store "Function" %25, %32 : i1
    %cst1_si32_50 = spirv.Constant 1 : si32
    %cst1_si32_51 = spirv.Constant 1 : si32
    %33 = spirv.SLessThan %cst1_si32_50, %cst1_si32_51 : si32
    spirv.Store "Function" %25, %33 : i1
    %cst1_ui32_52 = spirv.Constant 1 : ui32
    %cst1_ui32_53 = spirv.Constant 1 : ui32
    %34 = spirv.ULessThan %cst1_ui32_52, %cst1_ui32_53 : ui32
    spirv.Store "Function" %25, %34 : i1
    %cst_f32_54 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_55 = spirv.Constant 1.000000e+00 : f32
    %35 = spirv.FOrdLessThan %cst_f32_54, %cst_f32_55 : f32
    spirv.Store "Function" %25, %35 : i1
    %cst_f64_56 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_57 = spirv.Constant 1.000000e+00 : f64
    %36 = spirv.FOrdLessThan %cst_f64_56, %cst_f64_57 : f64
    spirv.Store "Function" %25, %36 : i1
    %cst1_si32_58 = spirv.Constant 1 : si32
    %cst1_si32_59 = spirv.Constant 1 : si32
    %37 = spirv.SLessThanEqual %cst1_si32_58, %cst1_si32_59 : si32
    spirv.Store "Function" %25, %37 : i1
    %cst1_ui32_60 = spirv.Constant 1 : ui32
    %cst1_ui32_61 = spirv.Constant 1 : ui32
    %38 = spirv.ULessThanEqual %cst1_ui32_60, %cst1_ui32_61 : ui32
    spirv.Store "Function" %25, %38 : i1
    %cst_f32_62 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_63 = spirv.Constant 1.000000e+00 : f32
    %39 = spirv.FOrdLessThanEqual %cst_f32_62, %cst_f32_63 : f32
    spirv.Store "Function" %25, %39 : i1
    %cst_f64_64 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_65 = spirv.Constant 1.000000e+00 : f64
    %40 = spirv.FOrdLessThanEqual %cst_f64_64, %cst_f64_65 : f64
    spirv.Store "Function" %25, %40 : i1
    %cst1_si32_66 = spirv.Constant 1 : si32
    %cst1_si32_67 = spirv.Constant 1 : si32
    %41 = spirv.SGreaterThan %cst1_si32_66, %cst1_si32_67 : si32
    spirv.Store "Function" %25, %41 : i1
    %cst1_ui32_68 = spirv.Constant 1 : ui32
    %cst1_ui32_69 = spirv.Constant 1 : ui32
    %42 = spirv.UGreaterThan %cst1_ui32_68, %cst1_ui32_69 : ui32
    spirv.Store "Function" %25, %42 : i1
    %cst_f32_70 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_71 = spirv.Constant 1.000000e+00 : f32
    %43 = spirv.FOrdGreaterThan %cst_f32_70, %cst_f32_71 : f32
    spirv.Store "Function" %25, %43 : i1
    %cst_f64_72 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_73 = spirv.Constant 1.000000e+00 : f64
    %44 = spirv.FOrdGreaterThan %cst_f64_72, %cst_f64_73 : f64
    spirv.Store "Function" %25, %44 : i1
    %cst1_si32_74 = spirv.Constant 1 : si32
    %cst1_si32_75 = spirv.Constant 1 : si32
    %45 = spirv.SGreaterThanEqual %cst1_si32_74, %cst1_si32_75 : si32
    spirv.Store "Function" %25, %45 : i1
    %cst1_ui32_76 = spirv.Constant 1 : ui32
    %cst1_ui32_77 = spirv.Constant 1 : ui32
    %46 = spirv.UGreaterThanEqual %cst1_ui32_76, %cst1_ui32_77 : ui32
    spirv.Store "Function" %25, %46 : i1
    %cst_f32_78 = spirv.Constant 1.000000e+00 : f32
    %cst_f32_79 = spirv.Constant 1.000000e+00 : f32
    %47 = spirv.FOrdGreaterThanEqual %cst_f32_78, %cst_f32_79 : f32
    spirv.Store "Function" %25, %47 : i1
    %cst_f64_80 = spirv.Constant 1.000000e+00 : f64
    %cst_f64_81 = spirv.Constant 1.000000e+00 : f64
    %48 = spirv.FOrdGreaterThanEqual %cst_f64_80, %cst_f64_81 : f64
    spirv.Store "Function" %25, %48 : i1
    %true = spirv.Constant true
    %49 = spirv.Variable : !spirv.ptr<i1, Function>
    spirv.Store "Function" %49, %true : i1
    %false = spirv.Constant false
    %50 = spirv.Variable : !spirv.ptr<i1, Function>
    spirv.Store "Function" %50, %false : i1
    %51 = spirv.Load "Function" %50 : i1
    %52 = spirv.Load "Function" %49 : i1
    %53 = spirv.LogicalAnd %52, %51 : i1
    spirv.Store "Function" %25, %53 : i1
    %54 = spirv.Load "Function" %50 : i1
    %55 = spirv.Load "Function" %49 : i1
    %56 = spirv.LogicalOr %55, %54 : i1
    spirv.Store "Function" %25, %56 : i1
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
