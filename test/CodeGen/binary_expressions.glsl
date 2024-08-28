void main() {
    // ADD

    // CHECK: %0 = spirv.IAdd %cst1_si32, %cst2_si32 : si3
    int a = 1 + 2;

    // CHECK: %2 = spirv.IAdd %cst1_ui32, %cst2_ui32 : ui32
    uint b = 1u + 2u;

    // CHECK: %4 = spirv.FAdd %cst_f32, %cst_f32_0 : f32
    float c = 1.0f + 2.0f;

    // CHECK: %6 = spirv.FAdd %cst_f64, %cst_f64_1 : f64
    double d = 1.0lf + 2.0lf;

    // SUB

    // CHECK: %8 = spirv.ISub %cst1_si32_2, %cst2_si32_3 : si32
    a = 1 - 2;

    // CHECK: %9 = spirv.ISub %cst2_ui32_4, %cst1_ui32_5 : ui32
    b = 2u - 1u;

    // CHECK: %10 = spirv.FSub %cst_f32_6, %cst_f32_7 : f32
    c = 1.0f - 2.0f;

    // CHECK: %11 = spirv.FSub %cst_f64_8, %cst_f64_9 : f64
    d = 1.0lf - 2.0lf;

    // MUL

    // CHECK: %12 = spirv.IMul %cst1_si32_10, %cst2_si32_11 : si32
    a = 1 * 2;

    // CHECK: %13 = spirv.IMul %cst2_ui32_12, %cst1_ui32_13 : ui32
    b = 2u * 1u;

    // CHECK: %14 = spirv.FMul %cst_f32_14, %cst_f32_15 : f32
    c = 1.0f * 2.0f;

    // CHECK: %15 = spirv.FMul %cst_f64_16, %cst_f64_17 : f64
    d = 1.0lf * 2.0lf;

    // DIV

    // CHECK: %16 = spirv.SDiv %cst1_si32_18, %cst2_si32_19 : si32
    a = 1 / 2;

    // CHECK: %17 = spirv.UDiv %cst2_ui32_20, %cst1_ui32_21 : ui32
    b = 2u / 1u;

    // CHECK: %18 = spirv.FDiv %cst_f32_22, %cst_f32_23 : f32
    c = 1.0f / 2.0f;

    // CHECK: %19 = spirv.FDiv %cst_f64_24, %cst_f64_25 : f64
    d = 1.0lf / 2.0lf;

    // MOD

    // CHECK: %20 = spirv.SRem %cst1_si32_26, %cst2_si32_27 : si32
    a = 1 % 2;

    // CHECK: %21 = spirv.SRem %cst2_ui32_28, %cst1_ui32_29 : ui32
    b = 2u % 1u;

    // CHECK: %22 = spirv.FRem %cst_f32_30, %cst_f32_31 : f32
    c = 1.0f % 2.0f;

    // CHECK: %23 = spirv.FRem %cst_f64_32, %cst_f64_33 : f64
    d = 1.0lf % 2.0lf;

    // EQ

    // CHECK: %24 = spirv.IEqual %cst1_si32_34, %cst1_si32_35 : si32
    bool e = 1 == 1;

    // CHECK: %26 = spirv.IEqual %cst1_ui32_36, %cst1_ui32_37 : ui32
    e = 1u == 1u;

    // CHECK: %27 = spirv.FOrdEqual %cst_f32_38, %cst_f32_39 : f32
    e = 1.0f == 1.0f;

    // CHECK: %28 = spirv.FOrdEqual %cst_f64_40, %cst_f64_41 : f64
    e = 1.0lf == 1.0lf;

    // NEQ

    // CHECK: %29 = spirv.INotEqual %cst1_si32_42, %cst1_si32_43 : si32
    e = 1 != 1;

    // CHECK: %30 = spirv.INotEqual %cst1_ui32_44, %cst1_ui32_45 : ui32
    e = 1u != 1u;

    // CHECK: %31 = spirv.FOrdNotEqual %cst_f32_46, %cst_f32_47 : f32
    e = 1.0f != 1.0f;

    // CHECK: %32 = spirv.FOrdNotEqual %cst_f64_48, %cst_f64_49 : f64
    e = 1.0lf != 1.0lf;

    // LT

    // CHECK: %33 = spirv.SLessThan %cst1_si32_50, %cst1_si32_51 : si32
    e = 1 < 1;

    // CHECK: %34 = spirv.ULessThan %cst1_ui32_52, %cst1_ui32_53 : ui32
    e = 1u < 1u;

    // CHECK: %35 = spirv.FOrdLessThan %cst_f32_54, %cst_f32_55 : f32
    e = 1.0f < 1.0f;

    // CHECK:  %36 = spirv.FOrdLessThan %cst_f64_56, %cst_f64_57 : f64
    e = 1.0lf < 1.0lf;

    // LTEQ

    // CHECK: %37 = spirv.SLessThanEqual %cst1_si32_58, %cst1_si32_59 : si32
    e = 1 <= 1;

    // CHECK: %38 = spirv.ULessThanEqual %cst1_ui32_60, %cst1_ui32_61 : ui32
    e = 1u <= 1u;

    // CHECK: %39 = spirv.FOrdLessThanEqual %cst_f32_62, %cst_f32_63 : f32
    e = 1.0f <= 1.0f;

    // CHECK: %40 = spirv.FOrdLessThanEqual %cst_f64_64, %cst_f64_65 : f64
    e = 1.0lf <= 1.0lf;

    // GT

    // CHECK: %41 = spirv.SGreaterThan %cst1_si32_66, %cst1_si32_67 : si32
    e = 1 > 1;

    // CHECK: %42 = spirv.UGreaterThan %cst1_ui32_68, %cst1_ui32_69 : ui32
    e = 1u > 1u;

    // CHECK: %43 = spirv.FOrdGreaterThan %cst_f32_70, %cst_f32_71 : f32
    e = 1.0f > 1.0f;

    // CHECK: %44 = spirv.FOrdGreaterThan %cst_f64_72, %cst_f64_73 : f64
    e = 1.0lf > 1.0lf;

    // GTEQ

    // CHECK: %45 = spirv.SGreaterThanEqual %cst1_si32_74, %cst1_si32_75 : si32
    e = 1 >= 1;

    // CHECK: %46 = spirv.UGreaterThanEqual %cst1_ui32_76, %cst1_ui32_77 : ui32
    e = 1u >= 1u;

    // CHECK: %47 = spirv.FOrdGreaterThanEqual %cst_f32_78, %cst_f32_79 : f32
    e = 1.0f >= 1.0f;

    // CHECK: %48 = spirv.FOrdGreaterThanEqual %cst_f64_80, %cst_f64_81 : f64
    e = 1.0lf >= 1.0lf;

    // AND

    bool f = true;
    bool g = false;

    // CHECK: %53 = spirv.LogicalAnd %52, %51 : i1
    e = f && g;

    // OR

    // CHECK: %56 = spirv.LogicalOr %55, %54 : i1
    e = f || g;
}
