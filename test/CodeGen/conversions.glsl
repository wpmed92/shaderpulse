void main() {
    int _int = 1;

    uint _uint = 2u;

    float _float = 1.0;

    double _double = 1.0lf;

    bool _bool = true;

    // CHECK: %6 = spirv.Bitcast %5 : ui32 to si32
    int e = int(_uint);

    // CHECK:  %8 = spirv.Load "Function" %4 : i1
    // CHECK-NEXT: %cst1_si32_0 = spirv.Constant 1 : si32
    // CHECK-NEXT: %cst0_si32 = spirv.Constant 0 : si32
    // CHECK-NEXT: %9 = spirv.Select %8, %cst1_si32_0, %cst0_si32 : i1, si32
    int f = int(_bool);

    // CHECK: %12 = spirv.ConvertFToS %11 : f32 to si32
    int g = int(_float);

    // CHECK: %15 = spirv.ConvertFToS %14 : f64 to si32
    int h = int(_double);


    // CHECK: %18 = spirv.Bitcast %17 : si32 to ui32
    uint i = uint(_int);

    // CHECK: %cst1_ui32 = spirv.Constant 1 : ui32
    // CHECK-NEXT: %cst0_ui32 = spirv.Constant 0 : ui32
    // CHECK-NEXT: %21 = spirv.Select %20, %cst1_ui32, %cst0_ui32 : i1, ui32
    uint j = uint(_bool);

    // CHECK: %24 = spirv.ConvertFToU %23 : f32 to ui32
    uint k = uint(_float);

    // CHECK: %27 = spirv.ConvertFToU %26 : f64 to ui32
    uint l = uint(_double);

    // CHECK: %30 = spirv.ConvertSToF %29 : si32 to f32
    float k = float(_int);

    // CHECK: %33 = spirv.ConvertUToF %32 : ui32 to f32
    float m = float(_uint);

    // float n = float(_bool);

    // CHECK: %36 = spirv.FConvert %35 : f64 to f32
    float o = float(_double);

    // CHECK: %39 = spirv.ConvertSToF %38 : si32 to f64
    double p = double(_int);

    // CHECK: %42 = spirv.ConvertUToF %41 : ui32 to f64
    double q = double(_uint);

    // double r = double(_bool);

    // CHECK: %45 = spirv.FConvert %44 : f32 to f64
    double s = double(_float);

    // CHECK: %cst0_si32_1 = spirv.Constant 0 : si32
    // CHECK-NEXT: %48 = spirv.INotEqual %47, %cst0_si32_1 : si32
    bool t = bool(_int);

    // CHECK: %cst0_ui32_2 = spirv.Constant 0 : ui32
    // CHECK-NEXT: %51 = spirv.INotEqual %50, %cst0_ui32_2 : ui32
    bool u = bool(_uint);

    // CHECK: %cst_f32_3 = spirv.Constant 0.000000e+00 : f32
    // CHECK-NEXT: %54 = spirv.FOrdNotEqual %53, %cst_f32_3 : f32
    bool v = bool(_float);

    // CHECK: %cst_f64_4 = spirv.Constant 0.000000e+00 : f64
    // CHECK-NEXT: %57 = spirv.FOrdNotEqual %56, %cst_f64_4 : f64
    bool w = bool(_double);
}
