void main() {
    int a = 1;

    uint b = 2u;

    float c = 1.0;

    double d = 1.0lf;

    bool e = true;

    // CHECK: %6 = spirv.Bitcast %5 : ui32 to si32
    //      int(uint)
    int e = int(b);

    // CHECK:  %8 = spirv.Load "Function" %4 : i1
    // CHECK-NEXT: %cst1_si32_0 = spirv.Constant 1 : si32
    // CHECK-NEXT: %cst0_si32 = spirv.Constant 0 : si32
    // CHECK-NEXT: %9 = spirv.Select %8, %cst1_si32_0, %cst0_si32 : i1, si32
    //      int(bool)
    int f = int(e);

    // CHECK: %12 = spirv.ConvertFToS %11 : f32 to si32
    //      int(float)
    int g = int(c);

    // CHECK: %15 = spirv.ConvertFToS %14 : f64 to si32
    //      int(double)
    int h = int(d);



    //       uint(int)
    uint i = uint(a);

    //       uint(bool)
    uint j = uint(e);

    //       uint(float)
    uint k = uint(c);

    //       uint(double)
    uint l = uint(d);
}
