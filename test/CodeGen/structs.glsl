struct MyStruct {
  float a;
  int b;
  uint c;
  bool d;
}

void main() {
    // CHECK: %0 = spirv.CompositeConstruct %cst_f32, %cst2_si32, %cst3_ui32, %true : (f32, si32, ui32, i1) -> !spirv.struct<(f32, si32, ui32, i1)>
    MyStruct myStruct = MyStruct(0.1, 2, 3u, true);

    // CHECK: %3 = spirv.CompositeExtract %2[0 : i32] : !spirv.struct<(f32, si32, ui32, i1)>
    // CHECK-NEXT: %4 = spirv.Variable : !spirv.ptr<f32, Function>
    // CHECK-NEXT: spirv.Store "Function" %4, %3 : f32
    float a = myStruct.a;

    // CHECK: %6 = spirv.CompositeExtract %5[1 : i32] : !spirv.struct<(f32, si32, ui32, i1)>
    // CHECK-NEXT: %7 = spirv.Variable : !spirv.ptr<si32, Function>
    // CHECK-NEXT: spirv.Store "Function" %7, %6 : si32
    int b = myStruct.b;

    // CHECK: %9 = spirv.CompositeExtract %8[2 : i32] : !spirv.struct<(f32, si32, ui32, i1)>
    // CHECK-NEXT: %10 = spirv.Variable : !spirv.ptr<ui32, Function>
    // CHECK-NEXT: spirv.Store "Function" %10, %9 : ui32
    uint c = myStruct.c;

    // CHECK: %12 = spirv.CompositeExtract %11[3 : i32] : !spirv.struct<(f32, si32, ui32, i1)>
    // CHECK-NEXT: %13 = spirv.Variable : !spirv.ptr<i1, Function>
    // CHECK-NEXT: spirv.Store "Function" %13, %12 : i1
    bool d = myStruct.d;

    return;
}
