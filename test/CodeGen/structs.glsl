struct MyStruct {
  float a;
  int b;
  uint c;
  bool d;
}

struct MyStruct2 {
  MyStruct structMember;
  int b;
}

struct StructWithArr {
  int[4] a;
}

struct Indices {
  int idx1;
  int idx2;
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

    // Struct in struct

    // CHECK: %14 = spirv.CompositeConstruct %cst_f32_0, %cst2_si32_1, %cst3_ui32_2, %true_3 : (f32, si32, ui32, i1) -> !spirv.struct<(f32, si32, ui32, i1)>
    // CHECK-NEXT: %cst1_si32 = spirv.Constant 1 : si32
    // CHECK-NEXT: %15 = spirv.CompositeConstruct %14, %cst1_si32 : (!spirv.struct<(f32, si32, ui32, i1)>, si32) -> !spirv.struct<(!spirv.struct<(f32, si32, ui32, i1)>, si32)>
    MyStruct2 myStruct2 = MyStruct2(MyStruct(0.1, 2, 3u, true), 1);

    // CHECK: %17 = spirv.Load "Function" %16 : !spirv.struct<(!spirv.struct<(f32, si32, ui32, i1)>, si32)>
    // CHECK-NEXT: %cst0_i32_4 = spirv.Constant 0 : i32
    // CHECK-NEXT: %cst3_i32_5 = spirv.Constant 3 : i32
    // CHECK-NEXT: %18 = spirv.CompositeExtract %17[0 : i32, 3 : i32] : !spirv.struct<(!spirv.struct<(f32, si32, ui32, i1)>, si32)>
    d = myStruct2.structMember.d;

    int[2] arr = int[2](1,2);
    Indices idxs = Indices(0,1);
    arr[idxs.idx1] = 12;

    // Struct with array
    // CHECK: %19 = spirv.CompositeConstruct %cst1_si32_6, %cst2_si32_7, %cst3_si32, %cst4_si32 : (si32, si32, si32, si32) -> !spirv.array<4 x si32>
    // CHECK-NEXT: %20 = spirv.CompositeConstruct %19 : (!spirv.array<4 x si32>) -> !spirv.struct<(!spirv.array<4 x si32>)>
    StructWithArr structWithArr = StructWithArr(int[4](1, 2, 3, 4));

    // TODO: This currently fails at the Parser level. Implement member parsing for arrays.
    int arrElemFromStruct = structWithArr.a[3];
}
