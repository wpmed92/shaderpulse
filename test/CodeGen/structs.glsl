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
    // CHECK-NEXT: %1 = spirv.Variable : !spirv.ptr<!spirv.struct<(f32, si32, ui32, i1)>, Function>
    MyStruct myStruct = MyStruct(0.1, 2, 3u, true);

    // Basic member access

    // CHECK: %cst0_i32 = spirv.Constant 0 : i32
    // CHECK-NEXT: %2 = spirv.AccessChain %1[%cst0_i32] : !spirv.ptr<!spirv.struct<(f32, si32, ui32, i1)>, Function>, i32
    float a = myStruct.a;

    // CHECK: %cst1_i32 = spirv.Constant 1 : i32
    // CHECK-NEXT: %5 = spirv.AccessChain %1[%cst1_i32] : !spirv.ptr<!spirv.struct<(f32, si32, ui32, i1)>, Function>, i32
    int b = myStruct.b;

    // CHECK: %cst2_i32 = spirv.Constant 2 : i32
    // CHECK-NEXT: %8 = spirv.AccessChain %1[%cst2_i32] : !spirv.ptr<!spirv.struct<(f32, si32, ui32, i1)>, Function>, i32
    uint c = myStruct.c;

    // CHECK: %cst3_i32 = spirv.Constant 3 : i32
    // CHECK-NEXT: %11 = spirv.AccessChain %1[%cst3_i32] : !spirv.ptr<!spirv.struct<(f32, si32, ui32, i1)>, Function>, i32
    bool d = myStruct.d;

    // Struct in struct

    // CHECK: %14 = spirv.CompositeConstruct %cst_f32_0, %cst2_si32_1, %cst3_ui32_2, %true_3 : (f32, si32, ui32, i1) -> !spirv.struct<(f32, si32, ui32, i1)>
    // CHECK-NEXT: %cst1_si32 = spirv.Constant 1 : si32
    // CHECK-NEXT: %15 = spirv.CompositeConstruct %14, %cst1_si32 : (!spirv.struct<(f32, si32, ui32, i1)>, si32) -> !spirv.struct<(!spirv.struct<(f32, si32, ui32, i1)>, si32)>
    MyStruct2 myStruct2 = MyStruct2(MyStruct(0.1, 2, 3u, true), 1);

    // CHECK: %cst0_i32_4 = spirv.Constant 0 : i32
    // CHECK-NEXT: %cst3_i32_5 = spirv.Constant 3 : i32
    // CHECK-NEXT: %17 = spirv.AccessChain %16[%cst0_i32_4, %cst3_i32_5] : !spirv.ptr<!spirv.struct<(!spirv.struct<(f32, si32, ui32, i1)>, si32)>, Function>, i32, i32
    d = myStruct2.structMember.d;

    // Struct with array

    // CHECK: %19 = spirv.CompositeConstruct %cst1_si32_6, %cst2_si32_7, %cst3_si32, %cst4_si32 : (si32, si32, si32, si32) -> !spirv.array<4 x si32>
    // CHECK-NEXT: %20 = spirv.CompositeConstruct %19 : (!spirv.array<4 x si32>) -> !spirv.struct<(!spirv.array<4 x si32>)>
    StructWithArr structWithArr = StructWithArr(int[4](1, 2, 3, 4));

    // CHECK: %cst0_i32_8 = spirv.Constant 0 : i32
    // CHECK-NEXT: %cst2_si32_9 = spirv.Constant 2 : si32
    // CHECK-NEXT: %22 = spirv.AccessChain %21[%cst0_i32_8, %cst2_si32_9] : !spirv.ptr<!spirv.struct<(!spirv.array<4 x si32>)>, Function>, i32, si32
    int arrElemFromStruct = structWithArr.a[2];

    // Member access as array index

    // CHECK: %25 = spirv.CompositeConstruct %cst1_si32_10, %cst2_si32_11 : (si32, si32) -> !spirv.array<2 x si32>
    // CHECK-NEXT: %26 = spirv.Variable : !spirv.ptr<!spirv.array<2 x si32>, Function>
    int[2] arr = int[2](1, 2);

    // CHECK: %28 = spirv.Variable : !spirv.ptr<!spirv.struct<(si32, si32)>, Function>
    Indices indices = Indices(0, 1);

    // CHECK: %cst1_i32_13 = spirv.Constant 1 : i32
    // CHECK-NEXT: %29 = spirv.AccessChain %28[%cst1_i32_13] : !spirv.ptr<!spirv.struct<(si32, si32)>, Function>, i32
    // CHECK-NEXT:%30 = spirv.Load "Function" %29 : si32
    // CHECK-NEXT:%31 = spirv.AccessChain %26[%30] : !spirv.ptr<!spirv.array<2 x si32>, Function>, si32
    arr[indices.idx2] = 24;
}
