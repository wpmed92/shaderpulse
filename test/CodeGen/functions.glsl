// CHECK: spirv.func @add(%arg0: si32, %arg1: si32) -> si32 "None" {
// CHECK-NEXT: %0 = spirv.IAdd %arg0, %arg1 : si32
// CHECK-NEXT: spirv.ReturnValue %0 : si32
int add(int a, int b) {
    return a + b;
}

// CHECK:  spirv.func @main() "None" {
void main() {
    int a = 1;
    int b = 2;

    // CHECK: %2 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT: %3 = spirv.Load "Function" %1 : si32
    // CHECK-NEXT: %4 = spirv.FunctionCall @add(%2, %3) : (si32, si32) -> si32
    // CHECK-NEXT: %5 = spirv.Load "Function" %0 : si32
    // CHECK-NEXT: %6 = spirv.Load "Function" %1 : si32
    // CHECK-NEXT: %7 = spirv.FunctionCall @add(%5, %6) : (si32, si32) -> si32
    // CHECK-NEXT: %8 = spirv.IMul %4, %7 : si32
    int c = add(a, b) * add(a, b);
}
