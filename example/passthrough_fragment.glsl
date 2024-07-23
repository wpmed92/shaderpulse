layout(location = 0) out vec4 outColor;

struct B {
  float d;
  float e;
};

// This is a commment and should be skipped
struct A {
  float x;
  float y;
  float z;
  B b;
};

/* This is a multine comment, and this should
   also be removed as part of the preprocessing*/
void main() {
  A a = A(1.0, 2.0, 3.0, B(1.2, 1.3));
  float test = a.b.d;
  bool cond = true;
  float condTruePart = 1.0 + 2.0;
  float condFalsePart = 2.0;
  float ternaryTest = cond ? condTruePart : condFalsePart;
  int decl1=0,decl2,decl3;
  float[2] arr = float[2](1.0, 2.0);
  float testAccess = arr[0];
  // No codegen yet of initializers
  // vec3 testInitExp = { 0.0, 1.0, 2.0 };

  for (int i = 0; i < 10; i++) {
    int j = 1;
  }

  a.b.d = 1.0;
  outColor = vec4(test, 0.0, 0.0, 1.0);
  return;
}