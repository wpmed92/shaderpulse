layout(location = 0) out vec4 outColor;

struct B {
  float d;
  float e;
};

struct A {
  float x;
  float y;
  float z;
  B b;
};

void main() {
  A a = A(1.0, 2.0, 3.0, B(1.2, 1.3));
  float test = a.b.d;
  a.b.d = 1.0;
  outColor = vec4(test, 0.0, 0.0, 1.0);
  return;
}