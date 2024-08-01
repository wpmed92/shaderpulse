layout(location = 0) out vec4 outColor;

struct Test {
  float a;
  float b;
}
void main() {
  Test t = Test(1.0, 1.0);
  outColor = vec4(t.a, 0.0, 0.0, 1.0);
  return;
}
