layout(location = 0) out vec4 outColor;

// Demonstrate struct handling
struct Color {
  float r;
  float g;
  float b;
  float a;
}

void main() {
  Color color = Color(1.0, 0.0, 0.0, 1.0);
  outColor = vec4(color.r, color.g, color.b, color.a);
  return;
}
