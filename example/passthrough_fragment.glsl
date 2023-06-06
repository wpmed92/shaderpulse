layout(location = 0) out vec4 outColor;

struct Light
{
  vec3 eyePosOrDir;
  bool isDirectional;
  vec3 intensity;
  float attenuation;
};

struct Test {
  float a;
  float b;
};

void main() {
  Light light = Light(vec3(1.0, 1.0, 1.0), false, vec3(0.2, 0.3, 0.4), 0.5);
  outColor = vec4(1.0, 0.0, 0.0, 1.0);
  return;
}