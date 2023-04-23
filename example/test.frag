uniform highp float a;
uniform int b;
uint c;
vec3 d;
mat2x2 e;

float foo() {
    return 1.0;
}

float myFunc(vec2 arg1, bool arg2) {
    float f;
    float g;
    f = 1.0;
    g = f + 2.0;
    return g + foo();
}
