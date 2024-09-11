void main() {
    vec4 test = vec4(1.1, 1.3, 1.5, 1.7);

    vec2 swizz = test.rrrr.ggg.bb;
    float f = swizz.y;
}