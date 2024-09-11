void main() {
    vec2 _vec2 = vec2(0.1, 0.2);

    vec3 _vec3 = vec3(1.1, 1.2, 1.3);

    vec4 _vec4 = vec4(3.4, 4.8, 5.6, 6.7);

    // CHECK: %7 = spirv.VectorShuffle [0 : i32, 1 : i32] %6 : vector<2xf32>, %6 : vector<2xf32> -> vector<2xf32>
    vec2 _swizz_vec2 = _vec2.xy;

    // CHECK: %10 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32] %9 : vector<3xf32>, %9 : vector<3xf32> -> vector<3xf32>
    vec3 _swizz_vec3 = _vec3.xyz;

    // CHECK: %13 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32] %12 : vector<4xf32>, %12 : vector<4xf32> -> vector<4xf32>
    vec4 _swizz_vec4 = _vec4.xyzw;

    // CHECK: %16 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32] %15 : vector<4xf32>, %15 : vector<4xf32> -> vector<4xf32>
    // CHECK-NEXT: %17 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32] %16 : vector<4xf32>, %16 : vector<4xf32> -> vector<3xf32>
    // CHECK-NEXT: %18 = spirv.VectorShuffle [0 : i32, 1 : i32] %17 : vector<3xf32>, %17 : vector<3xf32> -> vector<2xf32>
    vec2 _swizz_chain = _vec4.xyzw.xyz.xy;

    // CHECK: %21 = spirv.CompositeExtract %20[0 : i32] : vector<3xf32>
    float single_elem = _vec3.x;

    // CHECK: %24 = spirv.CompositeExtract %23[1 : i32] : vector<3xf32>
    single_elem = _vec3.y;

    // CHECK: %26 = spirv.CompositeExtract %25[2 : i32] : vector<3xf32>
    single_elem = _vec3.z;

    // CHECK: %28 = spirv.CompositeExtract %27[3 : i32] : vector<4xf32>
    single_elem = _vec4.a;

    // CHECK: %30 = spirv.VectorShuffle [2 : i32, 1 : i32, 0 : i32] %29 : vector<4xf32>, %29 : vector<4xf32> -> vector<3xf32>
    _swizz_vec3 = _vec4.bgr;
}