void main() {
    // CHECK: %0 = spirv.CompositeConstruct %cst_f32, %cst_f32_0 : (f32, f32) -> vector<2xf32>
    vec2 _vec2 = vec2(1.0, 0.0);

    // CHECK: %2 = spirv.CompositeConstruct %cst_f32_1, %cst_f32_2, %cst_f32_3 : (f32, f32, f32) -> vector<3xf32>
    vec3 _vec3 = vec3(1.0, 1.0, 1.0);

    // vec3 constructors
    // CHECK: %5 = spirv.CompositeConstruct %4, %cst_f32_4 : (vector<2xf32>, f32) -> vector<3xf32>
    vec3 _vec3_2_1 = vec3(_vec2, 1.0);

    // CHECK: %8 = spirv.CompositeConstruct %cst_f32_5, %7 : (f32, vector<2xf32>) -> vector<3xf32>
    vec3 _vec3_1_2 = vec3(1.0, _vec2);

    // CHECK: %11 = spirv.CompositeConstruct %10 : (vector<3xf32>) -> vector<3xf32>
    vec3 _vec3_3 = vec3(vec3(1.0, 1.0, 1.0));

    // vec4 constructors
    // CHECK: %15 = spirv.CompositeConstruct %13, %14 : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
    vec4 _vec4_2_2 = vec4(_vec2, _vec2);

    // CHECK: %18 = spirv.CompositeConstruct %17, %cst_f32_9 : (vector<3xf32>, f32) -> vector<4xf32>
    vec4 _vec4_3_1 = vec4(_vec3, 1.0);

    // CHECK: %21 = spirv.CompositeConstruct %cst_f32_10, %20 : (f32, vector<3xf32>) -> vector<4xf32>
    vec4 _vec4_1_3 = vec4(1.0, _vec3);

    // CHECK: %24 = spirv.CompositeConstruct %cst_f32_11, %23, %cst_f32_12 : (f32, vector<2xf32>, f32) -> vector<4xf32>
    vec4 _vec4_1_2_1 = vec4(1.0, _vec2, 1.0);

    // CHECK: %27 = spirv.CompositeConstruct %cst_f32_13, %cst_f32_14, %26 : (f32, f32, vector<2xf32>) -> vector<4xf32>
    vec4 _vec4_1_1_2 = vec4(1.0, 1.0, _vec2);

    // CHECK: %30 = spirv.CompositeConstruct %29, %cst_f32_15, %cst_f32_16 : (vector<2xf32>, f32, f32) -> vector<4xf32>
    vec4 _vec4_2_1_1 = vec4(_vec2, 1.0, 1.0);

    // CHECK: %33 = spirv.CompositeConstruct %32 : (vector<4xf32>) -> vector<4xf32>
    vec4 _vec4_4 = vec4(vec4(1.0, 1.0, 1.0, 1.0));

    // CHECK: %35 = spirv.CompositeConstruct %cst_f32_21, %cst_f32_22, %cst_f32_23, %cst_f32_24 : (f32, f32, f32, f32) -> vector<4xf32>
    // CHECK-NEXT: %36 = spirv.Variable : !spirv.ptr<vector<4xf32>, Function>
    vec4 _vec4_1 = vec4(1.0, 1.0, 1.0, 1.0);
}
