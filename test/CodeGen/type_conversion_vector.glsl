void main() {
    ivec2 _ivec2 = ivec2(1, 2);
    uvec2 _uvec2 = uvec2(3u, 4u);
    vec2 _vec2 = vec2(1.0, 2.0);
    dvec2 _dvec2 = dvec2(1.0lf, 2.0lf);
    bvec2 _bvec2 = bvec2(true, false);

    ivec3 _ivec3 = ivec3(1, 2, 3);
    uvec3 _uvec3 = uvec3(3u, 4u, 5u);
    vec3 _vec3 = vec3(1.0, 2.0, 3.0);
    dvec3 _dvec3 = dvec3(1.0lf, 2.0lf, 3.0lf);
    bvec3 _bvec3 = bvec3(true, false, true);

    ivec4 _ivec4 = ivec4(1, 2, 3, 4);
    uvec4 _uvec4 = uvec4(3u, 4u, 5u, 6u);
    vec4 _vec4 = vec4(1.0, 2.0, 3.0, 4.0);
    dvec4 _dvec4 = dvec4(1.0lf, 2.0lf, 3.0lf, 4.0lf);
    bvec4 _bvec4 = bvec4(true, false, true, false);

    // vec2
    // CHECK:  %31 = spirv.Bitcast %30 : vector<2xui32> to vector<2xsi32>
    ivec2 a = ivec2(_uvec2);

    // CHECK: %cst1_si32_33 = spirv.Constant 1 : si32
    // CHECK-NEXT: %cst0_si32 = spirv.Constant 0 : si32
    // CHECK-NEXT: %34 = spirv.CompositeConstruct %cst0_si32, %cst0_si32 : (si32, si32) -> vector<2xsi32>
    // CHECK-NEXT: %35 = spirv.CompositeConstruct %cst1_si32_33, %cst1_si32_33 : (si32, si32) -> vector<2xsi32>
    // CHECK-NEXT: %36 = spirv.Select %33, %35, %34 : vector<2xi1>, vector<2xsi32>
    ivec2 b = ivec2(_bvec2);

    // CHECK: %39 = spirv.ConvertFToS %38 : vector<2xf32> to vector<2xsi32>
    ivec2 c = ivec2(_vec2);

    // CHECK: %42 = spirv.ConvertFToS %41 : vector<2xf64> to vector<2xsi32>
    ivec2 d = ivec2(_dvec2);

    // CHECK: %45 = spirv.Bitcast %44 : vector<2xsi32> to vector<2xui32>
    uvec2 e = uvec2(_ivec2);

    // CHECK: %cst1_ui32 = spirv.Constant 1 : ui32
    // CHECK-NEXT: %cst0_ui32 = spirv.Constant 0 : ui32
    // CHECK-NEXT: %48 = spirv.CompositeConstruct %cst0_ui32, %cst0_ui32 : (ui32, ui32) -> vector<2xui32>
    // CHECK-NEXT: %49 = spirv.CompositeConstruct %cst1_ui32, %cst1_ui32 : (ui32, ui32) -> vector<2xui32>
    // CHECK-NEXT: %50 = spirv.Select %47, %49, %48 : vector<2xi1>, vector<2xui32>
    uvec2 f = uvec2(_bvec2);

    // CHECK: %53 = spirv.ConvertFToU %52 : vector<2xf32> to vector<2xui32>
    uvec2 g = uvec2(_vec2);

    // CHECK: %56 = spirv.ConvertFToU %55 : vector<2xf64> to vector<2xui32>
    uvec2 h = uvec2(_dvec2);

    // CHECK: %59 = spirv.ConvertSToF %58 : vector<2xsi32> to vector<2xf32>
    vec2 i = vec2(_ivec2);

    // CHECK: %62 = spirv.ConvertUToF %61 : vector<2xui32> to vector<2xf32>
    vec2 j = vec2(_uvec2);

    // CHECK: %cst_f32_34 = spirv.Constant 1.000000e+00 : f32
    // CHECK-NEXT: %cst_f32_35 = spirv.Constant 0.000000e+00 : f32
    // CHECK-NEXT: %65 = spirv.CompositeConstruct %cst_f32_35, %cst_f32_35 : (f32, f32) -> vector<2xf32>
    // CHECK-NEXT: %66 = spirv.CompositeConstruct %cst_f32_34, %cst_f32_34 : (f32, f32) -> vector<2xf32>
    // CHECK-NEXT: %67 = spirv.Select %64, %66, %65 : vector<2xi1>, vector<2xf32>
    vec2 k = vec2(_bvec2);

    // CHECK: %70 = spirv.FConvert %69 : vector<2xf64> to vector<2xf32>
    vec2 l = vec2(_dvec2);

    // CHECK: %73 = spirv.ConvertSToF %72 : vector<2xsi32> to vector<2xf64>
    dvec2 m = dvec2(_ivec2);

    // CHECK: %76 = spirv.ConvertUToF %75 : vector<2xui32> to vector<2xf64>
    dvec2 n = dvec2(_uvec2);

    // CHECK: %cst_f64_36 = spirv.Constant 1.000000e+00 : f64
    // CHECK-NEXT: %cst_f64_37 = spirv.Constant 0.000000e+00 : f64
    // CHECK-NEXT: %79 = spirv.CompositeConstruct %cst_f64_37, %cst_f64_37 : (f64, f64) -> vector<2xf64>
    // CHECK-NEXT: %80 = spirv.CompositeConstruct %cst_f64_36, %cst_f64_36 : (f64, f64) -> vector<2xf64>
    // CHECK-NEXT: %81 = spirv.Select %78, %80, %79 : vector<2xi1>, vector<2xf64>
    dvec2 o = dvec2(_bvec2);

    // CHECK: %84 = spirv.FConvert %83 : vector<2xf32> to vector<2xf64>
    dvec2 p = dvec2(_vec2);

    bvec2 q = bvec2(_ivec2);
    bvec2 r = bvec2(_uvec2);
    bvec2 s = bvec2(_vec2);
    bvec2 t = bvec2(_dvec2);

    // vec3
    ivec3 u = ivec3(_uvec3);
    ivec3 v = ivec3(_bvec3);
    ivec3 w = ivec3(_vec3);
    ivec3 x = ivec3(_dvec3);

    uvec3 y = uvec3(_ivec3);
    uvec3 z = uvec3(_bvec3);
    uvec3 aa = uvec3(_vec3);
    uvec3 bb = uvec3(_dvec3);

    vec3 cc = vec3(_ivec3);
    vec3 dd = vec3(_uvec3);
    vec3 ee = vec3(_bvec3);
    vec3 ff = vec3(_dvec3);

    dvec3 gg = dvec3(_ivec3);
    dvec3 hh = dvec3(_uvec3);
    dvec3 ii = dvec3(_bvec3);
    dvec3 jj = dvec3(_vec3);

    bvec3 kk = bvec3(_ivec3);
    bvec3 ll = bvec3(_uvec3);
    bvec3 mm = bvec3(_vec3);
    bvec3 nn = bvec3(_dvec3);

    // vec4
    ivec4 oo = ivec4(_uvec4);
    ivec4 pp = ivec4(_bvec4);
    ivec4 qq = ivec4(_vec4);
    ivec4 rr = ivec4(_dvec4);

    uvec4 ss = uvec4(_ivec4);
    uvec4 tt = uvec4(_bvec4);
    uvec4 uu = uvec4(_vec4);
    uvec4 vv = uvec4(_dvec4);

    vec4 ww = vec4(_ivec4);
    vec4 xx = vec4(_uvec4);
    vec4 yy = vec4(_bvec4);
    vec4 zz = vec4(_dvec4);

    dvec4 aaa = dvec4(_ivec4);
    dvec4 bbb = dvec4(_uvec4);
    dvec4 ccc = dvec4(_bvec4);
    dvec4 ddd = dvec4(_vec4);

    bvec4 eee = bvec4(_ivec4);
    bvec4 fff = bvec4(_uvec4);
    bvec4 ggg = bvec4(_vec4);
    bvec4 hhh = bvec4(_dvec4);
}
