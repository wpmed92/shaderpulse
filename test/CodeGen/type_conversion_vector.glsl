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

    // CHECK: %cst0_si32_38 = spirv.Constant 0 : si32
    // CHECK-NEXT: %87 = spirv.CompositeConstruct %cst0_si32_38, %cst0_si32_38 : (si32, si32) -> vector<2xsi32>
    // CHECK-NEXT: %88 = spirv.INotEqual %86, %87 : vector<2xsi32>
    bvec2 q = bvec2(_ivec2);

    // CHECK: %cst0_ui32_39 = spirv.Constant 0 : ui32
    // CHECK-NEXT: %91 = spirv.CompositeConstruct %cst0_ui32_39, %cst0_ui32_39 : (ui32, ui32) -> vector<2xui32>
    // CHECK-NEXT: %92 = spirv.INotEqual %90, %91 : vector<2xui32>
    bvec2 r = bvec2(_uvec2);

    // CHECK: %cst_f32_40 = spirv.Constant 0.000000e+00 : f32
    // CHECK-NEXT: %95 = spirv.CompositeConstruct %cst_f32_40, %cst_f32_40 : (f32, f32) -> vector<2xf32>
    // CHECK-NEXT: %96 = spirv.FOrdNotEqual %94, %95 : vector<2xf32>
    bvec2 s = bvec2(_vec2);

    // CHECK: %cst_f64_41 = spirv.Constant 0.000000e+00 : f64
    // CHECK-NEXT: %99 = spirv.CompositeConstruct %cst_f64_41, %cst_f64_41 : (f64, f64) -> vector<2xf64>
    // CHECK-NEXT: %100 = spirv.FOrdNotEqual %98, %99 : vector<2xf64>
    bvec2 t = bvec2(_dvec2);

    // vec3
    // CHECK: %103 = spirv.Bitcast %102 : vector<3xui32> to vector<3xsi32>
    ivec3 u = ivec3(_uvec3);

    // CHECK: %cst1_si32_42 = spirv.Constant 1 : si32
    // CHECK-NEXT: %cst0_si32_43 = spirv.Constant 0 : si32
    // CHECK-NEXT: %106 = spirv.CompositeConstruct %cst0_si32_43, %cst0_si32_43, %cst0_si32_43 : (si32, si32, si32) -> vector<3xsi32>
    // CHECK-NEXT: %107 = spirv.CompositeConstruct %cst1_si32_42, %cst1_si32_42, %cst1_si32_42 : (si32, si32, si32) -> vector<3xsi32>
    // CHECK-NEXT: %108 = spirv.Select %105, %107, %106 : vector<3xi1>, vector<3xsi32>
    ivec3 v = ivec3(_bvec3);

    // CHECK: %111 = spirv.ConvertFToS %110 : vector<3xf32> to vector<3xsi32>
    ivec3 w = ivec3(_vec3);

    // CHECK: %114 = spirv.ConvertFToS %113 : vector<3xf64> to vector<3xsi32>
    ivec3 x = ivec3(_dvec3);

    // CHECK: %117 = spirv.Bitcast %116 : vector<3xsi32> to vector<3xui32>
    uvec3 y = uvec3(_ivec3);

    // CHECK: %cst1_ui32_44 = spirv.Constant 1 : ui32
    // CHECK-NEXT: %cst0_ui32_45 = spirv.Constant 0 : ui32
    // CHECK-NEXT: %120 = spirv.CompositeConstruct %cst0_ui32_45, %cst0_ui32_45, %cst0_ui32_45 : (ui32, ui32, ui32) -> vector<3xui32>
    // CHECK-NEXT: %121 = spirv.CompositeConstruct %cst1_ui32_44, %cst1_ui32_44, %cst1_ui32_44 : (ui32, ui32, ui32) -> vector<3xui32>
    // CHECK-NEXT: %122 = spirv.Select %119, %121, %120 : vector<3xi1>, vector<3xui32>
    uvec3 z = uvec3(_bvec3);

    // CHECK: %125 = spirv.ConvertFToU %124 : vector<3xf32> to vector<3xui32>
    uvec3 aa = uvec3(_vec3);

    // CHECK: %128 = spirv.ConvertFToU %127 : vector<3xf64> to vector<3xui32>
    uvec3 bb = uvec3(_dvec3);

    // CHECK: %131 = spirv.ConvertSToF %130 : vector<3xsi32> to vector<3xf32>
    vec3 cc = vec3(_ivec3);

    // CHECK: %134 = spirv.ConvertUToF %133 : vector<3xui32> to vector<3xf32>
    vec3 dd = vec3(_uvec3);

    // CHECK: %cst_f32_46 = spirv.Constant 1.000000e+00 : f32
    // CHECK-NEXT: %cst_f32_47 = spirv.Constant 0.000000e+00 : f32
    // CHECK-NEXT: %137 = spirv.CompositeConstruct %cst_f32_47, %cst_f32_47, %cst_f32_47 : (f32, f32, f32) -> vector<3xf32>
    // CHECK-NEXT: %138 = spirv.CompositeConstruct %cst_f32_46, %cst_f32_46, %cst_f32_46 : (f32, f32, f32) -> vector<3xf32>
    // CHECK-NEXT: %139 = spirv.Select %136, %138, %137 : vector<3xi1>, vector<3xf32>
    vec3 ee = vec3(_bvec3);

    // CHECK: %142 = spirv.FConvert %141 : vector<3xf64> to vector<3xf32>
    vec3 ff = vec3(_dvec3);

    // CHECK: %145 = spirv.ConvertSToF %144 : vector<3xsi32> to vector<3xf64>
    dvec3 gg = dvec3(_ivec3);

    // CHECK: %148 = spirv.ConvertUToF %147 : vector<3xui32> to vector<3xf64>
    dvec3 hh = dvec3(_uvec3);

    // CHECK: %cst_f64_48 = spirv.Constant 1.000000e+00 : f64
    // CHECK-NEXT: %cst_f64_49 = spirv.Constant 0.000000e+00 : f64
    // CHECK-NEXT: %151 = spirv.CompositeConstruct %cst_f64_49, %cst_f64_49, %cst_f64_49 : (f64, f64, f64) -> vector<3xf64>
    // CHECK-NEXT: %152 = spirv.CompositeConstruct %cst_f64_48, %cst_f64_48, %cst_f64_48 : (f64, f64, f64) -> vector<3xf64>
    // CHECK-NEXT: %153 = spirv.Select %150, %152, %151 : vector<3xi1>, vector<3xf64>
    dvec3 ii = dvec3(_bvec3);

    // CHECK:  %156 = spirv.FConvert %155 : vector<3xf32> to vector<3xf64>
    dvec3 jj = dvec3(_vec3);

    // CHECK: %cst0_si32_50 = spirv.Constant 0 : si32
    // CHECK-NEXT: %159 = spirv.CompositeConstruct %cst0_si32_50, %cst0_si32_50, %cst0_si32_50 : (si32, si32, si32) -> vector<3xsi32>
    // CHECK-NEXT: %160 = spirv.INotEqual %158, %159 : vector<3xsi32>
    bvec3 kk = bvec3(_ivec3);

    // CHECK: %cst0_ui32_51 = spirv.Constant 0 : ui32
    // CHECK-NEXT: %163 = spirv.CompositeConstruct %cst0_ui32_51, %cst0_ui32_51, %cst0_ui32_51 : (ui32, ui32, ui32) -> vector<3xui32>
    // CHECK-NEXT: %164 = spirv.INotEqual %162, %163 : vector<3xui32>
    bvec3 ll = bvec3(_uvec3);

    // CHECK: %cst_f32_52 = spirv.Constant 0.000000e+00 : f32
    // CHECK-NEXT: %167 = spirv.CompositeConstruct %cst_f32_52, %cst_f32_52, %cst_f32_52 : (f32, f32, f32) -> vector<3xf32>
    // CHECK-NEXT: %168 = spirv.FOrdNotEqual %166, %167 : vector<3xf32>
    bvec3 mm = bvec3(_vec3);

    // CHECK: %cst_f64_53 = spirv.Constant 0.000000e+00 : f64
    // CHECK-NEXT: %171 = spirv.CompositeConstruct %cst_f64_53, %cst_f64_53, %cst_f64_53 : (f64, f64, f64) -> vector<3xf64>
    // CHECK-NEXT: %172 = spirv.FOrdNotEqual %170, %171 : vector<3xf64>
    bvec3 nn = bvec3(_dvec3);

    // vec4

    // CHECK: %175 = spirv.Bitcast %174 : vector<4xui32> to vector<4xsi32>
    ivec4 oo = ivec4(_uvec4);

    // CHECK: %cst1_si32_54 = spirv.Constant 1 : si32
    // CHECK-NEXT: %cst0_si32_55 = spirv.Constant 0 : si32
    // CHECK-NEXT: %178 = spirv.CompositeConstruct %cst0_si32_55, %cst0_si32_55, %cst0_si32_55, %cst0_si32_55 : (si32, si32, si32, si32) -> vector<4xsi32>
    // CHECK-NEXT: %179 = spirv.CompositeConstruct %cst1_si32_54, %cst1_si32_54, %cst1_si32_54, %cst1_si32_54 : (si32, si32, si32, si32) -> vector<4xsi32>
    // CHECK-NEXT: %180 = spirv.Select %177, %179, %178 : vector<4xi1>, vector<4xsi32>
    ivec4 pp = ivec4(_bvec4);

    // CHECK: %183 = spirv.ConvertFToS %182 : vector<4xf32> to vector<4xsi32>
    ivec4 qq = ivec4(_vec4);

    // CHECK: %186 = spirv.ConvertFToS %185 : vector<4xf64> to vector<4xsi32>
    ivec4 rr = ivec4(_dvec4);

    // CHECK: %189 = spirv.Bitcast %188 : vector<4xsi32> to vector<4xui32>
    uvec4 ss = uvec4(_ivec4);

    // CHECK: %cst1_ui32_56 = spirv.Constant 1 : ui32
    // CHECK-NEXT: %cst0_ui32_57 = spirv.Constant 0 : ui32
    // CHECK-NEXT: %192 = spirv.CompositeConstruct %cst0_ui32_57, %cst0_ui32_57, %cst0_ui32_57, %cst0_ui32_57 : (ui32, ui32, ui32, ui32) -> vector<4xui32>
    // CHECK-NEXT: %193 = spirv.CompositeConstruct %cst1_ui32_56, %cst1_ui32_56, %cst1_ui32_56, %cst1_ui32_56 : (ui32, ui32, ui32, ui32) -> vector<4xui32>
    // CHECK-NEXT: %194 = spirv.Select %191, %193, %192 : vector<4xi1>, vector<4xui32>
    uvec4 tt = uvec4(_bvec4);

    // CHECK: %197 = spirv.ConvertFToU %196 : vector<4xf32> to vector<4xui32>
    uvec4 uu = uvec4(_vec4);

    // CHECK:  %200 = spirv.ConvertFToU %199 : vector<4xf64> to vector<4xui32>
    uvec4 vv = uvec4(_dvec4);

    // CHECK: %203 = spirv.ConvertSToF %202 : vector<4xsi32> to vector<4xf32>
    vec4 ww = vec4(_ivec4);

    // CHECK: %206 = spirv.ConvertUToF %205 : vector<4xui32> to vector<4xf32>
    vec4 xx = vec4(_uvec4);

    // CHECK: %cst_f32_58 = spirv.Constant 1.000000e+00 : f32
    // CHECK-NEXT: %cst_f32_59 = spirv.Constant 0.000000e+00 : f32
    // CHECK-NEXT: %209 = spirv.CompositeConstruct %cst_f32_59, %cst_f32_59, %cst_f32_59, %cst_f32_59 : (f32, f32, f32, f32) -> vector<4xf32>
    // CHECK-NEXT: %210 = spirv.CompositeConstruct %cst_f32_58, %cst_f32_58, %cst_f32_58, %cst_f32_58 : (f32, f32, f32, f32) -> vector<4xf32>
    // CHECK-NEXT: %211 = spirv.Select %208, %210, %209 : vector<4xi1>, vector<4xf32>
    vec4 yy = vec4(_bvec4);

    // CHECK: %214 = spirv.FConvert %213 : vector<4xf64> to vector<4xf32>
    vec4 zz = vec4(_dvec4);

    // CHECK: %217 = spirv.ConvertSToF %216 : vector<4xsi32> to vector<4xf64>
    dvec4 aaa = dvec4(_ivec4);

    // CHECK: %220 = spirv.ConvertUToF %219 : vector<4xui32> to vector<4xf64>
    dvec4 bbb = dvec4(_uvec4);

    // CHECK: %cst_f64_60 = spirv.Constant 1.000000e+00 : f64
    // CHECK-NEXT: %cst_f64_61 = spirv.Constant 0.000000e+00 : f64
    // CHECK-NEXT: %223 = spirv.CompositeConstruct %cst_f64_61, %cst_f64_61, %cst_f64_61, %cst_f64_61 : (f64, f64, f64, f64) -> vector<4xf64>
    // CHECK-NEXT: %224 = spirv.CompositeConstruct %cst_f64_60, %cst_f64_60, %cst_f64_60, %cst_f64_60 : (f64, f64, f64, f64) -> vector<4xf64>
    // CHECK-NEXT: %225 = spirv.Select %222, %224, %223 : vector<4xi1>, vector<4xf64>
    dvec4 ccc = dvec4(_bvec4);
    
    // CHECK: %228 = spirv.FConvert %227 : vector<4xf32> to vector<4xf64>
    dvec4 ddd = dvec4(_vec4);

    // CHECK: %cst0_si32_62 = spirv.Constant 0 : si32
    // CHECK-NEXT: %231 = spirv.CompositeConstruct %cst0_si32_62, %cst0_si32_62, %cst0_si32_62, %cst0_si32_62 : (si32, si32, si32, si32) -> vector<4xsi32>
    // CHECK-NEXT: %232 = spirv.INotEqual %230, %231 : vector<4xsi32>
    bvec4 eee = bvec4(_ivec4);

    // CHECK: %cst0_ui32_63 = spirv.Constant 0 : ui32
    // CHECK-NEXT: %235 = spirv.CompositeConstruct %cst0_ui32_63, %cst0_ui32_63, %cst0_ui32_63, %cst0_ui32_63 : (ui32, ui32, ui32, ui32) -> vector<4xui32>
    // CHECK-NEXT: %236 = spirv.INotEqual %234, %235 : vector<4xui32>
    bvec4 fff = bvec4(_uvec4);

    // CHECK: %cst_f32_64 = spirv.Constant 0.000000e+00 : f32
    // CHECK-NEXT: %239 = spirv.CompositeConstruct %cst_f32_64, %cst_f32_64, %cst_f32_64, %cst_f32_64 : (f32, f32, f32, f32) -> vector<4xf32>
    // CHECK-NEXT: %240 = spirv.FOrdNotEqual %238, %239 : vector<4xf32>
    bvec4 ggg = bvec4(_vec4);

    // CHECK: %cst_f64_65 = spirv.Constant 0.000000e+00 : f64
    // CHECK-NEXT: %243 = spirv.CompositeConstruct %cst_f64_65, %cst_f64_65, %cst_f64_65, %cst_f64_65 : (f64, f64, f64, f64) -> vector<4xf64>
    // CHECK-NEXT: %244 = spirv.FOrdNotEqual %242, %243 : vector<4xf64>
    bvec4 hhh = bvec4(_dvec4);
}
