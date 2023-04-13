#ifndef TOK
#define TOK(X)
#endif
#ifndef KEYWORD
#define KEYWORD(X) TOK(kw_ ## X)
#endif
#ifndef PUNCTUATOR
#define PUNCTUATOR(X,Y) TOK(X)
#endif
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)


KEYWORD(const)
KEYWORD(uniform)
KEYWORD(buffer)
KEYWORD(shared)
KEYWORD(attribute)
KEYWORD(varying)
KEYWORD(coherent)
KEYWORD(volatile)
KEYWORD(restrict)
KEYWORD(readonly)
KEYWORD(writeonly)
KEYWORD(atomic_uint)
KEYWORD(layout)
KEYWORD(centroid)
KEYWORD(flat)
KEYWORD(smooth)
KEYWORD(noperspective)

KEYWORD(patch)
KEYWORD(sample)
KEYWORD(invariant)
KEYWORD(precise)

KEYWORD(break)
KEYWORD(continue)
KEYWORD(do)
KEYWORD(for)
KEYWORD(while)
KEYWORD(switch)
KEYWORD(case)
KEYWORD(default)

KEYWORD(if)
KEYWORD(else)

KEYWORD(subroutine)

KEYWORD(in)
KEYWORD(out)
KEYWORD(inout)

// Start of type keyword
// Scalar types
KEYWORD(int)
KEYWORD(uint)
KEYWORD(void)
KEYWORD(bool)
KEYWORD(true)
KEYWORD(false)
KEYWORD(float)
KEYWORD(double)

// Vector types
KEYWORD(vec2)
KEYWORD(vec3)
KEYWORD(vec4)
KEYWORD(ivec2)
KEYWORD(ivec3)
KEYWORD(ivec4)
KEYWORD(bvec2)
KEYWORD(bvec3)
KEYWORD(bvec4)

KEYWORD(uvec2)
KEYWORD(uvec3)
KEYWORD(uvec4)

KEYWORD(dvec2)
KEYWORD(dvec3)
KEYWORD(dvec4)

KEYWORD(mat2)
KEYWORD(mat3)
KEYWORD(mat4)

KEYWORD(mat2x2)
KEYWORD(mat2x3)
KEYWORD(mat2x4)

KEYWORD(mat3x2)
KEYWORD(mat3x3)
KEYWORD(mat3x4)

KEYWORD(mat4x2)
KEYWORD(mat4x3)
KEYWORD(mat4x4)

KEYWORD(dmat2)
KEYWORD(dmat3)
KEYWORD(dmat4)

KEYWORD(dmat2x2)
KEYWORD(dmat2x3)
KEYWORD(dmat2x4)

KEYWORD(dmat3x2)
KEYWORD(dmat3x3)
KEYWORD(dmat3x4)

KEYWORD(dmat4x2)
KEYWORD(dmat4x3)
KEYWORD(dmat4x4)

KEYWORD(sampler1D)
KEYWORD(sampler1DShadow)
KEYWORD(sampler1DArray)
KEYWORD(sampler1DArrayShadow)

KEYWORD(isampler1D)
KEYWORD(isampler1DArray)
KEYWORD(usampler1D)
KEYWORD(usampler1DArray)

KEYWORD(sampler2D)
KEYWORD(sampler2DShadow)
KEYWORD(sampler2DArray)
KEYWORD(sampler2DArrayShadow)

KEYWORD(isampler2D)
KEYWORD(isampler2DArray)
KEYWORD(usampler2D)
KEYWORD(usampler2DArray)

KEYWORD(sampler2DRect)
KEYWORD(sampler2DRectShadow)
KEYWORD(isampler2DRect)
KEYWORD(usampler2DRect)

KEYWORD(sampler2DMS)
KEYWORD(isampler2DMS)
KEYWORD(usampler2DMS)

KEYWORD(sampler2DMSArray)
KEYWORD(isampler2DMSArray)
KEYWORD(usampler2DMSArray)

KEYWORD(sampler3D)
KEYWORD(isampler3D)
KEYWORD(usampler3D)

KEYWORD(samplerCube)
KEYWORD(samplerCubeShadow)
KEYWORD(isamplerCube)
KEYWORD(usamplerCube)

KEYWORD(samplerCubeArray)
KEYWORD(samplerCubeArrayShadow)

KEYWORD(isamplerCubeArray)
KEYWORD(usamplerCubeArray)

KEYWORD(samplerBuffer)
KEYWORD(isamplerBuffer)
KEYWORD(usamplerBuffer)

KEYWORD(image1D)
KEYWORD(iimage1D)
KEYWORD(uimage1D)

KEYWORD(image1DArray)
KEYWORD(iimage1DArray)
KEYWORD(uimage1DArray)

KEYWORD(image2D)
KEYWORD(iimage2D)
KEYWORD(uimage2D)

KEYWORD(image2DArray)
KEYWORD(iimage2DArray)
KEYWORD(uimage2DArray)

KEYWORD(image2DRect)
KEYWORD(iimage2DRect)
KEYWORD(uimage2DRect)

KEYWORD(image2DMS)
KEYWORD(iimage2DMS)
KEYWORD(uimage2DMS)

KEYWORD(image2DMSArray)
KEYWORD(iimage2DMSArray)
KEYWORD(uimage2DMSArray)

KEYWORD(image3D)
KEYWORD(iimage3D)
KEYWORD(uimage3DimageCube)

KEYWORD(iimageCube)
KEYWORD(uimageCube)

KEYWORD(imageCubeArray)
KEYWORD(iimageCubeArray)
KEYWORD(uimageCubeArray)

KEYWORD(imageBuffer)
KEYWORD(iimageBuffer)
KEYWORD(uimageBuffer)

// End of type keywords

KEYWORD(lowp)
KEYWORD(mediump)
KEYWORD(highp)
KEYWORD(precision)

KEYWORD(discard)
KEYWORD(return)

KEYWORD(struct)

// Operators
PUNCTUATOR(lParen,      "(")
PUNCTUATOR(rParen,      ")")
PUNCTUATOR(lBracket,    "[")
PUNCTUATOR(rBracket,    "]")
PUNCTUATOR(dot,         ".")
PUNCTUATOR(increment,  "++")
PUNCTUATOR(decrement,  "--")
PUNCTUATOR(plus,        "+")
PUNCTUATOR(minus,       "-")
PUNCTUATOR(mul,         "*")
PUNCTUATOR(div,         "/")
PUNCTUATOR(modulo,      "%")
PUNCTUATOR(shiftLeft,  "<<")
PUNCTUATOR(shiftRight, ">>")
PUNCTUATOR(lt,          "<")
PUNCTUATOR(gt,          ">")
PUNCTUATOR(ltEq,       "<=")
PUNCTUATOR(gtEq,       ">=")
PUNCTUATOR(eq,         "==")
PUNCTUATOR(neq,        "!=")
PUNCTUATOR(band,        "&")
PUNCTUATOR(bxor,        "^")
PUNCTUATOR(bor,         "|")
PUNCTUATOR(bnot,       "~")
PUNCTUATOR(lnot,       "!")
PUNCTUATOR(land,       "&&")
PUNCTUATOR(lxor,       "^^")
PUNCTUATOR(lor,        "||")
PUNCTUATOR(question,   "?")
PUNCTUATOR(colon,      ":")
PUNCTUATOR(assign,     "=")
PUNCTUATOR(addAssign,  "+=")
PUNCTUATOR(subAssign,  "-=")
PUNCTUATOR(mulAssign,  "*=")
PUNCTUATOR(divAssign,  "/=")
PUNCTUATOR(modAssign,  "%=")
PUNCTUATOR(shiftLeftAssign,  "<<=")
PUNCTUATOR(shiftRightAssign,  ">>=")
PUNCTUATOR(landAssign,  "&=")
PUNCTUATOR(lxorAssign,  "^=")
PUNCTUATOR(lorAssign,  "|=")
PUNCTUATOR(comma,  ",")
PUNCTUATOR(semiColon,  ";")
PUNCTUATOR(lCurly,  "{")
PUNCTUATOR(rCurly,  "}")


#undef TOK
#undef KEYWORD
#undef PUNCTUATOR