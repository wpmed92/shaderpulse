#pragma once
#include <string>

namespace shaderpulse {

enum PrecisionQualifierKind { High, Medium, Low };

enum InterpolationQualifierKind { Smooth, Flat, Noperspective };

enum StorageQualifierKind {
  Const,
  In,
  Out,
  Inout,
  Centroid,
  Patch,
  Sample,
  Uniform,
  Buffer,
  Shared,
  Coherent,
  Volatile,
  Restrict,
  Readonly,
  Writeonly,
  Subroutine
};

enum TypeKind {
  Void,
  Bool,
  Integer,
  UnsignedInteger,
  Float,
  Double,
  Vector,
  Matrix,
  Opaque,
  Struct
};

enum TypeQualifierKind {
  Storage,
  Interpolation,
  Precision,
  Invariant,
  Precise,
  Layout
};

class TypeQualifier {

public:
  virtual ~TypeQualifier() = default;
  TypeQualifier(TypeQualifierKind kind) : kind(kind) {}

  TypeQualifierKind getKind() const { return kind; }

private:
  TypeQualifierKind kind;
};

class StorageQualifier : public TypeQualifier {

public:
  StorageQualifier(StorageQualifierKind kind)
      : TypeQualifier(TypeQualifierKind::Storage), kind(kind) {}

  StorageQualifierKind getKind() const { return kind; }

private:
  StorageQualifierKind kind;
};

class InterpolationQualifier : public TypeQualifier {

public:
  InterpolationQualifier(InterpolationQualifierKind kind)
      : TypeQualifier(TypeQualifierKind::Interpolation), kind(kind) {}

  InterpolationQualifierKind getKind() const { return kind; }

private:
  InterpolationQualifierKind kind;
};

class PreciseQualifier : public TypeQualifier {

public:
  PreciseQualifier() : TypeQualifier(TypeQualifierKind::Precise) {}
};

class InvariantQualifier : public TypeQualifier {

public:
  InvariantQualifier() : TypeQualifier(TypeQualifierKind::Invariant) {}
};

class PrecisionQualifier : public TypeQualifier {

public:
  PrecisionQualifier(PrecisionQualifierKind kind)
      : TypeQualifier(TypeQualifierKind::Precision), kind(kind) {}

  PrecisionQualifierKind getKind() const { return kind; }

private:
  PrecisionQualifierKind kind;
};

class Type {

public:

  Type(TypeKind kind)
      : kind(kind), qualifiers(std::vector<std::unique_ptr<TypeQualifier>>()) {}
  Type(TypeKind kind, std::vector<std::unique_ptr<TypeQualifier>> qualifiers)
      : kind(kind), qualifiers(std::move(qualifiers)) {}

  virtual ~Type() = default;

  TypeKind getKind() const { return kind; }
  TypeQualifier *getQualifier(TypeQualifierKind kind) {
    if (qualifiers.size() == 0) {
      return nullptr;
    }

    auto it =
        std::find_if(qualifiers.begin(), qualifiers.end(),
                     [&kind](const std::unique_ptr<TypeQualifier> &qualifier) {
                       return qualifier->getKind() == kind;
                     });

    if (it != qualifiers.end()) {
      return it->get();
    } else {
      return nullptr;
    }
  }

  inline bool isScalar() const {
    return kind >= TypeKind::Void && kind <= TypeKind::Double;
  }
  inline bool isVector() const { return kind == TypeKind::Vector; }
  inline bool isMatrix() const { return kind == TypeKind::Matrix; }
  inline bool isStruct() const { return kind == TypeKind::Struct; }
  inline bool isOpaque() const { return kind == TypeKind::Opaque; }

  const std::vector<std::unique_ptr<TypeQualifier>> &getQualifiers() {
    return qualifiers;
  };

private:
  TypeKind kind;
  std::vector<std::unique_ptr<TypeQualifier>> qualifiers;
};

class VectorType : public Type {

public:
VectorType(std::unique_ptr<Type> elementType, int length)
      : Type(TypeKind::Vector, std::vector<std::unique_ptr<TypeQualifier>>()),
        elementType(std::move(elementType)), length(length) {
    assert(this->elementType->isScalar());
  }
  VectorType(std::vector<std::unique_ptr<TypeQualifier>> qualifiers,
             std::unique_ptr<Type> elementType, int length)
      : Type(TypeKind::Vector, std::move(qualifiers)),
        elementType(std::move(elementType)), length(length) {
    assert(this->elementType->isScalar());
  }

  Type *getElementType() const { return elementType.get(); };
  int getLength() const { return length; };

private:
  std::unique_ptr<Type> elementType;
  int length;
};

class MatrixType : public Type {

public:
  MatrixType(std::vector<std::unique_ptr<TypeQualifier>> qualifiers,
             std::unique_ptr<Type> elementType, int rows, int cols)
      : Type(TypeKind::Matrix, std::move(qualifiers)),
        elementType(std::move(elementType)), rows(rows), cols(cols) {
    assert(this->elementType->isScalar());
  }

  int getRows() const { return rows; }
  int getCols() const { return cols; }
  Type *getElementType() const { return elementType.get(); }

private:
  std::unique_ptr<Type> elementType;
  int rows;
  int cols;
};

class StructType : public Type {

public:
  StructType(std::vector<std::unique_ptr<TypeQualifier>> qualifiers,
             const std::string &structName)
      : Type(TypeKind::Struct, std::move(qualifiers)), structName(structName) {}

  const std::string &getStructName() const { return structName; }

private:
  std::string structName;
};

}; // namespace shaderpulse
