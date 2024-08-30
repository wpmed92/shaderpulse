#pragma once
#include <string>
#include <vector>
#include <assert.h>
#include <memory>
#include <algorithm>

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
  Struct,
  Array
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

  virtual bool isEqual(const Type& other) {
    return other.kind == kind;
  }

  virtual bool isBoolLike() {
    return kind == TypeKind::Bool;
  }

  virtual bool isIntLike() {
    return kind == TypeKind::Integer || kind == TypeKind::UnsignedInteger;
  }

  virtual bool isSIntLike() {
    return kind == TypeKind::Integer;
  }

  virtual bool isUIntLike() {
    return kind == TypeKind::UnsignedInteger;
  }

  virtual bool isFloatLike() {
    return kind == TypeKind::Float || kind == TypeKind::Double;
  }

  virtual std::string toString() {
    switch (kind) {
      case TypeKind::Integer:
        return "int";
      case TypeKind::Bool:
        return "bool";
      case TypeKind::Double:
        return "double";
      case TypeKind::Float:
        return "float";
      case TypeKind::UnsignedInteger:
        return "uint";
      case TypeKind::Array:
        return "array";
      case TypeKind::Vector:
        return "vec";
      case TypeKind::Matrix:
        return "mat";
      case TypeKind::Struct:
        return "struct";
      case TypeKind::Void:
        return "void";
      default:
        return "opaque";
    }
  }

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

  bool isEqual(const Type& other) override {
    if (auto vecType = dynamic_cast<const VectorType*>(&other)) {
      return vecType->getElementType()->getKind() == elementType->getKind() && vecType->getLength() == length;
    }

    return false;
  }

  bool isBoolLike() override {
    return elementType->getKind() == TypeKind::Bool;
  }

  bool isIntLike() override {
    return elementType->getKind() == TypeKind::Integer || elementType->getKind() == TypeKind::UnsignedInteger;
  }

  bool isSIntLike() override {
    return elementType->getKind() == TypeKind::Integer;
  }

  bool isUIntLike() override {
    return elementType->getKind() == TypeKind::UnsignedInteger;
  }

  bool isFloatLike() override {
    return elementType->getKind() == TypeKind::Float || elementType->getKind() == TypeKind::Double;
  }

  std::string toString() override {
    std::string prefix = "";

    switch (elementType->getKind()) {
      case TypeKind::Integer:
        prefix = "i";
        break;
      case TypeKind::UnsignedInteger:
        prefix = "u";
        break;
      case TypeKind::Double:
        prefix = "d";
        break;
      case TypeKind::Bool:
        prefix = "b";
        break;
      default:
        prefix = "";
        break;
    }

    return prefix + "vec" + std::to_string(length);
  }

private:
  std::unique_ptr<Type> elementType;
  int length;
};

class ArrayType : public Type {

public:
  ArrayType(std::unique_ptr<Type> elementType, const std::vector<int>& shape)
      : Type(TypeKind::Array, std::vector<std::unique_ptr<TypeQualifier>>()),
        elementType(std::move(elementType)), shape(shape) {
  }
  
  ArrayType(std::vector<std::unique_ptr<TypeQualifier>> qualifiers,
             std::unique_ptr<Type> elementType, const std::vector<int>& shape)
      : Type(TypeKind::Array, std::move(qualifiers)),
        elementType(std::move(elementType)), shape(shape) {
  }

  Type *getElementType() const { return elementType.get(); };
  const std::vector<int> &getShape() { return shape; };

  bool isEqual(const Type& other) override {
    if (auto arrType = dynamic_cast<const ArrayType*>(&other)) {
      return arrType->elementType->getKind() == elementType->getKind() && arrType->shape == shape;
    }

    return false;
  }

  bool isBoolLike() override {
    return elementType->getKind() == TypeKind::Bool;
  }

  bool isIntLike() override {
    return elementType->getKind() == TypeKind::Integer || elementType->getKind() == TypeKind::UnsignedInteger;
  }

  bool isSIntLike() override {
    return elementType->getKind() == TypeKind::Integer;
  }

  bool isUIntLike() override {
    return elementType->getKind() == TypeKind::UnsignedInteger;
  }

  bool isFloatLike() override {
    return elementType->getKind() == TypeKind::Float || elementType->getKind() == TypeKind::Double;
  }

  std::string toString() override {
    std::string arrStr = elementType->toString();

    for (auto dim : shape) {
      arrStr += "[" + std::to_string(dim) + "]";
    }

    return arrStr;
  }

private:
  std::unique_ptr<Type> elementType;
  std::vector<int> shape;
};

class MatrixType : public Type {

public:
  MatrixType(std::unique_ptr<Type> elementType, int rows, int cols)
      : Type(TypeKind::Matrix, std::vector<std::unique_ptr<TypeQualifier>>()),
        elementType(std::move(elementType)), rows(rows), cols(cols) {
    assert(this->elementType->isScalar() && (this->elementType->getKind() == TypeKind::Float || this->elementType->getKind() == TypeKind::Double));
  }
  MatrixType(std::vector<std::unique_ptr<TypeQualifier>> qualifiers,
             std::unique_ptr<Type> elementType, int rows, int cols)
      : Type(TypeKind::Matrix, std::move(qualifiers)),
        elementType(std::move(elementType)), rows(rows), cols(cols) {
    assert(this->elementType->isScalar() && (this->elementType->getKind() == TypeKind::Float || this->elementType->getKind() == TypeKind::Double));
  }

  int getRows() const { return rows; }
  int getCols() const { return cols; }
  Type *getElementType() const { return elementType.get(); }

  bool isEqual(const Type& other) override {
    if (auto matType = dynamic_cast<const MatrixType*>(&other)) {
      return matType->elementType->getKind() == elementType->getKind() && matType->rows == rows && matType->cols == cols;
    }

    return false;
  }

  bool isBoolLike() override {
    return elementType->getKind() == TypeKind::Bool;
  }

  bool isIntLike() override {
    return elementType->getKind() == TypeKind::Integer || elementType->getKind() == TypeKind::UnsignedInteger;
  }

  bool isSIntLike() override {
    return elementType->getKind() == TypeKind::Integer;
  }

  bool isUIntLike() override {
    return elementType->getKind() == TypeKind::UnsignedInteger;
  }

  bool isFloatLike() override {
    return elementType->getKind() == TypeKind::Float || elementType->getKind() == TypeKind::Double;
  }

  std::string toString() override {
    std::string prefix = "";

    switch (elementType->getKind()) {
      case TypeKind::Double:
        prefix = "d";
        break;
      default:
        prefix = "";
        break;
    }

    return prefix + "mat" + std::to_string(rows) + "x" + std::to_string(cols);
  }

private:
  std::unique_ptr<Type> elementType;
  int rows;
  int cols;
};

// TODO: it should store member types
class StructType : public Type {

public:
  StructType(std::vector<std::unique_ptr<TypeQualifier>> qualifiers,
             const std::string &structName)
      : Type(TypeKind::Struct, std::move(qualifiers)), structName(structName) {}

  const std::string &getName() const { return structName; }

  bool isEqual(const Type& other) override {
    if (auto structType = dynamic_cast<const StructType*>(&other)) {
      return structType->structName == structName;
    }

    return false;
  }

  std::string toString() override {
    return "struct '" + structName + "'";
  }

private:
  std::string structName;

};

}; // namespace shaderpulse
