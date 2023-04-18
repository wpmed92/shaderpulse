#pragma once
#include <string>

namespace shaderpulse {

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

class Type {

public:
    Type(TypeKind kind) : kind(kind) {

    }
    virtual ~Type() = default;

    TypeKind getKind() const { return kind; }
    inline bool isScalar() const { return kind >= TypeKind::Void && kind <= TypeKind::Double; }
    inline bool isVector() const { return kind == TypeKind::Vector; }
    inline bool isMatrix() const { return kind == TypeKind::Matrix; }
    inline bool isStruct() const { return kind == TypeKind::Struct; }
    inline bool isOpaque() const { return kind == TypeKind::Opaque; }

private:
    TypeKind kind;
};

class VectorType : public Type {

public:
    VectorType(std::unique_ptr<Type> elementType, int length) : 
       Type(TypeKind::Vector), 
        elementType(std::move(elementType)), 
        length(length) {
        assert(this->elementType->isScalar());
    }

    Type* getElementType() const { return elementType.get(); };
    int getLength() const { return length; };

private:
    std::unique_ptr<Type> elementType;
    int length;
};

class MatrixType : public Type {

public:
    MatrixType(std::unique_ptr<Type> elementType, int rows, int cols) :
        Type(TypeKind::Matrix), 
        elementType(std::move(elementType)), 
        rows(rows), 
        cols(cols) {
            assert(this->elementType->isScalar());
        }

    int getRows() const { return rows; }
    int getCols() const { return cols; }
    Type* getElementType() const { return elementType.get(); }
    
private:
    std::unique_ptr<Type> elementType;
    int rows;
    int cols;

};

class StructType : public Type {

public:
    StructType(const std::string& structName) : 
        Type(TypeKind::Struct),
        structName(structName) {

        }
    
    const std::string &getStructName() const { return structName; }

private:    
    std::string structName;
};

}; // namespace shaderpulse