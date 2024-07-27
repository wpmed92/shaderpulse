//'return' with no value, in function returning non-void.
int valueFunction() {
    return;
}

//'return' value type does not match the function type.
float voidFunction() {
    return 1;
}
// OK
float floatFunction() {
    return 1.0;
}

// Scopes
int a = 1;

float functionScope(int b) {
    // OK, hides external a
    int a = 2;

    // Error: redeclaration
    int b = 3;
}

// error: redefinition of symbol (no overload support yet)
float functionScope() {

}