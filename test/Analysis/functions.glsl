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
