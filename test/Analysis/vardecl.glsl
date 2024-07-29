int myFunc() {
    // OK
    float a = 1.0;
    float b = 2.2;

    // OK
    float c = a + b;

    // error: cannot convert 'float' to 'int'
    int d = 1.0;

    // OK
    a = 3.0;

    // error: cannot convert 'int' to 'float'
    a = 1;

    bool test = true;

    // OK
    int e = test ? 1 : 2;

    // error: boolean expression expected
    e = d ? 1 : 2;

    return;
}
