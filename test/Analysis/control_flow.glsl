void testFunc() {
    bool a = true;
    int b = 1;

    // error: boolean expression expected in while condition
    if (b) {

    } else {

    }

    // OK
    if (a) {

    } else {

    }
}