void loopTest() {
    int a = 1;

    // error: boolean expression expected in while condition
    while (a) {

    }

    bool b = true;

    // OK
    while (b) {

    }

    int c = 1;

    // Test scopes
    while (b) {
        int c = 2;
    }
}