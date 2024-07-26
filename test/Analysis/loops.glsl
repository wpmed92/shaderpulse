void loopTest() {
    int a = 1;

    // error: boolean expression expected in while condition
    while (a) {

    }

    // error: boolean expression expected in while condition
    do {

    } while(a);

    bool b = true;

    // OK
    while (b) {

    }

    // OK
    do {

    } while(b);

    int c = 1;

    // Test scopes
    while (b) {
        int c = 2;
    }
}
