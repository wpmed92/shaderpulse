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

    // error: break only in loops and switches
    break;

    // error: continue only in loops
    continue;

    while (b) {
        // OK
        break;

        // OK
        continue;
    }

    do {
        // OK
        break;

        // OK
        continue;
    } while(b);
}
