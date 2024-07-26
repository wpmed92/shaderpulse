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

    // error: case in switch
    case 1: break;

    switch (b) {
        // OK
        case 1:
            // OK
            break;
    }

}
