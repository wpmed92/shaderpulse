void testFunc() {
    bool a = true;
    int b = 1;

    // error: boolean expression expected in if condition
    if (b) {

    } else {

    }

    // OK
    if (a) {

    } else {

    }

    // error: case only in switch
    case 1: break;

    // error: default only in switch
    default: break;

    switch (b) {
        // OK
        case 1:
            // OK
            break;

        // OK
        default:
            // OK
            break;
    }
}
