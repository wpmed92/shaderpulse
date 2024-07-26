 void testFunc() {
    int a = 1;

    // error: case only in switch
    // error: break only in loops and switches
    case 1: break;

    // error: default only in switch
    // error: break only in loops and switches
    default: break;

    switch (a) {
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
