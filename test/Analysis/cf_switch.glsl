 void testFunc() {
    int a = 1;
    float b = 1.0;

    // error: init-expression in a switch statement must be a scalar integer
    switch (b) {

    }

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

        // error: case label must be a scalar integer
        case 1.0:
            break;

        // OK
        default:
            // OK
            break;
    }
 }
