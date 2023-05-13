vec3 foo() {
    vec3 a = vec3(0.0, 0.1, 0.2);
    vec3 b = vec3(1.1, 1.2, 1.2);
    mat2 testMat = mat2(1.2, 2.3, 
                        3.4, 4.5);


    if (a < b) {
        a = 2.0;
    } else {
        a = 3.0;
    }

    return 0.0;
}