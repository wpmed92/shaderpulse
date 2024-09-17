layout(local_size_x = 16, local_size_y = 8, local_size_z = 4) in int;

layout(binding = 0) buffer InputBuffer {
    float data[];
};

layout(binding = 1) buffer OutputBuffer {
    float result[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    result[index] = data[index] * 2.0;
}
