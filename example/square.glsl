layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer InBuffer {
    int data[];
};

layout(binding = 1) buffer OutBuffer {
    int result[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    result[idx] = data[idx] * data[idx];
}
