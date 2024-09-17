layout(local_size_x = 16, local_size_y = 8, local_size_z = 4) in int;

// Test built-ins

// CHECK: spirv.GlobalVariable @gl_GlobalInvocationID built_in("GlobalInvocationId") : !spirv.ptr<vector<3xui32>, Input>
// CHECK-NEXT: spirv.GlobalVariable @gl_WorkGroupID built_in("WorkgroupId") : !spirv.ptr<vector<3xui32>, Input>
// CHECK-NEXT: spirv.GlobalVariable @gl_WorkGroupSize built_in("WorkgroupSize") : !spirv.ptr<vector<3xui32>, Input>
// CHECK-NEXT: spirv.GlobalVariable @gl_LocalInvocationID built_in("LocalInvocationId") : !spirv.ptr<vector<3xui32>, Input>

// CHECK: spirv.GlobalVariable @data {binding = 0 : i32} : !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>
layout(binding = 0) buffer InputBuffer {
    float data[];
};

// CHECK: spirv.GlobalVariable @result {binding = 1 : i32} : !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>
layout(binding = 1) buffer OutputBuffer {
    float result[];
};

void main() {
    // CHECK: %gl_GlobalInvocationID_addr = spirv.mlir.addressof @gl_GlobalInvocationID : !spirv.ptr<vector<3xui32>, Input>
    // CHECK-NEXT: %0 = spirv.Load "Input" %gl_GlobalInvocationID_addr : vector<3xui32>
    // CHECK-NEXT: %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xui32>
    uint index = gl_GlobalInvocationID.x;

    // CHECK: %3 = spirv.Load "Function" %2 : ui32
    // CHECK-NEXT: %4 = spirv.AccessChain %result_addr[%3] : !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>, ui32

    // CHECK: %5 = spirv.Load "Function" %2 : ui32
    // CHECK-NEXT: %6 = spirv.AccessChain %data_addr[%5] : !spirv.ptr<!spirv.rtarray<f32>, StorageBuffer>, ui32
    result[index] = data[index] * 2.0;
}

// CHECK: spirv.ExecutionMode @main "LocalSize", 16, 8, 4
// CHECK: spirv.EntryPoint "GLCompute" @main, @gl_GlobalInvocationID, @result, @data
