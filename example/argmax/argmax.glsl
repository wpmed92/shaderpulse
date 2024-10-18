// modified version of: https://github.com/google/uVkCompute/blob/main/benchmarks/argmax/one_workgroup_argmax_loop.glsl
layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

layout(binding=0) buffer InputBuffer { float inputData[]; };
layout(binding=1) buffer OutputBuffer { uint outputData; };

// Each workgroup contains just one subgroup.

void main() {
  uint laneID = gl_LocalInvocationID.x;

  if (laneID == 0u) {
    uint wgResult = 0u;
    float wgMax = inputData[0];

    for (uint i = 0u; i < 32u; ++i) {
      float elem = inputData[i];
      if (elem > wgMax) {
        wgResult = i;
        wgMax = elem;
      }
    }

    outputData = wgResult;
  }
}