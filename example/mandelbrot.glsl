// Source: https://github.com/Erkaman/vulkan_minimal_compute/blob/master/shaders/shader.comp
layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0) buffer buf
{
  vec4 imageData[];
};

void main() {
  /*
  In order to fit the work into workgroups, some unnecessary threads are launched.
  We terminate those threads here. 
  */
  // if(gl_GlobalInvocationID.x >= 3200 || gl_GlobalInvocationID.y >= 2400)
  //  return;

  float x = float(gl_GlobalInvocationID.x) / float(3200);
  float y = float(gl_GlobalInvocationID.y) / float(2400);

  /*
  What follows is code for rendering the mandelbrot set. 
  */
  vec2 uv = vec2(x, y);
  float n = 0.0;
  float _calc = 2.0 + 1.7 * 0.2;
  vec2 c = vec2(-.445, 0.0) + (uv - vec2(0.5, 0.5)) * vec2(_calc, _calc);
  vec2 z = vec2(0.0, 0.0);
  int M = 128;

  for (int i = 0; i < M; ++i) {
    z = vec2(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + c;

    if (int(dot(z, z)) > 2)
      break;

    n = n + 1.0;
  }
          
  // we use a simple cosine palette to determine color:
  // http://iquilezles.org/www/articles/palettes/palettes.htm         
  float t = float(n) / float(128);
  vec3 d = vec3(0.3, 0.3, 0.5);
  vec3 e = vec3(-0.2, -0.3 ,-0.5);
  vec3 f = vec3(2.1, 2.0, 3.0);
  vec3 g = vec3(0.0, 0.1, 0.0);
  vec4 color = vec4(d + e * cos(vec3(6.28318, 6.28318, 6.28318) *(f * vec3(t, t, t) + g)), 1.0);
          
  // store the rendered mandelbrot set into a storage buffer:
  imageData[3200u * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x] = color;
}
