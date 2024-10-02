cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(VulkanCompute)

find_package(Vulkan REQUIRED)

set(SOURCES
    "mandelbrot.cpp"
    "lodepng.cpp"
)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(mandelbrot_vulkan ${SOURCES})
target_link_libraries(mandelbrot_vulkan PRIVATE Vulkan::Vulkan)
