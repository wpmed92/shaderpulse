cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(VulkanCompute)

find_package(Vulkan REQUIRED)

set(SOURCES
    "argmax.cpp"
)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(kernel_argmax ${SOURCES})
target_link_libraries(kernel_argmax PRIVATE Vulkan::Vulkan)
