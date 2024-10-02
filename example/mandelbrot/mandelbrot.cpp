#include <vulkan/vulkan.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include "lodepng.h"

const uint32_t WIDTH = 3200;
const uint32_t HEIGHT = 2400;
const uint32_t WORKGROUP_SIZE = 32;

int main() {
    // Application Info
    vk::ApplicationInfo AppInfo{
        "VulkanCompute",      // Application Name
        1,                    // Application Version
        nullptr,              // Engine Name or nullptr
        0,                    // Engine Version
        VK_API_VERSION_1_1    // Vulkan API version
    };

    // Enable validation layers
    const std::vector<const char*> Layers = { "VK_LAYER_KHRONOS_validation" };

    // Specify instance extensions including VK_KHR_portability_enumeration
    const std::vector<const char*> InstanceExtensions = {
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    };

    // Instance create info with portability enumeration flag
    vk::InstanceCreateInfo InstanceCreateInfo(
        vk::InstanceCreateFlags(VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR), // Flags
        &AppInfo,                                                              // Application Info
        Layers,                                                                // Layers
        InstanceExtensions                                                      // Extensions
    );

    // Create Vulkan instance
    vk::Instance Instance = vk::createInstance(InstanceCreateInfo);

    // Enumerate physical devices
    std::vector<vk::PhysicalDevice> PhysicalDevices = Instance.enumeratePhysicalDevices();
    vk::PhysicalDevice PhysicalDevice = PhysicalDevices.front();
    vk::PhysicalDeviceProperties DeviceProps = PhysicalDevice.getProperties();
    std::cout << "Device Name    : " << DeviceProps.deviceName << std::endl;

    const uint32_t ApiVersion = DeviceProps.apiVersion;
    std::cout << "Vulkan Version : " << VK_VERSION_MAJOR(ApiVersion) << "." 
              << VK_VERSION_MINOR(ApiVersion) << "." 
              << VK_VERSION_PATCH(ApiVersion) << std::endl;

    vk::PhysicalDeviceLimits DeviceLimits = DeviceProps.limits;
    std::cout << "Max Compute Shared Memory Size: " << DeviceLimits.maxComputeSharedMemorySize / 1024 << " KB" << std::endl;

    // Find compute queue family
    std::vector<vk::QueueFamilyProperties> QueueFamilyProps = PhysicalDevice.getQueueFamilyProperties();
    auto PropIt = std::find_if(QueueFamilyProps.begin(), QueueFamilyProps.end(), [](const vk::QueueFamilyProperties& Prop) {
        return Prop.queueFlags & vk::QueueFlagBits::eCompute;
    });

    // Check if compute queue family was found
    if (PropIt == QueueFamilyProps.end()) {
        std::cerr << "No compute queue family found!" << std::endl;
        return EXIT_FAILURE;
    }

    const uint32_t ComputeQueueFamilyIndex = std::distance(QueueFamilyProps.begin(), PropIt);
    std::cout << "Compute Queue Family Index: " << ComputeQueueFamilyIndex << std::endl;

    // Device queue create info
    vk::DeviceQueueCreateInfo DeviceQueueCreateInfo(
        vk::DeviceQueueCreateFlags(),   // Flags
        ComputeQueueFamilyIndex,        // Queue Family Index
        1,                               // Number of Queues
        new float{1.0f}                 // Queue Priorities
    );

    // Specify device extensions including VK_KHR_portability_subset
    const std::vector<const char*> DeviceExtensions = {
        "VK_KHR_portability_subset"
    };

    // Device create info
    vk::DeviceCreateInfo DeviceCreateInfo(
        vk::DeviceCreateFlags(),          // Flags
        DeviceQueueCreateInfo,            // Device Queue Create Info struct
        {},                                // Layer names
        DeviceExtensions                   // Device Extensions
    );

    // Create Vulkan device
    vk::Device Device = PhysicalDevice.createDevice(DeviceCreateInfo);

    // Buffer creation
    /*
    void CreateBuffer() {
    auto buffer_create_info = vk::BufferCreateInfo();
    buffer_create_info.setSize(buffer_size)
        .setUsage(vk::BufferUsageFlagBits::eStorageBuffer)
        .setSharingMode(vk::SharingMode::eExclusive);
    buffer_ = device_->createBufferUnique(buffer_create_info);
  }*/
    const uint32_t BufferSize = 4 * sizeof(float) * WIDTH * HEIGHT;

    vk::BufferCreateInfo BufferCreateInfo{
        vk::BufferCreateFlags(),                    // Flags
        BufferSize,                                 // Size
        vk::BufferUsageFlagBits::eStorageBuffer,    // Usage
        vk::SharingMode::eExclusive,                // Sharing mode
        1,                                          // Number of queue family indices
        &ComputeQueueFamilyIndex                    // List of queue family indices
    };
    vk::Buffer OutBuffer = Device.createBuffer(BufferCreateInfo);
    vk::MemoryRequirements OutBufferMemoryRequirements = Device.getBufferMemoryRequirements(OutBuffer);

    vk::PhysicalDeviceMemoryProperties MemoryProperties = PhysicalDevice.getMemoryProperties();

    uint32_t MemoryTypeIndex = uint32_t(~0);
    vk::DeviceSize MemoryHeapSize = uint32_t(~0);
    for (uint32_t CurrentMemoryTypeIndex = 0; CurrentMemoryTypeIndex < MemoryProperties.memoryTypeCount; ++CurrentMemoryTypeIndex)
    {
        vk::MemoryType MemoryType = MemoryProperties.memoryTypes[CurrentMemoryTypeIndex];
        if ((vk::MemoryPropertyFlagBits::eHostVisible & MemoryType.propertyFlags) &&
            (vk::MemoryPropertyFlagBits::eHostCoherent & MemoryType.propertyFlags))
        {
            MemoryHeapSize = MemoryProperties.memoryHeaps[MemoryType.heapIndex].size;
            MemoryTypeIndex = CurrentMemoryTypeIndex;
            break;
        }
    }

    std::cout << "Memory Type Index: " << MemoryTypeIndex << std::endl;
    std::cout << "Memory Heap Size : " << MemoryHeapSize / 1024 / 1024 / 1024 << " GB" << std::endl;

    // Allocate device memory
    vk::MemoryAllocateInfo OutBufferMemoryAllocateInfo(OutBufferMemoryRequirements.size, MemoryTypeIndex);
    vk::DeviceMemory OutBufferMemory = Device.allocateMemory(OutBufferMemoryAllocateInfo);


    // Bind the allocated memory to the buffers
    Device.bindBufferMemory(OutBuffer, OutBufferMemory, 0);

    // Shader module
    std::vector<char> ShaderContents;
    if (std::ifstream ShaderFile{ "mandelbrot.spv", std::ios::binary | std::ios::ate })
    {
        const size_t FileSize = ShaderFile.tellg();
        ShaderFile.seekg(0);
        ShaderContents.resize(FileSize, '\0');
        ShaderFile.read(ShaderContents.data(), FileSize);
    }

    vk::ShaderModuleCreateInfo ShaderModuleCreateInfo(
        vk::ShaderModuleCreateFlags(),                                // Flags
        ShaderContents.size(),                                        // Code size
        reinterpret_cast<const uint32_t*>(ShaderContents.data()));    // Code
    vk::ShaderModule ShaderModule = Device.createShaderModule(ShaderModuleCreateInfo);

   const std::vector<vk::DescriptorSetLayoutBinding> DescriptorSetLayoutBinding = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    };

    vk::DescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo(
        vk::DescriptorSetLayoutCreateFlags(),
        DescriptorSetLayoutBinding);
    vk::DescriptorSetLayout DescriptorSetLayout = Device.createDescriptorSetLayout(DescriptorSetLayoutCreateInfo);

    // Pipeline creation
    vk::PipelineLayoutCreateInfo PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), DescriptorSetLayout);
    vk::PipelineLayout PipelineLayout = Device.createPipelineLayout(PipelineLayoutCreateInfo);
    vk::PipelineCache PipelineCache = Device.createPipelineCache(vk::PipelineCacheCreateInfo());

    vk::PipelineShaderStageCreateInfo PipelineShaderCreateInfo(
        vk::PipelineShaderStageCreateFlags(),  // Flags
        vk::ShaderStageFlagBits::eCompute,     // Stage
        ShaderModule,                          // Shader Module
        "main"                                 // Shader Entry Point
    );                               
    vk::ComputePipelineCreateInfo ComputePipelineCreateInfo(
        vk::PipelineCreateFlags(),    // Flags
        PipelineShaderCreateInfo,     // Shader Create Info struct
        PipelineLayout                // Pipeline Layout
    );              
    vk::ResultValue<vk::Pipeline> ComputePipeline = Device.createComputePipeline(PipelineCache, ComputePipelineCreateInfo);

    // Descriptor pool
    vk::DescriptorPoolSize DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1);
    vk::DescriptorPoolCreateInfo DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1, DescriptorPoolSize);
    vk::DescriptorPool DescriptorPool = Device.createDescriptorPool(DescriptorPoolCreateInfo);

    vk::DescriptorSetAllocateInfo DescriptorSetAllocInfo(DescriptorPool, 1, &DescriptorSetLayout);
    const std::vector<vk::DescriptorSet> DescriptorSets = Device.allocateDescriptorSets(DescriptorSetAllocInfo);
    vk::DescriptorSet DescriptorSet = DescriptorSets.front();
    vk::DescriptorBufferInfo OutBufferInfo(OutBuffer, 0, BufferSize);

    const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
        {DescriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &OutBufferInfo},
    };
    Device.updateDescriptorSets(WriteDescriptorSets, {});

    // Command pool
    vk::CommandPoolCreateInfo CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), ComputeQueueFamilyIndex);
    vk::CommandPool CommandPool = Device.createCommandPool(CommandPoolCreateInfo);

    // Command buffers
    vk::CommandBufferAllocateInfo CommandBufferAllocInfo(
    CommandPool,                         // Command Pool
    vk::CommandBufferLevel::ePrimary,    // Level
    1);                                  // Num Command Buffers
    const std::vector<vk::CommandBuffer> CmdBuffers = Device.allocateCommandBuffers(CommandBufferAllocInfo);
    vk::CommandBuffer CmdBuffer = CmdBuffers.front();

    vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    CmdBuffer.begin(CmdBufferBeginInfo);
    CmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, ComputePipeline.value);
    CmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,    // Bind point
                                    PipelineLayout,                  // Pipeline Layout
                                    0,                               // First descriptor set
                                    { DescriptorSet },               // List of descriptor sets
                                    {});                             // Dynamic offsets
    uint32_t WorkGroupCountX = (WIDTH + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE; // Calculate number of workgroups in X dimension
    uint32_t WorkGroupCountY = (HEIGHT + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE; // Calculate number of workgroups in Y dimension

    CmdBuffer.dispatch(WorkGroupCountX, WorkGroupCountY, 1);

    CmdBuffer.end();

    // Submit command
    vk::Queue Queue = Device.getQueue(ComputeQueueFamilyIndex, 0);
    vk::Fence Fence = Device.createFence(vk::FenceCreateInfo());

    vk::SubmitInfo SubmitInfo(0,                // Num Wait Semaphores
                                nullptr,        // Wait Semaphores
                                nullptr,        // Pipeline Stage Flags
                                1,              // Num Command Buffers
                                &CmdBuffer);    // List of command buffers
    Queue.submit({ SubmitInfo }, Fence);
    Device.waitForFences({ Fence },             // List of fences
                            true,               // Wait All
                            uint64_t(-1));      // Timeout

    float* OutBufferPtr = static_cast<float*>(Device.mapMemory(OutBufferMemory, 0, BufferSize));
    std::vector<unsigned char> image;
    constexpr int imageSize = WIDTH * HEIGHT * 4;
    image.reserve(imageSize);

    for (int i = 0; i < imageSize; i+=4) {
      image.push_back(static_cast<unsigned char>(255.0f * (OutBufferPtr[i])));
      image.push_back(static_cast<unsigned char>(255.0f * (OutBufferPtr[i+1])));
      image.push_back(static_cast<unsigned char>(255.0f * (OutBufferPtr[i+2])));
      image.push_back(static_cast<unsigned char>(255.0f * (OutBufferPtr[i+3])));
    }

    Device.unmapMemory(OutBufferMemory);
    unsigned error = lodepng::encode("mandelbrot.png", image, WIDTH, HEIGHT);

    if (error) {
      throw std::runtime_error(lodepng_error_text(error));
    }

    Device.resetCommandPool(CommandPool, vk::CommandPoolResetFlags());
    Device.destroyFence(Fence);
    Device.destroyDescriptorSetLayout(DescriptorSetLayout);
    Device.destroyPipelineLayout(PipelineLayout);
    Device.destroyPipelineCache(PipelineCache);
    Device.destroyShaderModule(ShaderModule);
    Device.destroyPipeline(ComputePipeline.value);
    Device.destroyDescriptorPool(DescriptorPool);
    Device.destroyCommandPool(CommandPool);
    Device.freeMemory(OutBufferMemory);
    Device.destroyBuffer(OutBuffer);
    Device.destroy();
    Instance.destroy();
    return 0;
}
