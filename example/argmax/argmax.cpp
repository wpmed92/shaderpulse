#include <vulkan/vulkan.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <cstring>

const uint32_t ELEMENT_SIZE = 32;  // Number of elements
const uint32_t WORKGROUP_SIZE = 4;  // Workgroup size (should be divisible by ELEMENT_SIZE)

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
        &AppInfo,                                                                // Application Info
        Layers,                                                                  // Layers
        InstanceExtensions                                                        // Extensions
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

    if (PropIt == QueueFamilyProps.end()) {
        std::cerr << "No compute queue family found!" << std::endl;
        return EXIT_FAILURE;
    }

    const uint32_t ComputeQueueFamilyIndex = std::distance(QueueFamilyProps.begin(), PropIt);
    std::cout << "Compute Queue Family Index: " << ComputeQueueFamilyIndex << std::endl;

    vk::DeviceQueueCreateInfo DeviceQueueCreateInfo(
        vk::DeviceQueueCreateFlags(),   // Flags
        ComputeQueueFamilyIndex,        // Queue Family Index
        1,                              // Number of Queues
        new float{1.0f}                 // Queue Priorities
    );

    const std::vector<const char*> DeviceExtensions = {
        "VK_KHR_portability_subset"
    };

    vk::DeviceCreateInfo DeviceCreateInfo(
        vk::DeviceCreateFlags(),          // Flags
        DeviceQueueCreateInfo,            // Device Queue Create Info struct
        {},                               // Layer names
        DeviceExtensions                  // Device Extensions
    );

    vk::Device Device = PhysicalDevice.createDevice(DeviceCreateInfo);

    vk::Queue ComputeQueue = Device.getQueue(ComputeQueueFamilyIndex, 0);

    // Create buffers for two inputs and one output
    const uint32_t BufferSize = ELEMENT_SIZE * sizeof(float);

    vk::BufferCreateInfo BufferCreateInfo(
        vk::BufferCreateFlags(),
        BufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::SharingMode::eExclusive,
        1,
        &ComputeQueueFamilyIndex
    );

    // Input Buffers
    vk::Buffer InBuffer = Device.createBuffer(BufferCreateInfo);

    // Output Buffer
    vk::Buffer OutBuffer = Device.createBuffer(BufferCreateInfo);

    // Memory Requirements and Allocation for buffers
    vk::MemoryRequirements MemRequirements = Device.getBufferMemoryRequirements(InBuffer);
    vk::PhysicalDeviceMemoryProperties MemoryProperties = PhysicalDevice.getMemoryProperties();

    uint32_t MemoryTypeIndex = uint32_t(~0);
    for (uint32_t i = 0; i < MemoryProperties.memoryTypeCount; ++i) {
        if ((MemoryProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible) &&
            (MemoryProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            MemoryTypeIndex = i;
            break;
        }
    }

    vk::MemoryAllocateInfo MemoryAllocInfo(MemRequirements.size, MemoryTypeIndex);
    vk::DeviceMemory InBufferMemory = Device.allocateMemory(MemoryAllocInfo);
    vk::DeviceMemory OutBufferMemory = Device.allocateMemory(MemoryAllocInfo);

    std::vector<float> InputData(ELEMENT_SIZE);

    for (int i = 0; i < ELEMENT_SIZE; i++) {
        InputData[i] = 0.0;
    }

    InputData[31] = 100.0;

    void* Data;
    Data = Device.mapMemory(InBufferMemory, 0, BufferSize);
    std::memcpy(Data, InputData.data(), BufferSize);
    Device.unmapMemory(InBufferMemory);

    Device.bindBufferMemory(InBuffer, InBufferMemory, 0);
    Device.bindBufferMemory(OutBuffer, OutBufferMemory, 0);

    // Load Shader
    std::vector<char> ShaderContents;
    if (std::ifstream ShaderFile{ "argmax.spv", std::ios::binary | std::ios::ate }) {
        const size_t FileSize = ShaderFile.tellg();
        ShaderFile.seekg(0);
        ShaderContents.resize(FileSize, '\0');
        ShaderFile.read(ShaderContents.data(), FileSize);
    }

    vk::ShaderModuleCreateInfo ShaderModuleCreateInfo(
        vk::ShaderModuleCreateFlags(),
        ShaderContents.size(),
        reinterpret_cast<const uint32_t*>(ShaderContents.data())
    );
    vk::ShaderModule ShaderModule = Device.createShaderModule(ShaderModuleCreateInfo);

    const std::vector<vk::DescriptorSetLayoutBinding> DescriptorSetLayoutBindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},  // Input float array
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}   // Output unsigned int
    };

    vk::DescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo(
        vk::DescriptorSetLayoutCreateFlags(),
        DescriptorSetLayoutBindings
    );
    vk::DescriptorSetLayout DescriptorSetLayout = Device.createDescriptorSetLayout(DescriptorSetLayoutCreateInfo);

    vk::PipelineLayoutCreateInfo PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), DescriptorSetLayout);
    vk::PipelineLayout PipelineLayout = Device.createPipelineLayout(PipelineLayoutCreateInfo);

    vk::PipelineCache PipelineCache = Device.createPipelineCache(vk::PipelineCacheCreateInfo());

    vk::PipelineShaderStageCreateInfo PipelineShaderCreateInfo(
        vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eCompute,
        ShaderModule,
        "main"
    );

    vk::ComputePipelineCreateInfo ComputePipelineCreateInfo(
        vk::PipelineCreateFlags(),
        PipelineShaderCreateInfo,
        PipelineLayout
    );

    vk::ResultValue<vk::Pipeline> ComputePipeline = Device.createComputePipeline(PipelineCache, ComputePipelineCreateInfo);

    // Descriptor pool and sets
    vk::DescriptorPoolSize DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 3);
    vk::DescriptorPoolCreateInfo DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1, DescriptorPoolSize);
    vk::DescriptorPool DescriptorPool = Device.createDescriptorPool(DescriptorPoolCreateInfo);

    vk::DescriptorSetAllocateInfo DescriptorSetAllocInfo(DescriptorPool, 1, &DescriptorSetLayout);
    const std::vector<vk::DescriptorSet> DescriptorSets = Device.allocateDescriptorSets(DescriptorSetAllocInfo);
    vk::DescriptorSet DescriptorSet = DescriptorSets.front();

    vk::DescriptorBufferInfo InBufferInfo(InBuffer, 0, BufferSize);
    vk::DescriptorBufferInfo OutBufferInfo(OutBuffer, 0, sizeof(uint32_t));

    const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
        {DescriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &InBufferInfo},
        {DescriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &OutBufferInfo}
    };
    Device.updateDescriptorSets(WriteDescriptorSets, nullptr);

    // Create command pool and allocate command buffer
    vk::CommandPoolCreateInfo CommandPoolCreateInfo(
        vk::CommandPoolCreateFlags(),
        ComputeQueueFamilyIndex
    );
    vk::CommandPool CommandPool = Device.createCommandPool(CommandPoolCreateInfo);

    vk::CommandBufferAllocateInfo CommandBufferAllocInfo(
        CommandPool,
        vk::CommandBufferLevel::ePrimary,
        1
    );
    std::vector<vk::CommandBuffer> CommandBuffers = Device.allocateCommandBuffers(CommandBufferAllocInfo);
    vk::CommandBuffer CommandBuffer = CommandBuffers.front();

    // Begin command buffer recording
    vk::CommandBufferBeginInfo BeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    CommandBuffer.begin(BeginInfo);

    // Bind pipeline and descriptor sets
    CommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, ComputePipeline.value);
    CommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, PipelineLayout, 0, DescriptorSet, nullptr);

    // Dispatch compute work
    CommandBuffer.dispatch(ELEMENT_SIZE / WORKGROUP_SIZE, 1, 1);

    // End command buffer recording
    CommandBuffer.end();

    // Submit command buffer to queue
    vk::SubmitInfo SubmitInfo(0, nullptr, nullptr, 1, &CommandBuffer);
    ComputeQueue.submit(SubmitInfo, nullptr);
    ComputeQueue.waitIdle();

    // Map output buffer and retrieve results
    uint32_t* OutData = reinterpret_cast<uint32_t*>(Device.mapMemory(OutBufferMemory, 0, sizeof(uint32_t)));
    std::cout << "Result = " << *OutData << std::endl;
    Device.unmapMemory(OutBufferMemory);

    // Cleanup
    Device.freeCommandBuffers(CommandPool, CommandBuffers);
    Device.destroyCommandPool(CommandPool);
    Device.destroyDescriptorPool(DescriptorPool);
    Device.destroyPipeline(ComputePipeline.value);
    Device.destroyPipelineLayout(PipelineLayout);
    Device.destroyDescriptorSetLayout(DescriptorSetLayout);
    Device.destroyShaderModule(ShaderModule);
    Device.destroyBuffer(InBuffer);
    Device.destroyBuffer(OutBuffer);
    Device.freeMemory(InBufferMemory);
    Device.freeMemory(OutBufferMemory);
    Device.destroy();
    Instance.destroy();

    return EXIT_SUCCESS;
}
