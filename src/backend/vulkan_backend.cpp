#include "vulkan_backend.hpp"
#include "../storage.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace munet {

// --- Global Vulkan State (Simplified for minimal backend) ---
static VkInstance instance;
static VkPhysicalDevice physicalDevice;
static VkDevice device;
static VkQueue computeQueue;
static uint32_t queueFamilyIndex;
static VkCommandPool commandPool;
static VkDescriptorSetLayout descriptorSetLayout;
static VkDescriptorPool descriptorPool;
static VkDescriptorSet descriptorSet;

// Pipeline State
static VkPipelineLayout pipelineLayout;
static VkPipeline addPipeline;
static VkPipeline mulPipeline;
static VkPipeline subPipeline;
static VkPipeline updatePipeline;
static VkPipeline matmulPipeline;
static VkPipeline reluPipeline;
static VkPipeline reluBackwardPipeline;

// Memory mappings to translate CPU void* back to Vulkan handles
static std::unordered_map<void *, VkBuffer> bufferMap;
static std::unordered_map<void *, VkDeviceMemory> memoryMap;

// Helper to compile GLSL to SPIR-V using the Vulkan SDK
static std::vector<uint32_t> compileShader(const std::string &name,
                                           const std::string &source) {
  std::string compPath = "/tmp/" + name + ".comp";
  std::string spvPath = "/tmp/" + name + ".spv";
  std::ofstream out(compPath);
  out << source;
  out.close();

  std::string cmd = "glslc " + compPath + " -o " + spvPath;
  if (std::system(cmd.c_str()) != 0) {
    throw std::runtime_error("Failed to compile shader. Is the Vulkan SDK "
                             "(glslc) installed and in your PATH?");
  }

  std::ifstream in(spvPath, std::ios::binary | std::ios::ate);
  size_t size = in.tellg();
  in.seekg(0);
  std::vector<uint32_t> buffer(size / 4);
  in.read((char *)buffer.data(), size);
  return buffer;
}

static VkShaderModule createShaderModule(const std::vector<uint32_t> &code) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size() * 4;
  createInfo.pCode = code.data();
  VkShaderModule module;
  vkCreateShaderModule(device, &createInfo, nullptr, &module);
  return module;
}

static uint32_t findMemoryType(uint32_t typeFilter,
                               VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable Vulkan memory type!");
}

VulkanBackend::VulkanBackend() {
  // 1. Instance
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "MuNet";
  appInfo.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;
  vkCreateInstance(&createInfo, nullptr, &instance);

  // 2. Physical Device
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
  physicalDevice = devices[0]; // Pick first GPU

  // 3. Queue Family
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           queueFamilies.data());

  queueFamilyIndex = 0;
  for (uint32_t i = 0; i < queueFamilies.size(); i++) {
    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      queueFamilyIndex = i;
      break;
    }
  }

  // 4. Logical Device
  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceCreateInfo{};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  deviceCreateInfo.queueCreateInfoCount = 1;
  vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
  vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);

  // 5. Command Pool
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = queueFamilyIndex;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

  // 6. Descriptor Layout (3 Storage Buffers)
  VkDescriptorSetLayoutBinding bindings[3] = {};
  for (int i = 0; i < 3; i++) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 3;
  layoutInfo.pBindings = bindings;
  vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                              &descriptorSetLayout);

  // 7. Descriptor Pool & Set
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 3;
  VkDescriptorPoolCreateInfo descPoolInfo{};
  descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descPoolInfo.poolSizeCount = 1;
  descPoolInfo.pPoolSizes = &poolSize;
  descPoolInfo.maxSets = 1;
  vkCreateDescriptorPool(device, &descPoolInfo, nullptr, &descriptorPool);

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &descriptorSetLayout;
  vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

  // 8. Pipeline Layout (Push constants for size)
  VkPushConstantRange pushRange{};
  pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushRange.offset = 0;
  pushRange.size = sizeof(int) * 5; // Max 5 ints for Matmul

  VkPipelineLayoutCreateInfo pLayoutInfo{};
  pLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pLayoutInfo.setLayoutCount = 1;
  pLayoutInfo.pSetLayouts = &descriptorSetLayout;
  pLayoutInfo.pushConstantRangeCount = 1;
  pLayoutInfo.pPushConstantRanges = &pushRange;
  vkCreatePipelineLayout(device, &pLayoutInfo, nullptr, &pipelineLayout);

  // 9. Compile and Create Pipelines
  auto createComputePipeline = [&](const std::string &name,
                                   const std::string &glsl) {
    VkShaderModule sm = createShaderModule(compileShader(name, glsl));
    VkComputePipelineCreateInfo compInfo{};
    compInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compInfo.layout = pipelineLayout;
    compInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.module = sm;
    compInfo.stage.pName = "main";
    VkPipeline pipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compInfo, nullptr,
                             &pipeline);
    vkDestroyShaderModule(device, sm, nullptr);
    return pipeline;
  };

  addPipeline = createComputePipeline(
      "add",
      R"(                                                                                                                                                 
        #version 450                                                                                                                                      
        layout(local_size_x = 256) in;                                                                                                                    
        layout(binding = 0) buffer A { float a[]; };                                                                                                      
        layout(binding = 1) buffer B { float b[]; };                                                                                                      
        layout(binding = 2) buffer C { float c[]; };                                                                                                      
        layout(push_constant) uniform Push { uint N; } p;                                                                                                 
        void main() { if (gl_GlobalInvocationID.x < p.N) c[gl_GlobalInvocationID.x] = a[gl_GlobalInvocationID.x] + b[gl_GlobalInvocationID.x]; }          
    )");

  mulPipeline = createComputePipeline(
      "mul",
      R"(                                                                                                                                                 
        #version 450                                                                                                                                      
        layout(local_size_x = 256) in;                                                                                                                    
        layout(binding = 0) buffer A { float a[]; };                                                                                                      
        layout(binding = 1) buffer B { float b[]; };                                                                                                      
        layout(binding = 2) buffer C { float c[]; };                                                                                                      
        layout(push_constant) uniform Push { uint N; } p;                                                                                                 
        void main() { if (gl_GlobalInvocationID.x < p.N) c[gl_GlobalInvocationID.x] = a[gl_GlobalInvocationID.x] * b[gl_GlobalInvocationID.x]; }          
    )");

  reluPipeline = createComputePipeline(
      "relu",
      R"(                                                                                                                                                 
        #version 450                                                                                                                                      
        layout(local_size_x = 256) in;                                                                                                                    
        layout(binding = 0) buffer A { float a[]; };                                                                                                      
        layout(binding = 1) buffer C { float c[]; };                                                                                                      
        layout(push_constant) uniform Push { uint N; } p;                                                                                                 
        void main() {                                                                                                                                     
            uint i = gl_GlobalInvocationID.x;                                                                                                             
            if (i < p.N) c[i] = a[i] > 0.0 ? a[i] : 0.0;                                                                                                  
        }                                                                                                                                                 
    )");

  reluBackwardPipeline = createComputePipeline(
      "relu_backward",
      R"(                                                                                                                                                 
                #version 450                                                                                                                              
                layout(local_size_x = 256) in;                                                                                                            
                layout(binding = 0) buffer GradOut { float grad_out[]; };                                                                                 
                layout(binding = 1) buffer In { float in_data[]; };                                                                                       
                layout(binding = 2) buffer GradIn { float grad_in[]; };                                                                                   
                layout(push_constant) uniform Push { uint N; } p;                                                                                         
                void main() {                                                                                                                             
                        uint i = gl_GlobalInvocationID.x;                                                                                                 
                        if (i < p.N) grad_in[i] = in_data[i] > 0.0 ? grad_out[i] : 0.0;                                                                   
                }                                                                                                                                         
    )");

  matmulPipeline = createComputePipeline(
      "matmul",
      R"(                                                                                                                                                 
        #version 450                                                                                                                                      
        layout(local_size_x = 16, local_size_y = 16) in;                                                                                                  
        layout(binding = 0) buffer A { float a[]; };                                                                                                      
        layout(binding = 1) buffer B { float b[]; };                                                                                                      
        layout(binding = 2) buffer C { float c[]; };                                                                                                      
        layout(push_constant) uniform Push { int M, K, N, tA, tB; } p;                                                                                    
        void main() {                                                                                                                                     
            int m = int(gl_GlobalInvocationID.x);                                                                                                         
            int n = int(gl_GlobalInvocationID.y);                                                                                                         
            if (m < p.M && n < p.N) {                                                                                                                     
                float sum = 0.0;                                                                                                                          
                for (int k = 0; k < p.K; ++k) {                                                                                                           
                    float a_val = p.tA == 1 ? a[k * p.M + m] : a[m * p.K + k];                                                                            
                    float b_val = p.tB == 1 ? b[n * p.K + k] : b[k * p.N + n];                                                                            
                    sum += a_val * b_val;                                                                                                                 
                }                                                                                                                                         
                c[m * p.N + n] = sum;                                                                                                                     
            }                                                                                                                                             
        }                                                                                                                                                 
    )");

  subPipeline = createComputePipeline(
      "sub",
      R"(                                                                                                          
        #version 450                                                                                                                                      
        layout(local_size_x = 256) in;                                                                                                                    
        layout(binding = 0) buffer A { float a[]; };                                                                                                      
        layout(binding = 1) buffer B { float b[]; };                                                                                                      
        layout(binding = 2) buffer C { float c[]; };                                                                                                      
        layout(push_constant) uniform Push { uint N; } p;                                                                                                 
        void main() {                                                                                                                                     
            uint i = gl_GlobalInvocationID.x;                                                                                                             
            if (i < p.N) c[i] = a[i] - b[i];                                                                                                              
        }                                                                                                                                                 
    )");

  updatePipeline = createComputePipeline(
      "update",
      R"(                                                                                                    
        #version 450                                                                                                                                      
        layout(local_size_x = 256) in;                                                                                                                    
        layout(binding = 0) buffer W { float w[]; };                                                                                                      
        layout(binding = 1) buffer G { float g[]; };                                                                                                      
        layout(push_constant) uniform Push { uint N; float lr; } p;                                                                                       
        void main() {                                                                                                                                     
            uint i = gl_GlobalInvocationID.x;                                                                                                             
            if (i < p.N) w[i] -= p.lr * g[i];                                                                                                             
        }                                                                                                                                                 
    )");
}

VulkanBackend::~VulkanBackend() {
  vkQueueWaitIdle(computeQueue);
  // Destructor omitted for brevity. Real system would vkDestroy everything
  // here.
}

void *VulkanBackend::allocate(size_t bytes) {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bytes;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer buffer;
  vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

  VkMemoryRequirements memReqs;
  vkGetBufferMemoryRequirements(device, buffer, &memReqs);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  // Request memory that we can map to the CPU natively
  allocInfo.memoryTypeIndex = findMemoryType(
      memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  VkDeviceMemory memory;
  vkAllocateMemory(device, &allocInfo, nullptr, &memory);
  vkBindBufferMemory(device, buffer, memory, 0);

  void *mapped_data;
  vkMapMemory(device, memory, 0, bytes, 0, &mapped_data);

  // Store handles so we can retrieve the VkBuffer when a kernel is launched
  // using just the void*
  bufferMap[mapped_data] = buffer;
  memoryMap[mapped_data] = memory;

  return mapped_data;
}

void VulkanBackend::deallocate(void *ptr) {
  if (bufferMap.count(ptr)) {
    vkUnmapMemory(device, memoryMap[ptr]);
    vkDestroyBuffer(device, bufferMap[ptr], nullptr);
    vkFreeMemory(device, memoryMap[ptr], nullptr);
    bufferMap.erase(ptr);
    memoryMap.erase(ptr);
  }
}

void VulkanBackend::memset(void *ptr, int value, size_t bytes) {
  std::memset(ptr, value, bytes);
}

void VulkanBackend::copy(const void *src, void *dst, size_t bytes,
                         Device src_dev, Device dst_dev) {
  // Because we used HOST_VISIBLE memory, the GPU memory is already mapped to
  // CPU space! A simple std::memcpy performs a zero-copy direct transfer over
  // the PCIe bus.
  std::memcpy(dst, src, bytes);
}

void VulkanBackend::synchronize() { vkQueueWaitIdle(computeQueue); }

void VulkanBackend::all_reduce(Storage &buffer, size_t num_elements) {}

// --- Kernel Execution Helper ---
static void dispatch(VkPipeline pipeline, const std::vector<void *> &buffers,
                     void *pushConstants, size_t pcSize, int bx, int by,
                     int bz) {
  // Update Descriptor Sets
  std::vector<VkDescriptorBufferInfo> bInfos(buffers.size());
  std::vector<VkWriteDescriptorSet> writes(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    bInfos[i].buffer = bufferMap[buffers[i]];
    bInfos[i].offset = 0;
    bInfos[i].range = VK_WHOLE_SIZE;

    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descriptorSet;
    writes[i].dstBinding = i;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].descriptorCount = 1;
    writes[i].pBufferInfo = &bInfos[i];
  }
  vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);

  // Record Command Buffer
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device, &allocInfo, &cmd);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &beginInfo);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                          0, 1, &descriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     pcSize, pushConstants);

  vkCmdDispatch(cmd, bx, by, bz);
  vkEndCommandBuffer(cmd);

  // Submit and Wait
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;
  vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(computeQueue);

  vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

// --- Compute Kernels ---
void VulkanBackend::add(const Storage &a, const Storage &b, Storage &out,
                        size_t num_elements) {
  uint32_t N = num_elements;
  dispatch(addPipeline, {a.data(), b.data(), out.data()}, &N, sizeof(uint32_t),
           (N + 255) / 256, 1, 1);
}

void VulkanBackend::mul(const Storage &a, const Storage &b, Storage &out,
                        size_t num_elements) {
  uint32_t N = num_elements;
  dispatch(mulPipeline, {a.data(), b.data(), out.data()}, &N, sizeof(uint32_t),
           (N + 255) / 256, 1, 1);
}

void VulkanBackend::matmul(const Storage &a, const Storage &b, Storage &out,
                           int M, int K, int N, bool transA, bool transB) {
  struct {
    int m, k, n, ta, tb;
  } pc = {M, K, N, transA ? 1 : 0, transB ? 1 : 0};
  dispatch(matmulPipeline, {a.data(), b.data(), out.data()}, &pc, sizeof(pc),
           (M + 15) / 16, (N + 15) / 16, 1);
}

void VulkanBackend::relu(const Storage &in, Storage &out, size_t num_elements) {
  uint32_t N = num_elements;
  dispatch(reluPipeline, {in.data(), out.data()}, &N, sizeof(uint32_t),
           (N + 255) / 256, 1, 1);
}

void VulkanBackend::relu_backward(const Storage &grad_out, const Storage &input,
                                  Storage &grad_in, size_t num_elements) {
  uint32_t N = num_elements;
  dispatch(reluBackwardPipeline,
           {grad_out.data(), input.data(), grad_in.data()}, &N,
           sizeof(uint32_t), (N + 255) / 256, 1, 1);
}

void VulkanBackend::sub(const Storage &a, const Storage &b, Storage &out,
                        size_t num_elements) {
  uint32_t N = num_elements;
  dispatch(subPipeline, {a.data(), b.data(), out.data()}, &N, sizeof(uint32_t),
           (N + 255) / 256, 1, 1);
}

void VulkanBackend::update(Storage &weight, const Storage &grad, float lr,
                           size_t num_elements) {
  struct {
    uint32_t N;
    float lr;
  } pc = {(uint32_t)num_elements, lr};
  dispatch(updatePipeline, {weight.data(), grad.data()}, &pc, sizeof(pc),
           (num_elements + 255) / 256, 1, 1);
}

} // namespace munet
