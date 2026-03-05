#include "vulkan_backend.hpp"
#include "../profiler.hpp"
#include "../storage.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// --- Error Handling Macro ---
#define VK_CHECK(call)                                                         \
  do {                                                                         \
    VkResult result = call;                                                    \
    if (result != VK_SUCCESS) {                                                \
      throw std::runtime_error(std::string("Vulkan Error at line ") +          \
                               std::to_string(__LINE__) + ": " +               \
                               std::to_string(result));                        \
    }                                                                          \
  } while (0)

namespace munet {

// --- Configuration ---
static const int MAX_FRAMES_IN_FLIGHT = 2;
static const int BATCH_SIZE_LIMIT = 1000; // Flush after this many ops
static const int MAX_DESCRIPTORS_PER_FRAME = 2048;

// --- Global Vulkan State ---
static VkInstance instance;
static VkPhysicalDevice physicalDevice;
static VkDevice device;
static VkQueue computeQueue;
static uint32_t queueFamilyIndex;
static VkCommandPool commandPool;
static VkDescriptorSetLayout descriptorSetLayout;

// Per-Frame Resources
static VkDescriptorPool descriptorPools[MAX_FRAMES_IN_FLIGHT];
static VkCommandBuffer commandBuffers[MAX_FRAMES_IN_FLIGHT];
static VkFence inFlightFences[MAX_FRAMES_IN_FLIGHT];
static VkQueryPool queryPools[MAX_FRAMES_IN_FLIGHT];

static float timestampPeriod = 1.0f;

// Batching State
static int currentFrame = 0;
static int currentBatchSize = 0;
static bool isRecording = false;

// Profiling names for the current batch
static std::vector<std::string> batchProfileNames[MAX_FRAMES_IN_FLIGHT];

// --- Allocator State ---
static std::unordered_map<size_t, std::vector<uint64_t>> free_pool;
static std::unordered_map<uint64_t, size_t> allocation_sizes;
static std::unordered_map<uint64_t, VkDeviceMemory> allocation_memory;
static std::vector<uint64_t> deferred_frees[MAX_FRAMES_IN_FLIGHT];

// --- Persistent Staging Buffers ---
static VkBuffer stagingBuffer = VK_NULL_HANDLE;
static VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
static size_t stagingOffset = 0;
static size_t stagingSize = 0;
static void *stagingMapped = nullptr;
static VkCommandBuffer immediateCmdBuffer = VK_NULL_HANDLE;

static VkPipelineLayout pipelineLayout;
static VkPipeline addPipeline, mulPipeline, subPipeline, updatePipeline;
static VkPipeline matmulPipeline, reluPipeline, reluBackwardPipeline;
static VkPipeline sigmoidPipeline, sigmoidBackwardPipeline;
static VkPipeline softmaxPipeline, softmaxBackwardPipeline;
static VkPipeline mseLossPipeline, mseLossBackwardPipeline;
static VkPipeline crossEntropyPipeline, crossEntropyBackwardPipeline;

// --- Helpers ---
static size_t round_up_alloc_size(size_t bytes) {
  if (bytes == 0)
    bytes = 4; // Minimal valid size
  // Round up to next power of 2 if small, or multiple of 1MB if large
  if (bytes > 16 * 1024 * 1024) {
    size_t align = 2 * 1024 * 1024;
    return ((bytes + align - 1) / align) * align;
  }
  size_t p = 256; // Minimum alignment usually
  while (p < bytes)
    p <<= 1;
  return p;
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

static void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags properties, VkBuffer &buffer,
                         VkDeviceMemory &bufferMemory) {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));

  VkMemoryRequirements memReqs;
  vkGetBufferMemoryRequirements(device, buffer, &memReqs);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memReqs.memoryTypeBits, properties);

  VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory));
  VK_CHECK(vkBindBufferMemory(device, buffer, bufferMemory, 0));
}

static std::vector<uint32_t> compileShader(const std::string &name,
                                           const std::string &source) {
  std::string compPath = "/tmp/" + name + ".comp";
  std::string spvPath = "/tmp/" + name + ".spv";
  std::ofstream out(compPath);
  out << source;
  out.close();
  std::string cmd = "glslc -O " + compPath + " -o " + spvPath;
  if (std::system(cmd.c_str()) != 0)
    throw std::runtime_error("Failed to compile shader '" + name +
                             "'. Ensure 'glslc' is installed (Vulkan SDK).");
  std::ifstream in(spvPath, std::ios::binary | std::ios::ate);
  size_t size = in.tellg();
  in.seekg(0);
  std::vector<uint32_t> buffer(size / 4);
  in.read((char *)buffer.data(), size);
  return buffer;
}

VulkanBackend::VulkanBackend() {
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;
  VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
  physicalDevice = devices[0];

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physicalDevice, &props);
  timestampPeriod = props.limits.timestampPeriod;

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           queueFamilies.data());
  for (uint32_t i = 0; i < queueFamilies.size(); i++) {
    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      queueFamilyIndex = i;
      break;
    }
  }

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
  VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));
  vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);

  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = queueFamilyIndex;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));

  // Layouts: Bindings 0 through 7
  VkDescriptorSetLayoutBinding bindings[8] = {};
  for (int i = 0; i < 8; i++) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 8;
  layoutInfo.pBindings = bindings;
  VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                       &descriptorSetLayout));

  VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                sizeof(int) * 6};
  VkPipelineLayoutCreateInfo pLayoutInfo{};
  pLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pLayoutInfo.setLayoutCount = 1;
  pLayoutInfo.pSetLayouts = &descriptorSetLayout;
  pLayoutInfo.pushConstantRangeCount = 1;
  pLayoutInfo.pPushConstantRanges = &pushRange;
  VK_CHECK(
      vkCreatePipelineLayout(device, &pLayoutInfo, nullptr, &pipelineLayout));

  // Allocate immediate command buffer for fast copies
  VkCommandBufferAllocateInfo immAlloc{};
  immAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  immAlloc.commandPool = commandPool;
  immAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  immAlloc.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(device, &immAlloc, &immediateCmdBuffer));

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
  VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers));

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VkQueryPoolCreateInfo queryPoolInfo{};
  queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  queryPoolInfo.queryCount = 2; // Index 0 (Start), Index 1 (End)

  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]));
#ifdef ENABLE_PROFILING
    VK_CHECK(
        vkCreateQueryPool(device, &queryPoolInfo, nullptr, &queryPools[i]));
#endif
    // Create one descriptor pool per frame
    // We have 8 bindings per set. So poolSize should be roughly 8x maxSets
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                  MAX_DESCRIPTORS_PER_FRAME * 8};
    VkDescriptorPoolCreateInfo descPoolInfo{};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.poolSizeCount = 1;
    descPoolInfo.pPoolSizes = &poolSize;
    descPoolInfo.maxSets = MAX_DESCRIPTORS_PER_FRAME;
    VK_CHECK(vkCreateDescriptorPool(device, &descPoolInfo, nullptr,
                                    &descriptorPools[i]));
  }

  // Load Kernels
  auto createComputePipeline = [&](const std::string &name,
                                   const std::string &glsl) {
    std::vector<uint32_t> code = compileShader(name, glsl);
    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = code.size() * 4;
    smInfo.pCode = code.data();
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, nullptr, &sm));
    VkComputePipelineCreateInfo compInfo{};
    compInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compInfo.layout = pipelineLayout;
    compInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.module = sm;
    compInfo.stage.pName = "main";
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compInfo,
                                      nullptr, &pipeline));
    vkDestroyShaderModule(device, sm, nullptr);
    return pipeline;
  };

  addPipeline = createComputePipeline("add",
                                      R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer A { float a[]; };
        layout(binding = 1) buffer B { float b[]; };
        layout(binding = 2) buffer C { float c[]; };
        layout(push_constant) uniform Push { uint N; } p;
        void main() { if (gl_GlobalInvocationID.x < p.N) c[gl_GlobalInvocationID.x] = a[gl_GlobalInvocationID.x] + b[gl_GlobalInvocationID.x]; }
    )");
  mulPipeline = createComputePipeline("mul",
                                      R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer A { float a[]; };
        layout(binding = 1) buffer B { float b[]; };
        layout(binding = 2) buffer C { float c[]; };
        layout(push_constant) uniform Push { uint N; } p;
        void main() { if (gl_GlobalInvocationID.x < p.N) c[gl_GlobalInvocationID.x] = a[gl_GlobalInvocationID.x] * b[gl_GlobalInvocationID.x]; }
    )");
  subPipeline = createComputePipeline("sub",
                                      R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer A { float a[]; };
        layout(binding = 1) buffer B { float b[]; };
        layout(binding = 2) buffer C { float c[]; };
        layout(push_constant) uniform Push { uint N; } p;
        void main() { if (gl_GlobalInvocationID.x < p.N) c[gl_GlobalInvocationID.x] = a[gl_GlobalInvocationID.x] - b[gl_GlobalInvocationID.x]; }
    )");
  updatePipeline = createComputePipeline("update",
                                         R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer W { float w[]; };
        layout(binding = 1) buffer G { float g[]; };
        layout(push_constant) uniform Push { uint N; float lr; } p;
        void main() { if (gl_GlobalInvocationID.x < p.N) w[gl_GlobalInvocationID.x] -= p.lr * g[gl_GlobalInvocationID.x]; }
    )");
  reluPipeline = createComputePipeline("relu",
                                       R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer A { float a[]; };
        layout(binding = 1) buffer C { float c[]; };
        layout(push_constant) uniform Push { uint N; } p;
        void main() { uint i = gl_GlobalInvocationID.x; if (i < p.N) c[i] = max(0.0, a[i]); }
    )");
  reluBackwardPipeline = createComputePipeline("relu_back",
                                               R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer GO { float go[]; };
        layout(binding = 1) buffer I { float i[]; };
        layout(binding = 2) buffer GI { float gi[]; };
        layout(push_constant) uniform Push { uint N; } p;
        void main() { uint id = gl_GlobalInvocationID.x; if (id < p.N) gi[id] = (i[id] > 0.0) ? go[id] : 0.0; }
    )");
  sigmoidPipeline = createComputePipeline("sigmoid",
                                          R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer A { float a[]; };
        layout(binding = 1) buffer C { float c[]; };
        layout(push_constant) uniform Push { uint N; } p;
        void main() { uint i = gl_GlobalInvocationID.x; if (i < p.N) c[i] = 1.0 / (1.0 + exp(-a[i])); }
    )");
  sigmoidBackwardPipeline = createComputePipeline("sigmoid_back",
                                                  R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer GO { float go[]; };
        layout(binding = 1) buffer O { float out_d[]; };
        layout(binding = 2) buffer GI { float gi[]; };
        layout(push_constant) uniform Push { uint N; } p;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i < p.N) { float s = out_d[i]; gi[i] = go[i] * s * (1.0 - s); }
        }
    )");
  softmaxPipeline = createComputePipeline("softmax",
                                          R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer I { float in_d[]; };
        layout(binding = 1) buffer O { float out_d[]; };
        layout(push_constant) uniform P { int B, C; } u;
        void main() {
            int b = int(gl_GlobalInvocationID.x);
            if (b < u.B) {
                float max_val = -1e30;
                for (int i = 0; i < u.C; ++i) { float v = in_d[b * u.C + i]; if (v > max_val) max_val = v; }
                float sum_exp = 0.0;
                for (int i = 0; i < u.C; ++i) { float e = exp(in_d[b * u.C + i] - max_val); out_d[b * u.C + i] = e; sum_exp += e; }
                for (int i = 0; i < u.C; ++i) out_d[b * u.C + i] /= sum_exp;
            }
        }
    )");
  softmaxBackwardPipeline = createComputePipeline("softmax_back",
                                                  R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer GO { float go[]; };
        layout(binding = 1) buffer O { float out_d[]; };
        layout(binding = 2) buffer GI { float gi[]; };
        layout(push_constant) uniform P { int B, C; } u;
        void main() {
            int b = int(gl_GlobalInvocationID.x);
            if (b < u.B) {
                float dot = 0.0;
                for (int i = 0; i < u.C; ++i) dot += out_d[b * u.C + i] * go[b * u.C + i];
                for (int i = 0; i < u.C; ++i) gi[b * u.C + i] = out_d[b * u.C + i] * (go[b * u.C + i] - dot);
            }
        }
    )");

  mseLossPipeline = createComputePipeline("mse_loss",
                                          R"(
         #version 450
         layout(local_size_x = 256) in;
         layout(binding = 0) buffer P { float p[]; };
         layout(binding = 1) buffer T { float t[]; };
         layout(binding = 2) buffer O { uint out_u[]; };
         layout(push_constant) uniform Push { uint N; } u;

         #define ATOMIC_ADD_FLOAT(BUFFER, IDX, VAL) \
         do { \
              uint expected = BUFFER[IDX]; \
              uint current; \
              while(true) { \
                     uint next = floatBitsToUint(uintBitsToFloat(expected) + (VAL)); \
                     current = atomicCompSwap(BUFFER[IDX], expected, next); \
                     if (current == expected) break; \
                     expected = current; \
              } \
         } while(false)

         void main() {
             uint i = gl_GlobalInvocationID.x;
             if (i < u.N) {
                 float diff = p[i] - t[i];
                 float val = (diff * diff) / float(u.N);
                 ATOMIC_ADD_FLOAT(out_u, 0, val);
             }
         }
     )");
  mseLossBackwardPipeline = createComputePipeline("mse_loss_back",
                                                  R"(
         #version 450
         layout(local_size_x = 256) in;
         layout(binding = 0) buffer GO { float go[]; };
         layout(binding = 1) buffer P { float p[]; };
         layout(binding = 2) buffer T { float t[]; };
         layout(binding = 3) buffer GI { float gi[]; };
         layout(push_constant) uniform Push { uint N; } u;
         void main() {
             uint i = gl_GlobalInvocationID.x;
             if (i < u.N) {
                 gi[i] = go[0] * 2.0 * (p[i] - t[i]) / float(u.N);
             }
         }
     )");

  crossEntropyPipeline = createComputePipeline("ce_loss",
                                               R"(
         #version 450
         layout(local_size_x = 256) in;
         layout(binding = 0) buffer L { float logits[]; };
         layout(binding = 1) buffer T { float targets[]; };
         layout(binding = 2) buffer O { uint out_u[]; };
         layout(push_constant) uniform Push { int B; int C; } u;

         #define ATOMIC_ADD_FLOAT(BUFFER, IDX, VAL) \
         do { \
              uint expected = BUFFER[IDX]; \
              uint current; \
              while(true) { \
                     uint next = floatBitsToUint(uintBitsToFloat(expected) + (VAL)); \
                     current = atomicCompSwap(BUFFER[IDX], expected, next); \
                     if (current == expected) break; \
                     expected = current; \
              } \
         } while(false)

         void main() {
             int b = int(gl_GlobalInvocationID.x);
             if (b < u.B) {
                 float max_val = -1e30;
                 for (int i=0; i<u.C; ++i) max_val = max(max_val, logits[b*u.C + i]);
                 float sum_exp = 0.0;
                 for (int i=0; i<u.C; ++i) sum_exp += exp(logits[b*u.C + i] - max_val);
                 float loss = 0.0;
                 for (int i=0; i<u.C; ++i) {
                     float prob = exp(logits[b*u.C + i] - max_val) / sum_exp;
                     float target = targets[b*u.C + i];
                     if (target > 0.0) loss -= target * log(prob + 1e-9);
                 }
                 ATOMIC_ADD_FLOAT(out_u, 0, loss / float(u.B));
             }
         }
     )");

  crossEntropyBackwardPipeline = createComputePipeline("ce_loss_back",
                                                       R"(
         #version 450
         layout(local_size_x = 256) in;
         layout(binding = 0) buffer GO { float go[]; };
         layout(binding = 1) buffer L { float logits[]; };
         layout(binding = 2) buffer T { float targets[]; };
         layout(binding = 3) buffer GI { float gi[]; };
         layout(push_constant) uniform Push { int B; int C; } u;
         void main() {
             int b = int(gl_GlobalInvocationID.x);
             if (b < u.B) {
                 float max_val = -1e30;
                 for (int i=0; i<u.C; ++i) max_val = max(max_val, logits[b*u.C + i]);
                 float sum_exp = 0.0;
                 for (int i=0; i<u.C; ++i) sum_exp += exp(logits[b*u.C + i] - max_val);
                 float go_val = go[0];
                 for (int i=0; i<u.C; ++i) {
                     float prob = exp(logits[b*u.C + i] - max_val) / sum_exp;
                     gi[b*u.C + i] = go_val * (prob - targets[b*u.C + i]) / float(u.B);
                 }
             }
         }
     )");

  matmulPipeline = createComputePipeline("matmul",
                                         R"(
        #version 450
        layout(local_size_x = 16, local_size_y = 16) in;
        layout(binding = 0) buffer A { float a[]; };
        layout(binding = 1) buffer B { float b[]; };
        layout(binding = 2) buffer C { float c[]; };
        layout(push_constant) uniform Push { int M, K, N, tA, tB; } p;
        void main() {
            int n = int(gl_GlobalInvocationID.x); // n maps to N (columns), fast dimension
            int m = int(gl_GlobalInvocationID.y); // m maps to M (rows), slow dimension
            if (m < p.M && n < p.N) {
                float sum = 0.0;

                // Branch manually hoisted out of the hot loop to match CUDA performance
                if (p.tA == 0 && p.tB == 0) {
                    for (int k = 0; k < p.K; ++k) sum += a[m * p.K + k] * b[k * p.N + n];
                } else if (p.tA == 1 && p.tB == 0) {
                    for (int k = 0; k < p.K; ++k) sum += a[k * p.M + m] * b[k * p.N + n];
                } else if (p.tA == 0 && p.tB == 1) {
                    for (int k = 0; k < p.K; ++k) sum += a[m * p.K + k] * b[n * p.K + k];
                } else {
                    for (int k = 0; k < p.K; ++k) sum += a[k * p.M + m] * b[n * p.K + k];
                }
                c[m * p.N + n] = sum;
            }
        }
    )");

  conv2dPipeline = createComputePipeline("conv2d",
                                         R"(
         #version 450
         layout(local_size_x=256) in;
         layout(binding=0) buffer I { float in_d[]; };
         layout(binding=1) buffer W { float w_d[]; };
         layout(binding=2) buffer B_ { float b_d[]; }; // Optional
         layout(binding=3) buffer O { float out_d[]; };
         layout(push_constant) uniform P { int B, iC, iH, iW, oC, kH, kW, s, p, oH, oW, has_bias; } u;
         void main() {
                 int idx = int(gl_GlobalInvocationID.x);
                 int total = u.B * u.oC * u.oH * u.oW;
                 if(idx < total) {
                         int ow = idx % u.oW; int tmp = idx / u.oW;
                         int oh = tmp % u.oH; tmp /= u.oH;
                         int oc = tmp % u.oC; int b = tmp / u.oC;
                         float sum = (u.has_bias == 1) ? b_d[oc] : 0.0;

                         for(int ic=0; ic<u.iC; ++ic) {
                                 for(int kh=0; kh<u.kH; ++kh) {
                                         for(int kw=0; kw<u.kW; ++kw) {
                                                 int ih = oh*u.s - u.p + kh;
                                                 int iw = ow*u.s - u.p + kw;
                                                 if(ih>=0 && ih<u.iH && iw>=0 && iw<u.iW) {
                                                         int i_idx = (b * u.iC + ic) * (u.iH * u.iW) + ih * u.iW + iw;
                                                         int w_idx = (oc * u.iC + ic) * (u.kH * u.kW) + kh * u.kW + kw;
                                                         sum += in_d[i_idx] * w_d[w_idx];
                                                 }
                                         }
                                 }
                         }
                         out_d[idx] = sum;
                 }
         }
     )");

  conv2dBackInputPipeline = createComputePipeline("conv2d_bi",
                                                  R"(
         #version 450
         layout(local_size_x=256) in;
         layout(binding=0) buffer GO { float go[]; };
         layout(binding=1) buffer W { float w[]; };
         layout(binding=2) buffer GI { float gi[]; };
         layout(push_constant) uniform P { int B, iC, iH, iW, oC, kH, kW, s, p, oH, oW; } u;
         void main() {
                 int idx = int(gl_GlobalInvocationID.x);
                 int total = u.B * u.iC * u.iH * u.iW;
                 if(idx < total) {
                         int iw = idx % u.iW; int tmp = idx / u.iW;
                         int ih = tmp % u.iH; tmp /= u.iH;
                         int ic = tmp % u.iC; int b = tmp / u.iC;
                         float d_in = 0.0;
                         for(int oc=0; oc<u.oC; ++oc) {
                                 for(int kh=0; kh<u.kH; ++kh) {
                                         for(int kw=0; kw<u.kW; ++kw) {
                                                 int num_h = ih + u.p - kh;
                                                 int num_w = iw + u.p - kw;
                                                 if (num_h >= 0 && num_w >= 0 && num_h % u.s == 0 && num_w % u.s == 0) {
                                                          int oh = num_h / u.s;
                                                          int ow = num_w / u.s;
                                                          if(oh>=0 && oh<u.oH && ow>=0 && ow<u.oW) {
                                                                  int go_idx = (b*u.oC+oc)*(u.oH*u.oW) + oh*u.oW + ow;
                                                                  int w_idx = (oc*u.iC+ic)*(u.kH*u.kW) + kh*u.kW + kw;
                                                                  d_in += go[go_idx] * w[w_idx];
                                                          }
                                                 }
                                         }
                                 }
                         }
                         gi[idx] = d_in;
                 }
         }
     )");

  conv2dBackWeightPipeline = createComputePipeline("conv2d_bw",
                                                   R"(
         #version 450
         layout(local_size_x=256) in;
         layout(binding=0) buffer GO { float go[]; };
         layout(binding=1) buffer I { float in_d[]; };
         layout(binding=2) buffer GW { float gw[]; };
         layout(push_constant) uniform P { int B, iC, iH, iW, oC, kH, kW, s, p, oH, oW; } u;
         void main() {
                 int idx = int(gl_GlobalInvocationID.x);
                 int total = u.oC * u.iC * u.kH * u.kW;
                 if(idx < total) {
                         int kw = idx % u.kW; int tmp = idx / u.kW;
                         int kh = tmp % u.kH; tmp /= u.kH;
                         int ic = tmp % u.iC; int oc = tmp / u.iC;
                         float dw = 0.0;
                         for(int b=0; b<u.B; ++b) {
                                 for(int oh=0; oh<u.oH; ++oh) {
                                         for(int ow=0; ow<u.oW; ++ow) {
                                                 int ih = oh*u.s - u.p + kh;
                                                 int iw = ow*u.s - u.p + kw;
                                                 if(ih>=0 && ih<u.iH && iw>=0 && iw<u.iW) {
                                                          int in_idx = (b*u.iC+ic)*(u.iH*u.iW) + ih*u.iW + iw;
                                                          int go_idx = (b*u.oC+oc)*(u.oH*u.oW) + oh*u.oW + ow;
                                                          dw += go[go_idx] * in_d[in_idx];
                                                 }
                                         }
                                 }
                         }
                         gw[idx] = dw;
                 }
         }
     )");

  conv2dBackBiasPipeline = createComputePipeline("conv2d_bb",
                                                 R"(
         #version 450
         layout(local_size_x=256) in;
         layout(binding=0) buffer GO { float go[]; };
         layout(binding=1) buffer GB { float gb[]; };
         layout(push_constant) uniform P { int B, oC, oH, oW; } u;
         void main() {
                 int oc = int(gl_GlobalInvocationID.x);
                 if(oc < u.oC) {
                         float db = 0.0;
                         for(int b=0; b<u.B; ++b) {
                                 for(int i=0; i<u.oH*u.oW; ++i) {
                                     int go_idx = (b*u.oC+oc)*(u.oH*u.oW) + i;
                                     db += go[go_idx];
                                 }
                         }
                         gb[oc] = db;
                 }
         }
     )");

  maxPoolPipeline = createComputePipeline("maxpool",
                                          R"(
         #version 450
         layout(local_size_x=256) in;
         layout(binding=0) buffer I { float in_d[]; };
         layout(binding=1) buffer O { float out_d[]; };
         layout(push_constant) uniform P { int B, C, iH, iW, k, s, p, oH, oW; } u;
         void main() {
                 int idx = int(gl_GlobalInvocationID.x);
                 int total = u.B*u.C*u.oH*u.oW;
                 if(idx < total) {
                         int ow = idx % u.oW; int tmp = idx / u.oW;
                         int oh = tmp % u.oH; tmp /= u.oH;
                         int c = tmp % u.C; int b = tmp / u.C;
                         float max_val = -1e37;
                         for(int kh=0; kh<u.k; ++kh) {
                                 for(int kw=0; kw<u.k; ++kw) {
                                         int ih = oh*u.s - u.p + kh;
                                         int iw = ow*u.s - u.p + kw;
                                         if(ih>=0 && ih<u.iH && iw>=0 && iw<u.iW) {
                                                 float val = in_d[(b*u.C+c)*(u.iH*u.iW) + ih*u.iW + iw];
                                                 if(val > max_val) max_val = val;
                                         }
                                 }
                         }
                         out_d[idx] = max_val;
                 }
         }
     )");

  maxPoolBackPipeline = createComputePipeline("maxpool_b",
                                              R"(
         #version 450
         layout(local_size_x=256) in;
         layout(binding=0) buffer GO { float go[]; };
         layout(binding=1) buffer I { float in_d[]; };
         layout(binding=2) buffer GI { uint gi_u[]; };
         layout(push_constant) uniform P { int B, C, iH, iW, k, s, p, oH, oW; } u;

         void atomicAddFloat(uint index, float val) {
              uint expected = gi_u[index];
              uint current;
              while(true) {
                     uint next = floatBitsToUint(uintBitsToFloat(expected) + val);
                     current = atomicCompSwap(gi_u[index], expected, next);
                     if (current == expected) break;
                     expected = current;
              }
         }

         void main() {
                 int idx = int(gl_GlobalInvocationID.x);
                 int total = u.B*u.C*u.oH*u.oW;
                 if(idx < total) {
                         int ow = idx % u.oW; int tmp = idx / u.oW;
                         int oh = tmp % u.oH; tmp /= u.oH;
                         int c = tmp % u.C; int b = tmp / u.C;

                         float max_val = -1e37;
                         int max_idx = -1;

                         for(int kh=0; kh<u.k; ++kh) {
                                 for(int kw=0; kw<u.k; ++kw) {
                                         int ih = oh*u.s - u.p + kh;
                                         int iw = ow*u.s - u.p + kw;
                                         if(ih>=0 && ih<u.iH && iw>=0 && iw<u.iW) {
                                                 int i_idx = (b*u.C+c)*(u.iH*u.iW) + ih*u.iW + iw;
                                                 float val = in_d[i_idx];
                                                 if(val > max_val) { max_val = val; max_idx = i_idx; }
                                         }
                                 }
                         }

                         if(max_idx != -1) {
                                 atomicAddFloat(uint(max_idx), go[idx]);
                         }
                 }
         }
     )");

  upsamplePipeline = createComputePipeline("upsample",
                                           R"(
         #version 450
         layout(local_size_x=256) in;
         layout(binding=0) buffer I { float in_d[]; };
         layout(binding=1) buffer O { float out_d[]; };
         layout(push_constant) uniform P { int B, C, iH, iW, scale; } u;
         void main() {
                 int idx = int(gl_GlobalInvocationID.x);
                 int oH = u.iH * u.scale; int oW = u.iW * u.scale;
                 int total = u.B*u.C*oH*oW;
                 if(idx < total) {
                          int ow = idx % oW; int tmp = idx / oW;
                          int oh = tmp % oH; tmp /= oH;
                          int c = tmp % u.C; int b = tmp / u.C;
                          out_d[idx] = in_d[(b*u.C+c)*(u.iH*u.iW) + (oh/u.scale)*u.iW + (ow/u.scale)];
                 }
         }
     )");

  upsampleBackPipeline = createComputePipeline("upsample_b",
                                               R"(
         #version 450
         layout(local_size_x=256) in;
         layout(binding=0) buffer GO { float go[]; };
         layout(binding=1) buffer GI { float gi[]; };
         layout(push_constant) uniform P { int B, C, iH, iW, scale; } u;
         void main() {
                 int idx = int(gl_GlobalInvocationID.x);
                 int total = u.B*u.C*u.iH*u.iW;
                 if(idx < total) {
                          int iw = idx % u.iW; int tmp = idx / u.iW;
                          int ih = tmp % u.iH; tmp /= u.iH;
                          int c = tmp % u.C; int b = tmp / u.C;
                          float sum = 0.0;
                          int oH = u.iH*u.scale; int oW = u.iW*u.scale;
                          int oh_start = ih * u.scale;
                          int ow_start = iw * u.scale;
                          for(int y=0; y<u.scale; ++y) {
                                  for(int x=0; x<u.scale; ++x) {
                                          int go_idx = (b*u.C+c)*(oH*oW) + (oh_start+y)*oW + (ow_start+x);
                                          sum += go[go_idx];
                                  }
                          }
                          gi[idx] = sum;
                 }
         }
     )");

  // Collect Stats (Training): Sum and SqSum
  bnCollectPipeline = createComputePipeline("bn_collect", R"(
		#version 450
		layout(local_size_x=256) in;
		layout(binding=0) buffer I { float in_d[]; };
		layout(binding=1) buffer S { uint sum_u[]; };
		layout(binding=2) buffer SS { uint sq_sum_u[]; };
		layout(push_constant) uniform P { int N, C, Spatial; } u;

      #define ATOMIC_ADD_FLOAT(BUFFER, IDX, VAL) \
      do { \
           uint expected = BUFFER[IDX]; \
           uint current; \
           while(true) { \
                  uint next = floatBitsToUint(uintBitsToFloat(expected) + (VAL)); \
                  current = atomicCompSwap(BUFFER[IDX], expected, next); \
                  if (current == expected) break; \
                  expected = current; \
           } \
      } while(false)

		void main() {
			 int idx = int(gl_GlobalInvocationID.x);
			 if(idx < u.N) {
					int s = idx % u.Spatial;
					int tmp = idx / u.Spatial;
					int c = tmp % u.C;
					float val = in_d[idx];
					ATOMIC_ADD_FLOAT(sum_u, c, val);
					ATOMIC_ADD_FLOAT(sq_sum_u, c, val * val);
			 }
		}
	)");

  // Update Stats (Training): Calc Mean/Var, Update Running
  bnUpdatePipeline = createComputePipeline("bn_update", R"(
		#version 450
		layout(local_size_x=256) in;
		layout(binding=0) buffer RM { float rm[]; };
		layout(binding=1) buffer RV { float rv[]; };
		layout(binding=2) buffer SM { uint sm_u[]; }; // Aliased as uint for reading sums, writing mean
		layout(binding=3) buffer SV { uint sv_u[]; }; // Aliased as uint for reading sq_sum, writing var
		layout(push_constant) uniform P { int C; float m; int N_samples; } u;

		void main() {
			 int c = int(gl_GlobalInvocationID.x);
			 if(c < u.C) {
					 // Read sums (bits)
					 float sum = uintBitsToFloat(sm_u[c]);
					 float sq_sum = uintBitsToFloat(sv_u[c]);
					 float n = float(u.N_samples);

					 float mu = sum / n;
					 float var = (sq_sum / n) - (mu * mu);

					 // Update Running
					 rm[c] = (1.0 - u.m) * rm[c] + u.m * mu;
					 rv[c] = (1.0 - u.m) * rv[c] + u.m * var;

					 // Store saved stats (overwrite sums)
					 sm_u[c] = floatBitsToUint(mu);
					 sv_u[c] = floatBitsToUint(var);
			 }
		}
	)");

  // Normalize (Training/Inference)
  bnNormalizePipeline = createComputePipeline("bn_norm", R"(
		#version 450
		layout(local_size_x=256) in;
		layout(binding=0) buffer I { float in_d[]; };
		layout(binding=1) buffer G { float gamma[]; };
		layout(binding=2) buffer B_ { float beta[]; };
		layout(binding=3) buffer M { float mean[]; };
		layout(binding=4) buffer V { float var[]; };
		layout(binding=5) buffer O { float out_d[]; };
		layout(push_constant) uniform P { int N, C, Spatial; float eps; } u;

		void main() {
			 int idx = int(gl_GlobalInvocationID.x);
			 if(idx < u.N) {
					int s = idx % u.Spatial;
					int tmp = idx / u.Spatial;
					int c = tmp % u.C;

					float mu = mean[c];
					float v = var[c];
					float inv_std = inversesqrt(v + u.eps);

					out_d[idx] = gamma[c] * (in_d[idx] - mu) * inv_std + beta[c];
			 }
		}
	)");

  // Backward Reduce
  bnBackReducePipeline = createComputePipeline("bn_back_reduce", R"(
		#version 450
		layout(local_size_x=256) in;
		layout(binding=0) buffer GO { float dy[]; };
		layout(binding=1) buffer I { float x[]; };
		layout(binding=2) buffer M { float mean[]; };
		layout(binding=3) buffer V { float var[]; };
		layout(binding=4) buffer DS { uint dgamma_u[]; }; // Accumulator
		layout(binding=5) buffer DB { uint dbeta_u[]; };  // Accumulator
		layout(push_constant) uniform P { int N, C, Spatial; float eps; } u;

    #define ATOMIC_ADD_FLOAT(BUFFER, IDX, VAL) \
		do { \
				 uint expected = BUFFER[IDX]; \
				 uint current; \
				 while(true) { \
								uint next = floatBitsToUint(uintBitsToFloat(expected) + (VAL)); \
								current = atomicCompSwap(BUFFER[IDX], expected, next); \
								if (current == expected) break; \
								expected = current; \
				 } \
		} while(false)

		void main() {
			 int idx = int(gl_GlobalInvocationID.x);
			 if(idx < u.N) {
					int s = idx % u.Spatial;
					int tmp = idx / u.Spatial;
					int c = tmp % u.C;

					float mu = mean[c];
					float v = var[c];
					float inv_std = inversesqrt(v + u.eps);
					float x_hat = (x[idx] - mu) * inv_std;
					float g_val = dy[idx];

					ATOMIC_ADD_FLOAT(dbeta_u, c, g_val);
					ATOMIC_ADD_FLOAT(dgamma_u, c, g_val * x_hat);

			 }
		}
	)");

  // Backward Compute DX
  bnBackDxPipeline = createComputePipeline("bn_back_dx", R"(
		#version 450
		layout(local_size_x=256) in;
		layout(binding=0) buffer GO { float dy[]; };
		layout(binding=1) buffer I { float x[]; };
		layout(binding=2) buffer G { float gamma[]; };
		layout(binding=3) buffer M { float mean[]; };
		layout(binding=4) buffer V { float var[]; };
		layout(binding=5) buffer DS { float dgamma[]; };
		layout(binding=6) buffer DB { float dbeta[]; };
		layout(binding=7) buffer DX { float dx[]; };
		layout(push_constant) uniform P { int N, C, Spatial; float eps; } u;

		void main() {
			 int idx = int(gl_GlobalInvocationID.x);
			 if(idx < u.N) {
					int s = idx % u.Spatial;
					int tmp = idx / u.Spatial;
					int c = tmp % u.C;

					float mu = mean[c];
					float v = var[c];
					float inv_std = inversesqrt(v + u.eps);
					float x_hat = (x[idx] - mu) * inv_std;

					float m_val = float(u.N) / float(u.C); // Batch * Spatial size
					float term1 = m_val * dy[idx];
					float term2 = dbeta[c];
					float term3 = x_hat * dgamma[c];

					float factor = gamma[c] * inv_std / m_val;
					dx[idx] = factor * (term1 - term2 - term3);
			 }
		}
	)");
}

VulkanBackend::~VulkanBackend() {
  vkDeviceWaitIdle(device); // Full stop
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroyFence(device, inFlightFences[i], nullptr);
    vkDestroyDescriptorPool(device, descriptorPools[i], nullptr);
#ifdef ENABLE_PROFILING
    vkDestroyQueryPool(device, queryPools[i], nullptr);
#endif
  }
  for (auto &pair : allocation_memory) {
    vkDestroyBuffer(device, (VkBuffer)pair.first, nullptr);
    vkFreeMemory(device, pair.second, nullptr);
  }
  if (stagingBuffer) {
    vkUnmapMemory(device, stagingMemory);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
  }
  vkDestroyPipeline(device, conv2dPipeline, nullptr);
  vkDestroyPipeline(device, conv2dBackInputPipeline, nullptr);
  vkDestroyPipeline(device, conv2dBackWeightPipeline, nullptr);
  vkDestroyPipeline(device, conv2dBackBiasPipeline, nullptr);

  vkDestroyPipeline(device, maxPoolPipeline, nullptr);
  vkDestroyPipeline(device, maxPoolBackPipeline, nullptr);

  vkDestroyPipeline(device, upsamplePipeline, nullptr);
  vkDestroyPipeline(device, upsampleBackPipeline, nullptr);

  vkDestroyPipeline(device, bnCollectPipeline, nullptr);
  vkDestroyPipeline(device, bnUpdatePipeline, nullptr);
  vkDestroyPipeline(device, bnNormalizePipeline, nullptr);
  vkDestroyPipeline(device, bnBackReducePipeline, nullptr);
  vkDestroyPipeline(device, bnBackDxPipeline, nullptr);

  vkDestroyPipeline(device, addPipeline, nullptr);
  vkDestroyPipeline(device, mulPipeline, nullptr);
  vkDestroyPipeline(device, subPipeline, nullptr);
  vkDestroyPipeline(device, updatePipeline, nullptr);

  vkDestroyPipeline(device, reluPipeline, nullptr);
  vkDestroyPipeline(device, reluBackwardPipeline, nullptr);
  vkDestroyPipeline(device, sigmoidPipeline, nullptr);
  vkDestroyPipeline(device, sigmoidBackwardPipeline, nullptr);

  vkDestroyPipeline(device, softmaxPipeline, nullptr);
  vkDestroyPipeline(device, softmaxBackwardPipeline, nullptr);
  vkDestroyPipeline(device, mseLossPipeline, nullptr);
  vkDestroyPipeline(device, mseLossBackwardPipeline, nullptr);
  vkDestroyPipeline(device, crossEntropyPipeline, nullptr);
  vkDestroyPipeline(device, crossEntropyBackwardPipeline, nullptr);

  vkDestroyPipeline(device, matmulPipeline, nullptr);

  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
  vkDestroyCommandPool(device, commandPool, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}

void *VulkanBackend::allocate(size_t bytes) {
  size_t aligned_size = round_up_alloc_size(bytes);
  if (!free_pool[aligned_size].empty()) {
    uint64_t handle = free_pool[aligned_size].back();
    free_pool[aligned_size].pop_back();
    return (void *)handle;
  }
  VkBuffer buffer;
  VkDeviceMemory memory;
  createBuffer(aligned_size,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);
  uint64_t handle = (uint64_t)buffer;
  allocation_memory[handle] = memory;
  allocation_sizes[handle] = aligned_size;
  return (void *)handle;
}

void VulkanBackend::deallocate(void *ptr) {
  // Defer to current batch completion
  if (!ptr)
    return;
  uint64_t handle = (uint64_t)ptr;
  deferred_frees[currentFrame].push_back(handle);
}

// --- Batch Management ---

void flush_batch() {
  if (!isRecording)
    return;

  VkCommandBuffer cmd = commandBuffers[currentFrame];

#ifdef ENABLE_PROFILING
  // Write timestamp to Index 1 (End of Batch)
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      queryPools[currentFrame], 1);
#endif

  VK_CHECK(vkEndCommandBuffer(cmd));

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;
  VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo,
                         inFlightFences[currentFrame]));

  // Advance Frame
  currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  isRecording = false;
  currentBatchSize = 0;
}

void ensure_recording() {
  if (isRecording)
    return;

  // Wait for the NEXT frame slot to be free (Fence Wait)
  VK_CHECK(vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
                           UINT64_MAX));

  // Process deferred frees safely now that the GPU is done with this frame
  for (uint64_t handle : deferred_frees[currentFrame]) {
    if (allocation_sizes.count(handle)) {
      free_pool[allocation_sizes[handle]].push_back(handle);
    }
  }
  deferred_frees[currentFrame].clear();

  // Process Profiling for the FINISHED frame
#ifdef ENABLE_PROFILING
  // We used index 0 and 1.
  uint64_t stamps[2] = {0, 0};
  vkGetQueryPoolResults(device, queryPools[currentFrame], 0, 2,
                        sizeof(uint64_t) * 2, stamps, sizeof(uint64_t),
                        VK_QUERY_RESULT_64_BIT);

  if (stamps[1] > stamps[0]) {
    double total_ns = (stamps[1] - stamps[0]) * timestampPeriod;
    Profiler::get().log("Vulkan Batch", "vulkan", total_ns / 1000.0);
  }
#endif

  // Reset Descriptor Pool logic: wipe the slate clean for this frame
  VK_CHECK(vkResetDescriptorPool(device, *descriptorPools, 0));

  // Reset Descriptor Pool logic: wipe the slate clean for this frame
  VK_CHECK(vkResetDescriptorPool(device, descriptorPools[currentFrame], 0));

  VK_CHECK(vkResetFences(device, 1, &inFlightFences[currentFrame]));

  // Start Recording
  VkCommandBuffer cmd = commandBuffers[currentFrame];
  VK_CHECK(vkResetCommandBuffer(cmd, 0));

#ifdef ENABLE_PROFILING
  vkCmdResetQueryPool(cmd, queryPools[currentFrame], 0, 2);
  // Write timestamp to Index 0 (Start of Batch)
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      queryPools[currentFrame], 0);
#endif

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
  isRecording = true;
}

static void runImmediateCommand(std::function<void(VkCommandBuffer)> func) {
  if (isRecording)
    flush_batch(); // Flush pending work first
  VK_CHECK(vkQueueWaitIdle(computeQueue));

  VK_CHECK(vkResetCommandBuffer(immediateCmdBuffer, 0));
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(immediateCmdBuffer, &beginInfo));
  func(immediateCmdBuffer);
  VK_CHECK(vkEndCommandBuffer(immediateCmdBuffer));
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &immediateCmdBuffer;
  VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
  VK_CHECK(vkQueueWaitIdle(computeQueue));
}

void VulkanBackend::memset(void *ptr, int value, size_t bytes) {
  if (bytes == 0)
    return;

  ensure_recording();
  vkCmdFillBuffer(commandBuffers[currentFrame], (VkBuffer)(uint64_t)ptr, 0,
                  bytes, value);

  VkMemoryBarrier mb{};
  mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(
      commandBuffers[currentFrame], VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

  currentBatchSize++;
  if (currentBatchSize >= BATCH_SIZE_LIMIT)
    flush_batch();
}

void VulkanBackend::copy(const void *src, void *dst, size_t bytes,
                         Device src_dev, Device dst_dev) {
  if (bytes == 0)
    return;

  if (src_dev.type == DeviceType::VULKAN &&
      dst_dev.type == DeviceType::VULKAN) {
    ensure_recording();
    VkBufferCopy copyRegion{};
    copyRegion.size = bytes;
    vkCmdCopyBuffer(commandBuffers[currentFrame], (VkBuffer)(uint64_t)src,
                    (VkBuffer)(uint64_t)dst, 1, &copyRegion);

    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffers[currentFrame],
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);

    currentBatchSize++;
    if (currentBatchSize >= BATCH_SIZE_LIMIT)
      flush_batch();
    return;
  }

  auto get_staging = [&](size_t req_bytes, size_t &offset) {
    size_t aligned = (req_bytes + 255) & ~255;
    if (!stagingBuffer || stagingOffset + aligned > stagingSize) {
      if (isRecording)
        flush_batch();
      VK_CHECK(vkQueueWaitIdle(computeQueue));
      stagingOffset = 0; // Safe because queue is idle
      if (stagingSize < aligned) {
        if (stagingBuffer) {
          vkUnmapMemory(device, stagingMemory);
          vkDestroyBuffer(device, stagingBuffer, nullptr);
          vkFreeMemory(device, stagingMemory, nullptr);
        }
        stagingSize =
            aligned * 2 < 16 * 1024 * 1024 ? 16 * 1024 * 1024 : aligned * 2;
        createBuffer(stagingSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingMemory);
        VK_CHECK(vkMapMemory(device, stagingMemory, 0, stagingSize, 0,
                             &stagingMapped));
      }
    }
    offset = stagingOffset;
    stagingOffset += aligned;
  };

  if (src_dev.type == DeviceType::CPU && dst_dev.type == DeviceType::VULKAN) {
    size_t offset = 0;
    get_staging(bytes, offset);
    std::memcpy((char *)stagingMapped + offset, src, bytes);

    ensure_recording();
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = offset;
    copyRegion.dstOffset = 0;
    copyRegion.size = bytes;
    vkCmdCopyBuffer(commandBuffers[currentFrame], stagingBuffer,
                    (VkBuffer)(uint64_t)dst, 1, &copyRegion);

    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffers[currentFrame],
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);

    currentBatchSize++;
    if (currentBatchSize >= BATCH_SIZE_LIMIT)
      flush_batch();
  } else if (src_dev.type == DeviceType::VULKAN &&
             dst_dev.type == DeviceType::CPU) {
    size_t offset = 0;
    get_staging(bytes, offset);

    ensure_recording();
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(
        commandBuffers[currentFrame], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = offset;
    copyRegion.size = bytes;
    vkCmdCopyBuffer(commandBuffers[currentFrame], (VkBuffer)(uint64_t)src,
                    stagingBuffer, 1, &copyRegion);
    currentBatchSize++;
    flush_batch();
    VK_CHECK(vkQueueWaitIdle(computeQueue));
    std::memcpy(dst, (char *)stagingMapped + offset, bytes);
    stagingOffset =
        0; // Free entire staging buffer since we just forced a full wait
  }
}

void VulkanBackend::synchronize() {
  if (isRecording)
    flush_batch();
  VK_CHECK(vkQueueWaitIdle(computeQueue));
}
void VulkanBackend::all_reduce(Storage &buffer, size_t num_elements) {}

void VulkanBackend::dispatch_kernel(VkPipeline pipeline,
                                    const std::vector<void *> &buffers,
                                    void *pc, size_t pcSize, int x, int y,
                                    int z) {
  ensure_recording();
  VkCommandBuffer cmd = commandBuffers[currentFrame];

  VkDescriptorSetAllocateInfo allocSetInfo{};
  allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocSetInfo.descriptorPool = descriptorPools[currentFrame];
  allocSetInfo.descriptorSetCount = 1;
  allocSetInfo.pSetLayouts = &descriptorSetLayout;
  VkDescriptorSet ds;
  VK_CHECK(vkAllocateDescriptorSets(device, &allocSetInfo, &ds));

  // Always update 8 bindings to match layout, padding with first buffer if
  // needed
  std::vector<VkDescriptorBufferInfo> bInfos(8);
  std::vector<VkWriteDescriptorSet> writes(8);

  for (size_t i = 0; i < 8; ++i) {
    void *ptr =
        (i < buffers.size() && buffers[i] != nullptr) ? buffers[i] : buffers[0];
    bInfos[i].buffer = (VkBuffer)(uint64_t)ptr;
    bInfos[i].offset = 0;
    bInfos[i].range = VK_WHOLE_SIZE;

    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].descriptorCount = 1;
    writes[i].pBufferInfo = &bInfos[i];
  }
  vkUpdateDescriptorSets(device, 8, writes.data(), 0, nullptr);

  VkMemoryBarrier memoryBarrier{};
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask =
      VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  memoryBarrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memoryBarrier, 0, nullptr, 0, nullptr);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                          0, 1, &ds, 0, nullptr);
  vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     pcSize, pc);
  vkCmdDispatch(cmd, x, y, z);

  currentBatchSize++;
  if (currentBatchSize >= BATCH_SIZE_LIMIT) {
    flush_batch();
  }
}

// --- Kernel Wrappers ---
void VulkanBackend::add(const Storage &a, const Storage &b, Storage &out,
                        size_t num_elements) {
  uint32_t N = num_elements;
  dispatch_kernel(addPipeline, {a.data(), b.data(), out.data()}, &N, sizeof(N),
                  (N + 255) / 256, 1, 1);
}
void VulkanBackend::mul(const Storage &a, const Storage &b, Storage &out,
                        size_t num_elements) {
  uint32_t N = num_elements;
  dispatch_kernel(mulPipeline, {a.data(), b.data(), out.data()}, &N, sizeof(N),
                  (N + 255) / 256, 1, 1);
}
void VulkanBackend::sub(const Storage &a, const Storage &b, Storage &out,
                        size_t num_elements) {
  uint32_t N = num_elements;
  dispatch_kernel(subPipeline, {a.data(), b.data(), out.data()}, &N, sizeof(N),
                  (N + 255) / 256, 1, 1);
}
void VulkanBackend::relu(const Storage &in, Storage &out, size_t num_elements) {
  uint32_t N = num_elements;
  dispatch_kernel(reluPipeline, {in.data(), out.data()}, &N, sizeof(N),
                  (N + 255) / 256, 1, 1);
}
void VulkanBackend::relu_backward(const Storage &grad_out, const Storage &input,
                                  Storage &grad_in, size_t num_elements) {
  uint32_t N = num_elements;
  dispatch_kernel(reluBackwardPipeline,
                  {grad_out.data(), input.data(), grad_in.data()}, &N,
                  sizeof(N), (N + 255) / 256, 1, 1);
}
void VulkanBackend::sigmoid(const Storage &in, Storage &out,
                            size_t num_elements) {
  uint32_t N = num_elements;
  dispatch_kernel(sigmoidPipeline, {in.data(), out.data()}, &N, sizeof(N),
                  (N + 255) / 256, 1, 1);
}
void VulkanBackend::sigmoid_backward(const Storage &grad_out,
                                     const Storage &out, Storage &grad_in,
                                     size_t num_elements) {
  uint32_t N = num_elements;
  dispatch_kernel(sigmoidBackwardPipeline,
                  {grad_out.data(), out.data(), grad_in.data()}, &N, sizeof(N),
                  (N + 255) / 256, 1, 1);
}
void VulkanBackend::softmax(const Storage &in, Storage &out, int batch_size,
                            int num_classes) {
  struct {
    int B, C;
  } pc = {batch_size, num_classes};
  dispatch_kernel(softmaxPipeline, {in.data(), out.data()}, &pc, sizeof(pc),
                  (batch_size + 255) / 256, 1, 1);
}
void VulkanBackend::softmax_backward(const Storage &grad_out,
                                     const Storage &out, Storage &grad_in,
                                     int batch_size, int num_classes) {
  struct {
    int B, C;
  } pc = {batch_size, num_classes};
  dispatch_kernel(softmaxBackwardPipeline,
                  {grad_out.data(), out.data(), grad_in.data()}, &pc,
                  sizeof(pc), (batch_size + 255) / 256, 1, 1);
}

void VulkanBackend::mse_loss(const Storage &pred, const Storage &target,
                             Storage &out_loss, size_t num_elements) {
  memset(out_loss.data(), 0, out_loss.size_bytes());
  uint32_t N = num_elements;
  dispatch_kernel(mseLossPipeline,
                  {pred.data(), target.data(), out_loss.data()}, &N, sizeof(N),
                  (N + 255) / 256, 1, 1);
}

void VulkanBackend::mse_loss_backward(const Storage &grad_out,
                                      const Storage &pred,
                                      const Storage &target, Storage &grad_in,
                                      size_t num_elements) {
  uint32_t N = num_elements;
  dispatch_kernel(mseLossBackwardPipeline,
                  {grad_out.data(), pred.data(), target.data(), grad_in.data()},
                  &N, sizeof(N), (N + 255) / 256, 1, 1);
}

void VulkanBackend::cross_entropy(const Storage &logits, const Storage &targets,
                                  Storage &out_loss, int batch_size,
                                  int num_classes) {
  memset(out_loss.data(), 0, out_loss.size_bytes());
  struct {
    int B, C;
  } pc = {batch_size, num_classes};
  dispatch_kernel(crossEntropyPipeline,
                  {logits.data(), targets.data(), out_loss.data()}, &pc,
                  sizeof(pc), (batch_size + 255) / 256, 1, 1);
}

void VulkanBackend::cross_entropy_backward(const Storage &grad_out,
                                           const Storage &logits,
                                           const Storage &targets,
                                           Storage &grad_in, int batch_size,
                                           int num_classes) {
  struct {
    int B, C;
  } pc = {batch_size, num_classes};
  dispatch_kernel(
      crossEntropyBackwardPipeline,
      {grad_out.data(), logits.data(), targets.data(), grad_in.data()}, &pc,
      sizeof(pc), (batch_size + 255) / 256, 1, 1);
}

void VulkanBackend::update(Storage &weight, const Storage &grad, float lr,
                           size_t num_elements) {
  struct {
    uint32_t N;
    float lr;
  } pc = {(uint32_t)num_elements, lr};
  dispatch_kernel(updatePipeline, {weight.data(), grad.data()}, &pc, sizeof(pc),
                  (num_elements + 255) / 256, 1, 1);
}
void VulkanBackend::matmul(const Storage &a, const Storage &b, Storage &out,
                           int M, int K, int N, bool transA, bool transB) {
  struct {
    int m, k, n, ta, tb;
  } pc = {M, K, N, transA, transB};
  // Dispatch x maps to N (cols), y maps to M (rows)
  dispatch_kernel(matmulPipeline, {a.data(), b.data(), out.data()}, &pc,
                  sizeof(pc), (N + 15) / 16, (M + 15) / 16, 1);
}

// --- Spatial Stubs ---
void VulkanBackend::conv2d(const Storage &in, const Storage &weight,
                           const Storage *bias, Storage &out, int B, int iC,
                           int iH, int iW, int oC, int kH, int kW, int s,
                           int p) {
  int oH = (iH + 2 * p - kH) / s + 1;
  int oW = (iW + 2 * p - kW) / s + 1;
  struct {
    int B, iC, iH, iW, oC, kH, kW, s, p, oH, oW, has_bias;
  } pc = {B, iC, iH, iW, oC, kH, kW, s, p, oH, oW, bias ? 1 : 0};
  dispatch_kernel(
      conv2dPipeline,
      {in.data(), weight.data(), bias ? bias->data() : nullptr, out.data()},
      &pc, sizeof(pc), (B * oC * oH * oW + 255) / 256, 1, 1);
}

void VulkanBackend::conv2d_backward(const Storage &grad_out, const Storage &in,
                                    const Storage &weight, Storage &grad_in,
                                    Storage &grad_w, Storage *grad_b, int B,
                                    int iC, int iH, int iW, int oC, int kH,
                                    int kW, int s, int p) {
  int oH = (iH + 2 * p - kH) / s + 1;
  int oW = (iW + 2 * p - kW) / s + 1;
  struct {
    int B, iC, iH, iW, oC, kH, kW, s, p, oH, oW;
  } pc = {B, iC, iH, iW, oC, kH, kW, s, p, oH, oW};

  // 1. Grad Input
  dispatch_kernel(conv2dBackInputPipeline,
                  {grad_out.data(), weight.data(), grad_in.data()}, &pc,
                  sizeof(pc), (B * iC * iH * iW + 255) / 256, 1, 1);

  // 2. Grad Weight
  dispatch_kernel(conv2dBackWeightPipeline,
                  {grad_out.data(), in.data(), grad_w.data()}, &pc, sizeof(pc),
                  (oC * iC * kH * kW + 255) / 256, 1, 1);

  // 3. Grad Bias
  if (grad_b) {
    struct {
      int B, oC, oH, oW;
    } pcb = {B, oC, oH, oW};
    dispatch_kernel(conv2dBackBiasPipeline, {grad_out.data(), grad_b->data()},
                    &pcb, sizeof(pcb), (oC + 255) / 256, 1, 1);
  }
}

void VulkanBackend::max_pool2d(const Storage &in, Storage &out, int B, int C,
                               int iH, int iW, int k, int s, int p) {
  int oH = (iH + 2 * p - k) / s + 1;
  int oW = (iW + 2 * p - k) / s + 1;
  struct {
    int B, C, iH, iW, k, s, p, oH, oW;
  } pc = {B, C, iH, iW, k, s, p, oH, oW};
  dispatch_kernel(maxPoolPipeline, {in.data(), out.data()}, &pc, sizeof(pc),
                  (B * C * oH * oW + 255) / 256, 1, 1);
}

void VulkanBackend::max_pool2d_backward(const Storage &grad_out,
                                        const Storage &in, Storage &grad_in,
                                        int B, int C, int iH, int iW, int k,
                                        int s, int p) {
  int oH = (iH + 2 * p - k) / s + 1;
  int oW = (iW + 2 * p - k) / s + 1;
  struct {
    int B, C, iH, iW, k, s, p, oH, oW;
  } pc = {B, C, iH, iW, k, s, p, oH, oW};
  dispatch_kernel(maxPoolBackPipeline,
                  {grad_out.data(), in.data(), grad_in.data()}, &pc, sizeof(pc),
                  (B * C * oH * oW + 255) / 256, 1, 1);
}

void VulkanBackend::upsample2d(const Storage &in, Storage &out, int B, int C,
                               int iH, int iW, int scale) {
  struct {
    int B, C, iH, iW, scale;
  } pc = {B, C, iH, iW, scale};
  int oH = iH * scale, oW = iW * scale;
  dispatch_kernel(upsamplePipeline, {in.data(), out.data()}, &pc, sizeof(pc),
                  (B * C * oH * oW + 255) / 256, 1, 1);
}

void VulkanBackend::upsample2d_backward(const Storage &grad_out,
                                        Storage &grad_in, int B, int C, int iH,
                                        int iW, int scale) {
  struct {
    int B, C, iH, iW, scale;
  } pc = {B, C, iH, iW, scale};
  dispatch_kernel(upsampleBackPipeline, {grad_out.data(), grad_in.data()}, &pc,
                  sizeof(pc), (B * C * iH * iW + 255) / 256, 1, 1);
}

void VulkanBackend::batch_norm(const Storage &in, const Storage &scale,
                               const Storage &bias, Storage &running_mean,
                               Storage &running_var, Storage &save_mean,
                               Storage &save_var, Storage &out, int B, int C,
                               int H, int W, float momentum, float eps,
                               bool training) {
  int Spatial = H * W;
  int Total = B * C * Spatial;

  if (training) {
    // 1. Zero accumulator buffers (save_mean/save_var act as temps)
    memset(save_mean.data(), 0, save_mean.size_bytes());
    memset(save_var.data(), 0, save_var.size_bytes());

    // 2. Collect stats (Sum, SqSum)
    struct {
      int N, C, Spatial;
    } pc = {Total, C, Spatial};
    dispatch_kernel(bnCollectPipeline,
                    {in.data(), save_mean.data(), save_var.data()}, &pc,
                    sizeof(pc), (Total + 255) / 256, 1, 1);

    // 3. Finalize and Update
    struct {
      int C;
      float m;
      int N_samples;
    } pc_up = {C, momentum, B * Spatial};
    dispatch_kernel(bnUpdatePipeline,
                    {running_mean.data(), running_var.data(), save_mean.data(),
                     save_var.data()},
                    &pc_up, sizeof(pc_up), (C + 255) / 256, 1, 1);
  }

  // 4. Normalize
  void *m_ptr = training ? save_mean.data() : running_mean.data();
  void *v_ptr = training ? save_var.data() : running_var.data();

  struct {
    int N, C, Spatial;
    float eps;
  } pc_norm = {Total, C, Spatial, eps};
  dispatch_kernel(
      bnNormalizePipeline,
      {in.data(), scale.data(), bias.data(), m_ptr, v_ptr, out.data()},
      &pc_norm, sizeof(pc_norm), (Total + 255) / 256, 1, 1);
}

void VulkanBackend::batch_norm_backward(const Storage &grad_out,
                                        const Storage &in, const Storage &scale,
                                        const Storage &save_mean,
                                        const Storage &save_var,
                                        Storage &grad_in, Storage &grad_scale,
                                        Storage &grad_bias, int B, int C, int H,
                                        int W, float eps) {
  int Spatial = H * W;
  int Total = B * C * Spatial;

  // 1. Zero Accumulators (grad_scale / grad_bias)
  memset(grad_scale.data(), 0, grad_scale.size_bytes());
  memset(grad_bias.data(), 0, grad_bias.size_bytes());

  // 2. Reduce (Compute dGamma, dBeta)
  struct {
    int N, C, Spatial;
    float eps;
  } pc = {Total, C, Spatial, eps};
  dispatch_kernel(bnBackReducePipeline,
                  {grad_out.data(), in.data(), save_mean.data(),
                   save_var.data(), grad_scale.data(), grad_bias.data()},
                  &pc, sizeof(pc), (Total + 255) / 256, 1, 1);

  // 3. Compute DX
  dispatch_kernel(bnBackDxPipeline,
                  {grad_out.data(), in.data(), scale.data(), save_mean.data(),
                   save_var.data(), grad_scale.data(), grad_bias.data(),
                   grad_in.data()},
                  &pc, sizeof(pc), (Total + 255) / 256, 1, 1);
}

} // namespace munet
