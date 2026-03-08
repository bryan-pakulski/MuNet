#include "vulkan_backend.hpp"
#include "storage.hpp"
#include "util.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
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
static const int BATCH_SIZE_LIMIT = 2048; // Flush after this many ops
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
static std::vector<VkDescriptorSet> frameDescriptorSets[MAX_FRAMES_IN_FLIGHT];
static uint32_t descriptorSetCursor[MAX_FRAMES_IN_FLIGHT] = {0};
static VkCommandBuffer commandBuffers[MAX_FRAMES_IN_FLIGHT];
static VkFence inFlightFences[MAX_FRAMES_IN_FLIGHT];
static VkQueryPool queryPools[MAX_FRAMES_IN_FLIGHT];

static float timestampPeriod = 1.0f;

// Batching State
static int currentFrame = 0;
static int currentBatchSize = 0;
static bool isRecording = false;

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
static VkPipeline addPipeline, mulPipeline, subPipeline, matmulPipeline;
static VkPipeline addBCPipeline, mulBCPipeline, subBCPipeline,
    sumToShapePipeline;
static VkPipeline reluPipeline, reluBackwardPipeline, updatePipeline;
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



static void allocate_frame_descriptor_sets(int frame) {
  std::vector<VkDescriptorSetLayout> layouts(MAX_DESCRIPTORS_PER_FRAME,
                                             descriptorSetLayout);
  frameDescriptorSets[frame].resize(MAX_DESCRIPTORS_PER_FRAME);

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPools[frame];
  allocInfo.descriptorSetCount = MAX_DESCRIPTORS_PER_FRAME;
  allocInfo.pSetLayouts = layouts.data();

  VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo,
                                    frameDescriptorSets[frame].data()));
  descriptorSetCursor[frame] = 0;
}
VulkanBackend::VulkanBackend(int device_index) : device_index_(device_index) {
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;
  VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    throw std::runtime_error("No Vulkan physical devices found.");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  if (device_index_ < 0 || device_index_ >= static_cast<int>(deviceCount)) {
    throw std::runtime_error("Requested Vulkan device index out of range: " +
                             std::to_string(device_index_) +
                             " (available: " + std::to_string(deviceCount) + ")");
  }

  physicalDevice = devices[static_cast<size_t>(device_index_)];

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physicalDevice, &props);
  timestampPeriod = props.limits.timestampPeriod;
  MUNET_INFO << "Vulkan backend using device index " << device_index_ << " ("
            << props.deviceName << ")" << std::endl;

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

  VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
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
    // Create query pools for profiling
    VK_CHECK(
        vkCreateQueryPool(device, &queryPoolInfo, nullptr, &queryPools[i]));
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
    allocate_frame_descriptor_sets(i);
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

  // TODO: Make the local size X dynamic at compile time depending on the target
  // arch NVIDIA	128 or 256
  // AMD	256
  // Apple	64
  // Intel	128
  addPipeline = createComputePipeline("add", R"(
				#version 450
				layout(local_size_x = 256) in;
				layout(binding = 0) buffer A { float a[]; };
				layout(binding = 1) buffer B { float b[]; };
				layout(binding = 2) buffer C { float c[]; };
				layout(push_constant) uniform Push { uint N; } p;
				void main() {
						uint i = gl_GlobalInvocationID.x;
						if (i < p.N) {
								c[i] = a[i] + b[i];
						}
				}
  )");

  mulPipeline = createComputePipeline("mul", R"(
				#version 450
				layout(local_size_x = 256) in;
				layout(binding = 0) buffer A { float a[]; };
				layout(binding = 1) buffer B { float b[]; };
				layout(binding = 2) buffer C { float c[]; };
				layout(push_constant) uniform Push { uint N; } p;
				void main() {
						uint i = gl_GlobalInvocationID.x;
						if (i < p.N) c[i] = a[i] * b[i];
				}
  )");

  subPipeline = createComputePipeline("sub", R"(
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

  addBCPipeline = createComputePipeline("add_bc_fast", R"(
        #version 450

				layout(local_size_x = 256) in;
				layout(binding = 0) buffer A { float a[]; };
				layout(binding = 1) buffer B { float b[]; };
				layout(binding = 2) buffer C { float c[]; };
				layout(push_constant) uniform P { int info[26]; } u;
				void main() {
						uint total = uint(u.info[25]);
						uint idx = gl_GlobalInvocationID.x;
						if (idx >= total) return;
						int ndim = u.info[0];
						uint off_a = 0;
						uint off_b = 0;
						for (int d = 0; d < ndim; ++d) {
								uint stride = uint(u.info[7 + d]);
								uint coord  = (idx / stride) % uint(u.info[1 + d]);
								off_a += coord * uint(u.info[13 + d]);
								off_b += coord * uint(u.info[19 + d]);
						}
						c[idx] = a[off_a] + b[off_b];
				}
				)");

  mulBCPipeline = createComputePipeline("mul_bc_fast", R"(
        #version 450

				layout(local_size_x = 256) in;
				layout(binding = 0) buffer A { float a[]; };
				layout(binding = 1) buffer B { float b[]; };
				layout(binding = 2) buffer C { float c[]; };
				layout(push_constant) uniform P { int info[26]; } u;
				void main() {
						uint total = uint(u.info[25]);
						uint idx = gl_GlobalInvocationID.x;
						if (idx >= total) return;
						int ndim = u.info[0];
						uint off_a = 0;
						uint off_b = 0;
						for (int d = 0; d < ndim; ++d) {
								uint stride = uint(u.info[7 + d]);
								uint coord  = (idx / stride) % uint(u.info[1 + d]);
								off_a += coord * uint(u.info[13 + d]);
								off_b += coord * uint(u.info[19 + d]);
						}
						c[idx] = a[off_a] * b[off_b];
				}
				)");

  subBCPipeline = createComputePipeline("sub_bc_fast", R"(
        #version 450

				layout(local_size_x = 256) in;
				layout(binding = 0) buffer A { float a[]; };
				layout(binding = 1) buffer B { float b[]; };
				layout(binding = 2) buffer C { float c[]; };
				layout(push_constant) uniform P { int info[26]; } u;
				void main() {
						uint total = uint(u.info[25]);
						uint idx = gl_GlobalInvocationID.x;
						if (idx >= total) return;
						int ndim = u.info[0];
						uint off_a = 0;
						uint off_b = 0;
						for (int d = 0; d < ndim; ++d) {
								uint stride = uint(u.info[7 + d]);
								uint coord  = (idx / stride) % uint(u.info[1 + d]);
								off_a += coord * uint(u.info[13 + d]);
								off_b += coord * uint(u.info[19 + d]);
						}
						c[idx] = a[off_a] - b[off_b];
				}
				)");

  sumToShapePipeline = createComputePipeline("sum_to_shape", R"(
        #version 450

				layout(local_size_x = 256) in;
				layout(binding = 0) readonly buffer I { float in_d[]; };
				layout(binding = 1) buffer O { uint out_u[]; };
				layout(push_constant) uniform P { int info[21]; } u;
				void main() {
						uint N = uint(u.info[20]);
						uint i = gl_GlobalInvocationID.x;
						if (i >= N) return;
						int ndim = u.info[0];
						int out_ndim = u.info[1];
						uint out_off = 0;
						uint curr = i;
						for (int d = ndim - 1; d >= 0; --d) {
								uint shape_d = uint(u.info[2 + d]);
								uint coord = curr % shape_d;
								curr /= shape_d;
								int out_d_idx = d - (ndim - out_ndim);
								if (out_d_idx >= 0) {
										uint out_shape_d = uint(u.info[8 + out_d_idx]);
										if (out_shape_d != 1) {
												out_off += coord * uint(u.info[14 + out_d_idx]);
										}
								}
						}
						uint expected = out_u[out_off];
						while (true) {
								float current_f = uintBitsToFloat(expected);
								float next_f = current_f + in_d[i];
								uint next = floatBitsToUint(next_f);
								uint current = atomicCompSwap(out_u[out_off], expected, next);
								if (current == expected) break;
								expected = current;
						}
				}
			  )");

  updatePipeline = createComputePipeline("update",
                                         R"(
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer W { float w[]; };
        layout(binding = 1) buffer G { float g[]; };

        layout(push_constant) uniform Push { uint N; float lr; } p;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i < p.N) {
                // Fetch to register to ensure safety if memory aliases
                float grad = g[i];
                w[i] -= (p.lr * grad);
            }
        }
    )");

  broadcastRowPipeline = createComputePipeline("broadcast_row",
                                               R"(
         #version 450
         layout(local_size_x = 256) in;
         layout(binding = 0) buffer Src { float src[]; };
         layout(binding = 1) buffer Dst { float dst[]; };
         layout(push_constant) uniform P { int rows; int cols; } u;
         void main() {
             int idx = int(gl_GlobalInvocationID.x);
             if (idx >= u.rows * u.cols) return;
             dst[idx] = src[idx % u.cols];
         }
     )");

  adamStepPipeline = createComputePipeline("adam_step",
                                           R"(
         #version 450
         layout(local_size_x = 256) in;

         layout(binding = 0) buffer P { float p[]; };
         layout(binding = 1) buffer G { float g[]; };
         layout(binding = 2) buffer M { float m[]; };
         layout(binding = 3) buffer V { float v[]; };

         layout(push_constant) uniform Push {
             uint N;
             float lr;
             float beta1;
             float beta2;
             float eps;
             int step;
         } u;

         void main() {
             uint i = gl_GlobalInvocationID.x;
             if (i < u.N) {
                 float grad = g[i];
                 m[i] = u.beta1 * m[i] + (1.0 - u.beta1) * grad;
                 v[i] = u.beta2 * v[i] + (1.0 - u.beta2) * (grad * grad);

                 float m_hat = m[i] / (1.0 - pow(u.beta1, float(u.step)));
                 float v_hat = v[i] / (1.0 - pow(u.beta2, float(u.step)));

                 p[i] -= u.lr * m_hat / (sqrt(v_hat) + u.eps);
             }
         }
     )");

  reluPipeline = createComputePipeline("relu",
                                       R"(
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer A { float a[]; };
        layout(binding = 1) writeonly buffer C { float c[]; };

        layout(push_constant) uniform Push { uint N; } p;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i < p.N)
                c[i] = max(0.0, a[i]);
        }
    )");

  reluBackwardPipeline = createComputePipeline("relu_back",
                                               R"(
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer GO { float go[]; };
        layout(binding = 1) readonly buffer I { float i[]; };
        layout(binding = 2) writeonly buffer GI { float gi[]; };

        layout(push_constant) uniform Push { uint N; } p;

        void main() {
            uint id = gl_GlobalInvocationID.x;
            if (id < p.N)
                gi[id] = (i[id] > 0.0) ? go[id] : 0.0;
        }
    )");

  sigmoidPipeline = createComputePipeline("sigmoid",
                                          R"(
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer A { float a[]; };
        layout(binding = 1) writeonly buffer C { float c[]; };

        layout(push_constant) uniform Push { uint N; } p;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i < p.N)
                c[i] = 1.0 / (1.0 + exp(-a[i]));
        }
    )");

  sigmoidBackwardPipeline = createComputePipeline("sigmoid_back",
                                                  R"(
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer GO { float go[]; };
        layout(binding = 1) readonly buffer O { float out_d[]; };
        layout(binding = 2) writeonly buffer GI { float gi[]; };

        layout(push_constant) uniform Push { uint N; } p;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i < p.N) {
                float s = out_d[i];
                gi[i] = go[i] * s * (1.0 - s);
            }
        }
    )");

  softmaxPipeline = createComputePipeline("softmax",
                                          R"(
        #version 450
				layout(local_size_x = 256) in;

				layout(binding = 0) readonly buffer I { float in_d[]; };
				layout(binding = 1) writeonly buffer O { float out_d[]; };

				layout(push_constant) uniform P {
						int B;
						int C;
				} u;

				void main() {

						int b = int(gl_GlobalInvocationID.x);
						if (b >= u.B) return;

						int base = b * u.C;

						float max_val = -3.402823e38;

						for (int i = 0; i < u.C; ++i) {
								float v = in_d[base + i];
								max_val = max(max_val, v);
						}

						float sum_exp = 0.0;

						for (int i = 0; i < u.C; ++i) {
								float e = exp(in_d[base + i] - max_val);
								out_d[base + i] = e;
								sum_exp += e;
						}

						float inv_sum = 1.0 / sum_exp;

						for (int i = 0; i < u.C; ++i) {
								out_d[base + i] *= inv_sum;
						}
				}
    )");
  softmaxBackwardPipeline = createComputePipeline("softmax_back",
                                                  R"(
        #version 450
				layout(local_size_x = 256) in;

				layout(binding = 0) readonly buffer GO { float go[]; };
				layout(binding = 1) readonly buffer O  { float out_d[]; };
				layout(binding = 2) writeonly buffer GI { float gi[]; };

				layout(push_constant) uniform P {
						int B;
						int C;
				} u;

				void main() {

						int b = int(gl_GlobalInvocationID.x);
						if (b >= u.B) return;

						int base = b * u.C;

						float dot = 0.0;

						for (int i = 0; i < u.C; ++i) {
								float s = out_d[base + i];
								float g = go[base + i];
								dot += s * g;
						}

						for (int i = 0; i < u.C; ++i) {
								float s = out_d[base + i];
								float g = go[base + i];
								gi[base + i] = s * (g - dot);
						}
				}
      )");

  mseLossPipeline = createComputePipeline("mse_loss",
                                          R"(
				#version 450

				layout(local_size_x = 256) in;

				layout(push_constant) uniform PushConsts {
						uint N;
				} u;

				layout(binding = 0) readonly buffer A { float a[]; };
				layout(binding = 1) readonly buffer B { float b[]; };
				layout(binding = 2) buffer O { uint out_u[]; };

				shared float sdata[256];

				void main()
				{
						uint gid = gl_GlobalInvocationID.x;
						uint lid = gl_LocalInvocationID.x;

						float sum = 0.0;

						// grid-stride loop
						for (uint i = gid; i < u.N; i += gl_WorkGroupSize.x * gl_NumWorkGroups.x)
						{
								float d = a[i] - b[i];
								sum += d * d;
						}

						sdata[lid] = sum;
						barrier();

						// reduction in shared memory
						for (uint s = gl_WorkGroupSize.x >> 1; s > 0; s >>= 1)
						{
								if (lid < s)
										sdata[lid] += sdata[lid + s];
								barrier();
						}

						// one atomic add per workgroup
						if (lid == 0)
						{
								uint old = out_u[0];
								uint assumed;

								do {
										assumed = old;

										float f = uintBitsToFloat(assumed);
										f += sdata[0] / float(u.N);

										old = atomicCompSwap(
												out_u[0],
												assumed,
												floatBitsToUint(f)
										);

								} while (assumed != old);
						}
				}
     )");
  mseLossBackwardPipeline = createComputePipeline("mse_loss_back",
                                                  R"(
        #version 450
				layout(local_size_x = 256) in;

				layout(binding = 0) readonly buffer GO { float go[]; };
				layout(binding = 1) readonly buffer P { float p[]; };
				layout(binding = 2) readonly buffer T { float t[]; };
				layout(binding = 3) writeonly buffer GI { float gi[]; };

				layout(push_constant) uniform Push {
						uint N;
				} u;

				void main() {

						uint i = gl_GlobalInvocationID.x;
						if (i >= u.N) return;

						float scale = go[0] * 2.0 / float(u.N);
						gi[i] = scale * (p[i] - t[i]);
				}
     )");

  crossEntropyPipeline = createComputePipeline("ce_loss",
                                               R"(
        #version 450
				layout(local_size_x = 256) in;

				layout(binding = 0) readonly buffer L { float logits[]; };
				layout(binding = 1) readonly buffer T { float targets[]; };
				layout(binding = 2) buffer O { uint out_u[]; };

				layout(push_constant) uniform Push {
						int B;
						int C;
						int Spatial;
				} u;

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
						int total_pixels = u.B * u.Spatial;
						if (idx >= total_pixels) return;

						int b = idx / u.Spatial;
						int s = idx % u.Spatial;

						int stride = u.Spatial;
						int base = b * (u.C * u.Spatial) + s;

						float max_val = -3.402823e38;

						for (int c = 0; c < u.C; ++c)
								max_val = max(max_val, logits[base + c * stride]);

						float sum_exp = 0.0;

						for (int c = 0; c < u.C; ++c) {
								float e = exp(logits[base + c * stride] - max_val);
								sum_exp += e;
						}

						float inv_sum = 1.0 / sum_exp;

						float loss = 0.0;

						for (int c = 0; c < u.C; ++c) {

								float prob = exp(logits[base + c * stride] - max_val) * inv_sum;
								float target = targets[base + c * stride];

								if (target > 0.0)
										loss -= target * log(prob + 1e-9);
						}

						ATOMIC_ADD_FLOAT(out_u, 0, loss / u.B);
				}
      )");

  crossEntropyBackwardPipeline = createComputePipeline("ce_loss_back",
                                                       R"(
        #version 450
				layout(local_size_x = 256) in;

				layout(binding = 0) readonly buffer GO { float go[]; };
				layout(binding = 1) readonly buffer L { float logits[]; };
				layout(binding = 2) readonly buffer T { float targets[]; };
				layout(binding = 3) writeonly buffer GI { float gi[]; };

				layout(push_constant) uniform Push {
						int B;
						int C;
						int Spatial;
				} u;

				void main() {

						int idx = int(gl_GlobalInvocationID.x);
						int total_pixels = u.B * u.Spatial;
						if (idx >= total_pixels) return;

						int b = idx / u.Spatial;
						int s = idx % u.Spatial;

						int stride = u.Spatial;
						int base = b * (u.C * u.Spatial) + s;

						float max_val = -3.402823e38;

						for (int c = 0; c < u.C; ++c)
								max_val = max(max_val, logits[base + c * stride]);

						float sum_exp = 0.0;

						for (int c = 0; c < u.C; ++c)
								sum_exp += exp(logits[base + c * stride] - max_val);

						float inv_sum = 1.0 / sum_exp;
						float scale = go[0] / float(u.B);

						for (int c = 0; c < u.C; ++c) {

								float prob = exp(logits[base + c * stride] - max_val) * inv_sum;
								gi[base + c * stride] = scale * (prob - targets[base + c * stride]);
						}
				}
      )");

  matmulPipeline = createComputePipeline("matmul",
                                         R"(
        #version 450
				layout(local_size_x = 16, local_size_y = 16) in;

				layout(binding = 0) readonly buffer A { float a[]; };
				layout(binding = 1) readonly buffer B { float b[]; };
				layout(binding = 2) writeonly buffer C { float c[]; };

				layout(push_constant) uniform Push {
						int M;
						int K;
						int N;
						int tA;
						int tB;
				} p;

				void main() {

						int n = int(gl_GlobalInvocationID.x);
						int m = int(gl_GlobalInvocationID.y);

						if (m >= p.M || n >= p.N) return;

						float sum = 0.0;

						if (p.tA == 0 && p.tB == 0) {

								int a_row = m * p.K;
								int b_col = n;

								for (int k = 0; k < p.K; ++k)
										sum += a[a_row + k] * b[k * p.N + b_col];

						} else if (p.tA == 1 && p.tB == 0) {

								int b_col = n;

								for (int k = 0; k < p.K; ++k)
										sum += a[k * p.M + m] * b[k * p.N + b_col];

						} else if (p.tA == 0 && p.tB == 1) {

								int a_row = m * p.K;

								for (int k = 0; k < p.K; ++k)
										sum += a[a_row + k] * b[n * p.K + k];

						} else {

								for (int k = 0; k < p.K; ++k)
										sum += a[k * p.M + m] * b[n * p.K + k];
						}

						c[m * p.N + n] = sum;
				}
    )");

  conv2dPipeline = createComputePipeline("conv2d",
                                         R"(
				#version 450
				layout(local_size_x=256) in;

				layout(binding=0) buffer I { float in_d[]; };
				layout(binding=1) buffer W { float w_d[]; };
				layout(binding=2) buffer B { float b_d[]; };
				layout(binding=3) buffer O { float out_d[]; };

				layout(push_constant) uniform P {
						int B, iC, iH, iW;
						int oC, kH, kW;
						int s, p;
						int oH, oW;
						int has_bias;
				} u;

				void main() {

						int idx = int(gl_GlobalInvocationID.x);
						int total = u.B * u.oC * u.oH * u.oW;
						if (idx >= total) return;

						int ow = idx % u.oW;
						int tmp = idx / u.oW;

						int oh = tmp % u.oH;
						tmp /= u.oH;

						int oc = tmp % u.oC;
						int b  = tmp / u.oC;

						float sum = (u.has_bias == 1) ? b_d[oc] : 0.0;

						int in_b_off = b * u.iC * u.iH * u.iW;
						int w_oc_off = oc * u.iC * u.kH * u.kW;

						for (int ic = 0; ic < u.iC; ic++) {

								int in_c_off = in_b_off + ic * u.iH * u.iW;
								int w_ic_off = w_oc_off + ic * u.kH * u.kW;

								for (int kh = 0; kh < u.kH; kh++) {

										int ih = oh * u.s - u.p + kh;
										if (ih < 0 || ih >= u.iH) continue;

										int in_h_off = in_c_off + ih * u.iW;
										int w_h_off  = w_ic_off + kh * u.kW;

										for (int kw = 0; kw < u.kW; kw++) {

												int iw = ow * u.s - u.p + kw;
												if (iw < 0 || iw >= u.iW) continue;

												sum += in_d[in_h_off + iw] *
															 w_d[w_h_off + kw];
										}
								}
						}

						out_d[idx] = sum;
				}
     )");

  conv2dBackInputPipeline = createComputePipeline("conv2d_bi",
                                                  R"(
        #version 450
				layout(local_size_x=256) in;

				layout(binding=0) buffer GO { float go[]; };
				layout(binding=1) buffer W  { float w[]; };
				layout(binding=2) buffer GI { float gi[]; };

				layout(push_constant) uniform P {
						int B, iC, iH, iW;
						int oC, kH, kW;
						int s, p;
						int oH, oW;
				} u;

				void main() {

						int idx = int(gl_GlobalInvocationID.x);
						int total = u.B * u.iC * u.iH * u.iW;
						if (idx >= total) return;

						int iw = idx % u.iW;
						int tmp = idx / u.iW;

						int ih = tmp % u.iH;
						tmp /= u.iH;

						int ic = tmp % u.iC;
						int b  = tmp / u.iC;

						float d_in = 0.0;

						int go_b_off = b * u.oC * u.oH * u.oW;

						for (int oc = 0; oc < u.oC; oc++) {

								int go_oc_off = go_b_off + oc * u.oH * u.oW;
								int w_oc_off  = (oc * u.iC + ic) * u.kH * u.kW;

								for (int kh = 0; kh < u.kH; kh++) {

										int num_h = ih + u.p - kh;
										if (num_h < 0 || num_h % u.s != 0) continue;

										int oh = num_h / u.s;
										if (oh >= u.oH) continue;

										int go_h_off = go_oc_off + oh * u.oW;
										int w_h_off  = w_oc_off + kh * u.kW;

										for (int kw = 0; kw < u.kW; kw++) {

												int num_w = iw + u.p - kw;
												if (num_w < 0 || num_w % u.s != 0) continue;

												int ow = num_w / u.s;
												if (ow >= u.oW) continue;

												d_in += go[go_h_off + ow] *
																w[w_h_off + kw];
										}
								}
						}

						gi[idx] = d_in;
				}
     )");

  conv2dBackWeightPipeline = createComputePipeline("conv2d_bw",
                                                   R"(
        #version 450
				layout(local_size_x=256) in;

				layout(binding=0) buffer GO { float go[]; };
				layout(binding=1) buffer I  { float in_d[]; };
				layout(binding=2) buffer GW { float gw[]; };

				layout(push_constant) uniform P {
						int B, iC, iH, iW;
						int oC, kH, kW;
						int s, p;
						int oH, oW;
				} u;

				void main() {

						int idx = int(gl_GlobalInvocationID.x);
						int total = u.oC * u.iC * u.kH * u.kW;
						if (idx >= total) return;

						int kw = idx % u.kW;
						int tmp = idx / u.kW;

						int kh = tmp % u.kH;
						tmp /= u.kH;

						int ic = tmp % u.iC;
						int oc = tmp / u.iC;

						float dw = 0.0;

						int in_c_off = ic * u.iH * u.iW;
						int go_oc_off = oc * u.oH * u.oW;

						for (int b = 0; b < u.B; b++) {

								int in_b_off = b * u.iC * u.iH * u.iW + in_c_off;
								int go_b_off = b * u.oC * u.oH * u.oW + go_oc_off;

								for (int oh = 0; oh < u.oH; oh++) {

										int ih = oh * u.s - u.p + kh;
										if (ih < 0 || ih >= u.iH) continue;

										int in_h_off = in_b_off + ih * u.iW;
										int go_h_off = go_b_off + oh * u.oW;

										for (int ow = 0; ow < u.oW; ow++) {

												int iw = ow * u.s - u.p + kw;
												if (iw < 0 || iw >= u.iW) continue;

												dw += go[go_h_off + ow] *
															in_d[in_h_off + iw];
										}
								}
						}

						gw[idx] = dw;
				}
     )");

  conv2dBackBiasPipeline = createComputePipeline("conv2d_bb",
                                                 R"(
        #version 450
				layout(local_size_x=256) in;

				layout(binding=0) readonly buffer GO { float go[]; };
				layout(binding=1) writeonly buffer GB { float gb[]; };

				layout(push_constant) uniform P { uint B, oC, oH, oW; } u;

				void main() {

						uint oc = gl_GlobalInvocationID.x;
						if (oc >= u.oC) return;

						uint spatial = u.oH * u.oW;
						uint strideBC = u.oC * spatial;

						float db = 0.0;

						for (uint b = 0; b < u.B; ++b) {

								uint base = b * strideBC + oc * spatial;

								for (uint i = 0; i < spatial; ++i)
										db += go[base + i];
						}

						gb[oc] = db;
				}
     )");

  maxPoolPipeline = createComputePipeline("maxpool",
                                          R"(
        #version 450
				layout(local_size_x=256) in;

				layout(binding=0) readonly buffer I { float in_d[]; };
				layout(binding=1) writeonly buffer O { float out_d[]; };

				layout(push_constant) uniform P { uint B,C,iH,iW,k,s,p,oH,oW; } u;

				void main() {

						uint idx = gl_GlobalInvocationID.x;
						uint total = u.B * u.C * u.oH * u.oW;

						if (idx >= total) return;

						uint ow = idx % u.oW;
						uint tmp = idx / u.oW;

						uint oh = tmp % u.oH;
						tmp /= u.oH;

						uint c = tmp % u.C;
						uint b = tmp / u.C;

						uint inSpatial = u.iH * u.iW;
						uint base = (b * u.C + c) * inSpatial;

						float max_val = -3.402823e38;

						for (uint kh = 0; kh < u.k; ++kh)
						for (uint kw = 0; kw < u.k; ++kw) {

								int ih = int(oh * u.s + kh) - int(u.p);
								int iw = int(ow * u.s + kw) - int(u.p);

								if (ih >= 0 && ih < int(u.iH) &&
										iw >= 0 && iw < int(u.iW)) {

										float v = in_d[base + uint(ih)*u.iW + uint(iw)];
										max_val = max(max_val, v);
								}
						}

						out_d[idx] = max_val;
				}
     )");

  maxPoolBackPipeline = createComputePipeline("maxpool_b",
                                              R"(
        #version 450
				layout(local_size_x=256) in;

				layout(binding=0) readonly buffer GO { float go[]; };
				layout(binding=1) readonly buffer I { float in_d[]; };
				layout(binding=2) buffer GI { uint gi_u[]; };

				layout(push_constant) uniform P { uint B,C,iH,iW,k,s,p,oH,oW; } u;

				void atomicAddFloat(uint index, float val)
				{
						uint expected = gi_u[index];
						uint current;

						while (true)
						{
								float f = uintBitsToFloat(expected) + val;
								uint next = floatBitsToUint(f);

								current = atomicCompSwap(gi_u[index], expected, next);

								if (current == expected) break;
								expected = current;
						}
				}

				void main()
				{
						uint idx = gl_GlobalInvocationID.x;
						uint total = u.B*u.C*u.oH*u.oW;

						if (idx >= total) return;

						uint ow = idx % u.oW;
						uint tmp = idx / u.oW;

						uint oh = tmp % u.oH;
						tmp /= u.oH;

						uint c = tmp % u.C;
						uint b = tmp / u.C;

						uint inSpatial = u.iH*u.iW;
						uint base = (b*u.C + c)*inSpatial;

						float max_val = -3.402823e38;
						int max_idx = -1;

						for(uint kh=0; kh<u.k; ++kh)
						for(uint kw=0; kw<u.k; ++kw)
						{
								int ih = int(oh*u.s + kh) - int(u.p);
								int iw = int(ow*u.s + kw) - int(u.p);

								if(ih>=0 && ih<int(u.iH) &&
									 iw>=0 && iw<int(u.iW))
								{
										uint i_idx = base + uint(ih)*u.iW + uint(iw);
										float v = in_d[i_idx];

										if(v > max_val){
												max_val = v;
												max_idx = int(i_idx);
										}
								}
						}

						if(max_idx != -1)
								atomicAddFloat(uint(max_idx), go[idx]);
				}
     )");

  upsamplePipeline = createComputePipeline("upsample",
                                           R"(
        #version 450
				layout(local_size_x=256) in;

				layout(binding=0) readonly buffer I { float in_d[]; };
				layout(binding=1) writeonly buffer O { float out_d[]; };

				layout(push_constant) uniform P { uint B,C,iH,iW,scale; } u;

				void main()
				{
						uint idx = gl_GlobalInvocationID.x;

						uint oH = u.iH * u.scale;
						uint oW = u.iW * u.scale;

						uint total = u.B*u.C*oH*oW;
						if(idx >= total) return;

						uint ow = idx % oW;
						uint tmp = idx / oW;

						uint oh = tmp % oH;
						tmp /= oH;

						uint c = tmp % u.C;
						uint b = tmp / u.C;

						uint inSpatial = u.iH*u.iW;
						uint base = (b*u.C + c)*inSpatial;

						uint ih = oh / u.scale;
						uint iw = ow / u.scale;

						out_d[idx] = in_d[base + ih*u.iW + iw];
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

  concatPipeline = createComputePipeline("concat", R"(
      #version 450
      layout(local_size_x = 256) in;
      layout(binding = 0) buffer Src { float src[]; };
      layout(binding = 1) buffer Dst { float dst[]; };
      layout(push_constant) uniform P {
          int outer_size;
          int src_dim_size;
          int dst_dim_size;
          int inner_size;
          int offset;
          int forward;
      } u;
      void main() {
          int idx = int(gl_GlobalInvocationID.x);
          int total = u.outer_size * u.src_dim_size * u.inner_size;
          if (idx >= total) return;

          int i = idx % u.inner_size;
          int tmp = idx / u.inner_size;
          int d = tmp % u.src_dim_size;
          int o = tmp / u.src_dim_size;

          int src_idx = (o * u.src_dim_size + d) * u.inner_size + i;
          int dst_idx = (o * u.dst_dim_size + (u.offset + d)) * u.inner_size + i;

          if (u.forward == 1)
              dst[dst_idx] = src[src_idx];
          else
              src[src_idx] = dst[dst_idx];
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

  uniformPipeline = createComputePipeline("uniform", R"(
        #version 450
				layout(local_size_x=256) in;

				layout(binding=0) buffer O { float out_d[]; };

				layout(push_constant) uniform P {
						uint N;
						float low;
						float range;
						uint seed;
				} u;

				uint hash(uint x)
				{
						x += (x << 10u);
						x ^= (x >> 6u);
						x += (x << 3u);
						x ^= (x >> 11u);
						x += (x << 15u);
						return x;
				}

				void main()
				{
						uint idx = gl_GlobalInvocationID.x;
						if(idx >= u.N) return;

						uint r = hash(idx + u.seed);

						float r_norm = float(r) * (1.0 / 4294967295.0);

						out_d[idx] = u.low + r_norm * u.range;
				}
   )");

  sumPipeline = createComputePipeline("sum", R"(

        #version 450

				layout(local_size_x = 256) in;

				layout(binding = 0) buffer I { float in_d[]; };
				layout(binding = 1) buffer O { uint out_u[]; };

				layout(push_constant) uniform P {
						uint N;
				} u;

				shared float sdata[256];

				void atomicAddFloat(uint index, float val)
				{
						uint expected = out_u[index];
						uint current;

						while (true)
						{
								float f = uintBitsToFloat(expected) + val;
								uint next = floatBitsToUint(f);

								current = atomicCompSwap(out_u[index], expected, next);

								if (current == expected)
										break;

								expected = current;
						}
				}

				void main()
				{
						uint gid = gl_GlobalInvocationID.x;
						uint lid = gl_LocalInvocationID.x;

						float v = 0.0;
						if (gid < u.N)
								v = in_d[gid];

						sdata[lid] = v;

						barrier();

						// Workgroup reduction
						for (uint stride = 128; stride > 0; stride >>= 1)
						{
								if (lid < stride)
										sdata[lid] += sdata[lid + stride];

								barrier();
						}

						// One atomic add per workgroup
						if (lid == 0)
								atomicAddFloat(0, sdata[0]);
				}
   )");
}

VulkanBackend::~VulkanBackend() {
  vkDeviceWaitIdle(device); // Full stop
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroyFence(device, inFlightFences[i], nullptr);
    vkDestroyDescriptorPool(device, descriptorPools[i], nullptr);
    vkDestroyQueryPool(device, queryPools[i], nullptr);
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
  vkDestroyPipeline(device, adamStepPipeline, nullptr);

  vkDestroyPipeline(device, conv2dPipeline, nullptr);
  vkDestroyPipeline(device, conv2dBackInputPipeline, nullptr);
  vkDestroyPipeline(device, conv2dBackWeightPipeline, nullptr);
  vkDestroyPipeline(device, conv2dBackBiasPipeline, nullptr);

  vkDestroyPipeline(device, maxPoolPipeline, nullptr);
  vkDestroyPipeline(device, maxPoolBackPipeline, nullptr);

  vkDestroyPipeline(device, concatPipeline, nullptr);
  vkDestroyPipeline(device, broadcastRowPipeline, nullptr);

  vkDestroyPipeline(device, upsamplePipeline, nullptr);
  vkDestroyPipeline(device, upsampleBackPipeline, nullptr);

  vkDestroyPipeline(device, uniformPipeline, nullptr);
  vkDestroyPipeline(device, sumPipeline, nullptr);

  vkDestroyPipeline(device, bnCollectPipeline, nullptr);
  vkDestroyPipeline(device, bnUpdatePipeline, nullptr);
  vkDestroyPipeline(device, bnNormalizePipeline, nullptr);
  vkDestroyPipeline(device, bnBackReducePipeline, nullptr);
  vkDestroyPipeline(device, bnBackDxPipeline, nullptr);

  vkDestroyPipeline(device, addPipeline, nullptr);
  vkDestroyPipeline(device, mulPipeline, nullptr);
  vkDestroyPipeline(device, subPipeline, nullptr);

  vkDestroyPipeline(device, addBCPipeline, nullptr);
  vkDestroyPipeline(device, mulBCPipeline, nullptr);
  vkDestroyPipeline(device, subBCPipeline, nullptr);
  vkDestroyPipeline(device, sumToShapePipeline, nullptr);

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

  // Reset Descriptor Pool logic: wipe the slate clean for this frame
  VK_CHECK(vkResetDescriptorPool(device, descriptorPools[currentFrame], 0));
  allocate_frame_descriptor_sets(currentFrame);

  VK_CHECK(vkResetFences(device, 1, &inFlightFences[currentFrame]));

  // Start Recording
  VkCommandBuffer cmd = commandBuffers[currentFrame];
  VK_CHECK(vkResetCommandBuffer(cmd, 0));

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
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

  // Only attempt to fetch results if profiling is actually active and work was
  // done
  if (is_profile_enabled()) {
    uint64_t results[2];
    // The work we want to measure is in the frame we JUST flushed
    int frameToQuery =
        (currentFrame + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT;

    VkResult res = vkGetQueryPoolResults(
        device, queryPools[frameToQuery], 0, 2, sizeof(uint64_t) * 2, results,
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

    if (res == VK_SUCCESS) {
      double nanoseconds = (double)(results[1] - results[0]) * timestampPeriod;
      last_kernel_us_ = nanoseconds / 1000.0;
    } else {
      last_kernel_us_ = 0.0;
    }
  }
}
void VulkanBackend::all_reduce(Storage &buffer, size_t num_elements) {}

void VulkanBackend::dispatch_kernel(VkPipeline pipeline,
                                    const std::vector<void *> &buffers,
                                    void *pc, size_t pcSize, int x, int y,
                                    int z) {
  ensure_recording();
  VkCommandBuffer cmd = commandBuffers[currentFrame];

  if (descriptorSetCursor[currentFrame] >=
      frameDescriptorSets[currentFrame].size()) {
    flush_batch();
    ensure_recording();
  }

  VkDescriptorSet ds = frameDescriptorSets[currentFrame][descriptorSetCursor[currentFrame]++];

  // Update only the bindings used by this kernel to lower CPU overhead.
  const uint32_t write_count =
      static_cast<uint32_t>(std::max<size_t>(1, std::min<size_t>(buffers.size(), 8)));

  VkDescriptorBufferInfo bInfos[8]{};
  VkWriteDescriptorSet writes[8]{};

  for (uint32_t i = 0; i < write_count; ++i) {
    void *ptr = (buffers[i] != nullptr) ? buffers[i] : buffers[0];
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
  vkUpdateDescriptorSets(device, write_count, writes, 0, nullptr);

  VkMemoryBarrier memoryBarrier{};
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
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

  if (is_profile_enabled()) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        queryPools[currentFrame], 0);
  }
  vkCmdDispatch(cmd, x, y, z);
  if (is_profile_enabled()) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        queryPools[currentFrame], 1);
  }

  currentBatchSize++;
  if (currentBatchSize >= BATCH_SIZE_LIMIT) {
    flush_batch();
  }
}

// --- Kernel Wrappers ---
void VulkanBackend::add(const Storage &a, const Storage &b, Storage &out,
                        const BroadcastInfo &info) {
  size_t total = numel(info.out_shape);
  // Fast Path: Check if inputs are contiguous and match output shape
  if (info.strides_a == default_strides(info.out_shape) &&
      info.strides_b == default_strides(info.out_shape)) {
    uint32_t N = (uint32_t)total;
    dispatch_kernel(addPipeline, {a.data(), b.data(), out.data()}, &N,
                    sizeof(N), (N + 255) / 256, 1, 1);
  } else {
    auto gpu_info = to_gpu_info(info);
    size_t total = numel(info.out_shape);
    dispatch_kernel(addBCPipeline, {a.data(), b.data(), out.data()}, &gpu_info,
                    sizeof(gpu_info), (total + 255) / 256, 1, 1);
  }
}

void VulkanBackend::mul(const Storage &a, const Storage &b, Storage &out,
                        const BroadcastInfo &info) {
  size_t total = numel(info.out_shape);
  if (info.strides_a == default_strides(info.out_shape) &&
      info.strides_b == default_strides(info.out_shape)) {
    uint32_t N = (uint32_t)total;
    dispatch_kernel(mulPipeline, {a.data(), b.data(), out.data()}, &N,
                    sizeof(N), (N + 255) / 256, 1, 1);
  } else {
    auto gpu_info = to_gpu_info(info);
    size_t total = numel(info.out_shape);
    dispatch_kernel(mulBCPipeline, {a.data(), b.data(), out.data()}, &gpu_info,
                    sizeof(gpu_info), (total + 255) / 256, 1, 1);
  }
}

void VulkanBackend::sub(const Storage &a, const Storage &b, Storage &out,
                        const BroadcastInfo &info) {
  size_t total = numel(info.out_shape);
  if (info.strides_a == default_strides(info.out_shape) &&
      info.strides_b == default_strides(info.out_shape)) {
    uint32_t N = (uint32_t)total;
    dispatch_kernel(subPipeline, {a.data(), b.data(), out.data()}, &N,
                    sizeof(N), (N + 255) / 256, 1, 1);
  } else {
    auto gpu_info = to_gpu_info(info);
    size_t total = numel(info.out_shape);
    dispatch_kernel(subBCPipeline, {a.data(), b.data(), out.data()}, &gpu_info,
                    sizeof(gpu_info), (total + 255) / 256, 1, 1);
  }
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
                                  int num_classes, int spatial) {
  memset(out_loss.data(), 0, out_loss.size_bytes());
  struct {
    int B, C, Spatial;
  } pc = {batch_size, num_classes, spatial};
  // Dispatch threads = B * Spatial
  int total = batch_size * spatial;
  dispatch_kernel(crossEntropyPipeline,
                  {logits.data(), targets.data(), out_loss.data()}, &pc,
                  sizeof(pc), (total + 255) / 256, 1, 1);
}

void VulkanBackend::cross_entropy_backward(const Storage &grad_out,
                                           const Storage &logits,
                                           const Storage &targets,
                                           Storage &grad_in, int batch_size,
                                           int num_classes, int spatial) {
  struct {
    int B, C, Spatial;
  } pc = {batch_size, num_classes, spatial};
  int total = batch_size * spatial;
  dispatch_kernel(
      crossEntropyBackwardPipeline,
      {grad_out.data(), logits.data(), targets.data(), grad_in.data()}, &pc,
      sizeof(pc), (total + 255) / 256, 1, 1);
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

void VulkanBackend::fill_uniform(Storage &out, float low, float high,
                                 size_t num_elements) {
  struct {
    uint32_t N;
    float l;
    float r;
    uint32_t s;
  } pc = {(uint32_t)num_elements, low, high - low, (uint32_t)rand()};
  dispatch_kernel(uniformPipeline, {out.data()}, &pc, sizeof(pc),
                  (num_elements + 255) / 256, 1, 1);
}

void VulkanBackend::sum(const Storage &in, Storage &out, size_t num_elements) {
  memset(out.data(), 0, out.size_bytes());
  uint32_t N = num_elements;
  dispatch_kernel(sumPipeline, {in.data(), out.data()}, &N, sizeof(N),
                  (num_elements + 255) / 256, 1, 1);
}

void VulkanBackend::concat(const std::vector<Storage *> &inputs, Storage &out,
                           int dim, const std::vector<Shape> &shapes) {
  int outer_size = 1;
  for (int i = 0; i < dim; ++i)
    outer_size *= shapes[0][i];

  int inner_size = 1;
  for (int i = dim + 1; i < shapes[0].size(); ++i)
    inner_size *= shapes[0][i];

  int dst_dim_size = 0;
  for (const auto &s : shapes)
    dst_dim_size += s[dim];

  int current_offset = 0;
  for (size_t j = 0; j < inputs.size(); ++j) {
    int src_dim_size =
        shapes[j][dim]; // Use the correct shape for each input tensor

    // Define push constants
    struct {
      int outer_size, src_dim_size, dst_dim_size, inner_size, offset, forward;
    } pc = {outer_size, src_dim_size,   dst_dim_size,
            inner_size, current_offset, 1};

    // Dispatch the Vulkan compute kernel with the correct data
    dispatch_kernel(concatPipeline, {inputs[j]->data(), out.data()}, &pc,
                    sizeof(pc),
                    (outer_size * src_dim_size * inner_size + 255) / 256, 1, 1);

    current_offset += src_dim_size; // Update offset for the next tensor
  }
}

void VulkanBackend::concat_backward(const Storage &grad_out,
                                    std::vector<Storage *> &grad_inputs,
                                    int dim, const std::vector<Shape> &shapes) {
  int outer_size = 1;
  for (int i = 0; i < dim; ++i)
    outer_size *= shapes[0][i];

  int inner_size = 1;
  for (int i = dim + 1; i < shapes[0].size(); ++i)
    inner_size *= shapes[0][i];

  int dst_dim_size = 0;
  for (const auto &s : shapes)
    dst_dim_size += s[dim];

  int current_offset = 0;
  for (size_t j = 0; j < grad_inputs.size(); ++j) {
    int src_dim_size =
        shapes[j][dim]; // Use the correct shape for each gradient input tensor

    // Define push constants for backward pass
    struct {
      int outer_size, src_dim_size, dst_dim_size, inner_size, offset, forward;
    } pc = {outer_size, src_dim_size,   dst_dim_size,
            inner_size, current_offset, 0};

    // Dispatch the Vulkan compute kernel with the correct data
    dispatch_kernel(
        concatPipeline, {grad_inputs[j]->data(), (void *)grad_out.data()}, &pc,
        sizeof(pc), (outer_size * src_dim_size * inner_size + 255) / 256, 1, 1);

    current_offset += src_dim_size; // Update offset for the next gradient input
  }
}

void VulkanBackend::broadcast_row(const Storage &src, Storage &dst, int rows,
                                  int cols) {
  struct {
    int rows, cols;
  } pc = {rows, cols};
  dispatch_kernel(broadcastRowPipeline, {src.data(), dst.data()}, &pc,
                  sizeof(pc), (rows * cols + 255) / 256, 1, 1);
}

// Add method implementation
void VulkanBackend::adam_step(Storage &params, const Storage &grads,
                              Storage &exp_avg, Storage &exp_avg_sq, float lr,
                              float beta1, float beta2, float eps, int step,
                              size_t num_elements) {
  struct {
    uint32_t N;
    float lr, beta1, beta2, eps;
    int step;
  } pc = {(uint32_t)num_elements, lr, beta1, beta2, eps, step};

  dispatch_kernel(
      adamStepPipeline,
      {params.data(), grads.data(), exp_avg.data(), exp_avg_sq.data()}, &pc,
      sizeof(pc), (num_elements + 255) / 256, 1, 1);
}

void VulkanBackend::sum_to_shape(const Storage &in, Storage &out,
                                 const Shape &in_shape,
                                 const Shape &out_shape) {

  memset(out.data(), 0, out.size_bytes());

  struct {
    int ndim;
    int out_ndim;
    int in_shape[6];
    int out_shape[6];
    int out_strides[6];
    uint32_t N;
  } pc{};

  pc.ndim = in_shape.size();
  pc.out_ndim = out_shape.size();
  pc.N = numel(in_shape);

  auto ost = default_strides(out_shape);

  for (int i = 0; i < 6; ++i) {
    pc.in_shape[i] = i < pc.ndim ? in_shape[i] : 1;
    pc.out_shape[i] = i < pc.out_ndim ? out_shape[i] : 1;
    pc.out_strides[i] = i < pc.out_ndim ? ost[i] : 0;
  }

  dispatch_kernel(sumToShapePipeline, {in.data(), out.data()}, &pc, sizeof(pc),
                  (pc.N + 255) / 256, 1, 1);
}

} // namespace munet
