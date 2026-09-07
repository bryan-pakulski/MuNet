#include "core.hpp"
#include <stdexcept>
#ifdef MUNET_VULKAN
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "vulkan_loader.hpp"
#include <algorithm>
#include <cstring>
#include <limits>

namespace munet {
namespace {
void check(VkResult result,const char* operation) {
  if(result!=VK_SUCCESS) throw std::runtime_error(std::string(operation)+" failed (VkResult "+std::to_string(result)+")");
}
VkInstance instance_create() {
  load_vulkan();
  uint32_t count=0;check(vkEnumerateInstanceExtensionProperties(nullptr,&count,nullptr),"enumerate instance extensions");
  std::vector<VkExtensionProperties> props(count);check(vkEnumerateInstanceExtensionProperties(nullptr,&count,props.data()),"enumerate instance extensions");
  std::vector<const char*> extensions;
  for(auto& p:props) if(std::strcmp(p.extensionName,"VK_KHR_portability_enumeration")==0)
    extensions.push_back("VK_KHR_portability_enumeration");
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="MuNet Next";app.apiVersion=VK_API_VERSION_1_1;
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ci.pApplicationInfo=&app;
  ci.enabledExtensionCount=extensions.size();ci.ppEnabledExtensionNames=extensions.data();
  if(!extensions.empty()) ci.flags=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  VkInstance instance=VK_NULL_HANDLE;check(vkCreateInstance(&ci,nullptr,&instance),"vkCreateInstance");return instance;
}
std::vector<VkPhysicalDevice> enumerate(VkInstance instance) {
  uint32_t count=0;check(vkEnumeratePhysicalDevices(instance,&count,nullptr),"enumerate devices");
  std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(instance,&count,devices.data()),"enumerate devices");return devices;
}
struct Buffer {VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;void* mapped=nullptr;};
class Vulkan final:public DeviceExecutor {
  VkInstance instance_=VK_NULL_HANDLE;
  VkPhysicalDevice physical_=VK_NULL_HANDLE;
  VkDevice device_=VK_NULL_HANDLE;
  VkQueue queue_=VK_NULL_HANDLE;
  VkCommandPool pool_=VK_NULL_HANDLE;
  VkCommandBuffer replay_=VK_NULL_HANDLE,transfer_=VK_NULL_HANDLE;
  VkFence fence_=VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout_=VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_=VK_NULL_HANDLE;
  VkDescriptorSet set_=VK_NULL_HANDLE;
  VkPipelineLayout layout_=VK_NULL_HANDLE;
  VkPipelineCache cache_=VK_NULL_HANDLE;
  std::vector<VkPipeline> pipelines_;
  Buffer arena_,staging_;
  VkPhysicalDeviceProperties props_{};
  VkPhysicalDeviceMemoryProperties memory_{};
  bool pending_=false;
  Counters& stats_;
  void cleanup() noexcept {
    if(device_) {
      vkDeviceWaitIdle(device_);
      for(auto pipeline:pipelines_) vkDestroyPipeline(device_,pipeline,nullptr);
      if(cache_)vkDestroyPipelineCache(device_,cache_,nullptr);
      if(layout_)vkDestroyPipelineLayout(device_,layout_,nullptr);
      if(descriptor_pool_)vkDestroyDescriptorPool(device_,descriptor_pool_,nullptr);
      if(set_layout_)vkDestroyDescriptorSetLayout(device_,set_layout_,nullptr);
      if(fence_)vkDestroyFence(device_,fence_,nullptr);
      if(pool_)vkDestroyCommandPool(device_,pool_,nullptr);
      for(auto* b:{&arena_,&staging_}) {
        if(b->mapped)vkUnmapMemory(device_,b->memory);
        if(b->buffer)vkDestroyBuffer(device_,b->buffer,nullptr);
        if(b->memory)vkFreeMemory(device_,b->memory,nullptr);
      }
      vkDestroyDevice(device_,nullptr);
    }
    if(instance_)vkDestroyInstance(instance_,nullptr);
  }
  void buffer(Buffer& out,VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags flags) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=size;bi.usage=usage;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device_,&bi,nullptr,&out.buffer),"create buffer");
    VkMemoryRequirements req;vkGetBufferMemoryRequirements(device_,out.buffer,&req);
    uint32_t index=UINT32_MAX;
    for(uint32_t i=0;i<memory_.memoryTypeCount;++i) if((req.memoryTypeBits&(1u<<i))&&(memory_.memoryTypes[i].propertyFlags&flags)==flags) {index=i;break;}
    if(index==UINT32_MAX)throw std::runtime_error("required Vulkan memory type is unavailable");
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=index;
    check(vkAllocateMemory(device_,&ai,nullptr,&out.memory),"allocate memory");check(vkBindBufferMemory(device_,out.buffer,out.memory,0),"bind buffer");
    if(flags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) check(vkMapMemory(device_,out.memory,0,VK_WHOLE_SIZE,0,&out.mapped),"map staging memory");
  }
  void barrier(VkCommandBuffer cmd,VkPipelineStageFlags src_stage,VkAccessFlags src,VkPipelineStageFlags dst_stage,VkAccessFlags dst) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=src;b.dstAccessMask=dst;
    vkCmdPipelineBarrier(cmd,src_stage,dst_stage,0,1,&b,0,nullptr,0,nullptr);
  }
  void begin_transfer() {
    synchronize();check(vkResetCommandBuffer(transfer_,0),"reset transfer commands");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(transfer_,&bi),"begin transfer commands");
    barrier(transfer_,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_MEMORY_WRITE_BIT|VK_ACCESS_MEMORY_READ_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT);
  }
  void submit(VkCommandBuffer cmd) {
    check(vkResetFences(device_,1,&fence_),"reset fence");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cmd;
    check(vkQueueSubmit(queue_,1,&si,fence_),"submit commands");pending_=true;++stats_.submissions;
  }
 public:
  Vulkan(const std::vector<float>& initial,const std::vector<Kernel>& kernels,
         const std::vector<std::vector<uint32_t>>& spirv,
         const std::vector<std::pair<size_t,size_t>>& inputs,
         const std::vector<std::pair<size_t,std::pair<size_t,size_t>>>& updates,
         unsigned index,Counters& counters):stats_(counters) {
    try {
      if(spirv.size()!=kernels.size())throw std::invalid_argument("SPIR-V count does not match kernels");
      instance_=instance_create();auto devices=enumerate(instance_);
      if(index>=devices.size())throw std::runtime_error("Vulkan device index unavailable; inspect munet.devices()");
      physical_=devices[index];vkGetPhysicalDeviceProperties(physical_,&props_);vkGetPhysicalDeviceMemoryProperties(physical_,&memory_);
      if(props_.apiVersion<VK_API_VERSION_1_1)throw std::runtime_error("MuNet v0 requires Vulkan 1.1");
      if(initial.empty()||initial.size()*sizeof(float)>props_.limits.maxStorageBufferRange)
        throw std::runtime_error("graph arena exceeds device maxStorageBufferRange; split-arena allocation is not implemented");
      if(props_.limits.maxComputeWorkGroupInvocations<64||props_.limits.maxComputeWorkGroupSize[0]<64)
        throw std::runtime_error("device cannot execute the 64-thread baseline kernel");
      uint32_t count=0;vkGetPhysicalDeviceQueueFamilyProperties(physical_,&count,nullptr);
      std::vector<VkQueueFamilyProperties> queues(count);vkGetPhysicalDeviceQueueFamilyProperties(physical_,&count,queues.data());
      uint32_t family=UINT32_MAX;
      for(uint32_t i=0;i<count;++i)if(queues[i].queueCount&&(queues[i].queueFlags&VK_QUEUE_COMPUTE_BIT)){family=i;break;}
      if(family==UINT32_MAX)throw std::runtime_error("device has no compute queue");
      check(vkEnumerateDeviceExtensionProperties(physical_,nullptr,&count,nullptr),"enumerate device extensions");
      std::vector<VkExtensionProperties> extensions(count);check(vkEnumerateDeviceExtensionProperties(physical_,nullptr,&count,extensions.data()),"enumerate device extensions");
      std::vector<const char*> enabled;
      for(auto& e:extensions)if(std::strcmp(e.extensionName,"VK_KHR_portability_subset")==0)enabled.push_back("VK_KHR_portability_subset");
      float priority=1.f;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
      VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;di.enabledExtensionCount=enabled.size();di.ppEnabledExtensionNames=enabled.data();
      check(vkCreateDevice(physical_,&di,nullptr,&device_),"create device");vkGetDeviceQueue(device_,family,0,&queue_);
      VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.queueFamilyIndex=family;pi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      check(vkCreateCommandPool(device_,&pi,nullptr,&pool_),"create command pool");
      VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=pool_;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;
      check(vkAllocateCommandBuffers(device_,&ca,&replay_),"allocate replay commands");check(vkAllocateCommandBuffers(device_,&ca,&transfer_),"allocate transfer commands");
      VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(device_,&fi,nullptr,&fence_),"create fence");
      auto bytes=initial.size()*sizeof(float);
      buffer(arena_,bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      buffer(staging_,bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      VkDescriptorSetLayoutBinding binding{};binding.binding=0;binding.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;binding.descriptorCount=1;binding.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
      VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sl.bindingCount=1;sl.pBindings=&binding;
      check(vkCreateDescriptorSetLayout(device_,&sl,nullptr,&set_layout_),"create descriptor layout");
      VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1};VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=1;dp.poolSizeCount=1;dp.pPoolSizes=&ps;
      check(vkCreateDescriptorPool(device_,&dp,nullptr,&descriptor_pool_),"create descriptor pool");
      VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};da.descriptorPool=descriptor_pool_;da.descriptorSetCount=1;da.pSetLayouts=&set_layout_;
      check(vkAllocateDescriptorSets(device_,&da,&set_),"allocate descriptor set");
      VkDescriptorBufferInfo db{arena_.buffer,0,bytes};VkWriteDescriptorSet wd{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};wd.dstSet=set_;wd.dstBinding=0;wd.descriptorCount=1;wd.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;wd.pBufferInfo=&db;
      vkUpdateDescriptorSets(device_,1,&wd,0,nullptr);
      VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&set_layout_;
      check(vkCreatePipelineLayout(device_,&pl,nullptr,&layout_),"create pipeline layout");
      VkPipelineCacheCreateInfo pc{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};check(vkCreatePipelineCache(device_,&pc,nullptr,&cache_),"create pipeline cache");
      for(const auto& code:spirv) {
        if(code.size()<5||code[0]!=0x07230203)throw std::invalid_argument("invalid SPIR-V module");
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=code.size()*4;sm.pCode=code.data();VkShaderModule module=VK_NULL_HANDLE;
        check(vkCreateShaderModule(device_,&sm,nullptr,&module),"create shader module");
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.layout=layout_;ci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";
        VkPipeline pipeline=VK_NULL_HANDLE;auto result=vkCreateComputePipelines(device_,cache_,1,&ci,nullptr,&pipeline);
        vkDestroyShaderModule(device_,module,nullptr);if(pipeline)pipelines_.push_back(pipeline);check(result,"create compute pipeline");
      }
      write(0,initial); // Parameters and constants are uploaded once, before replay.
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(replay_,&bi),"begin replay commands");
      barrier(replay_,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_MEMORY_WRITE_BIT|VK_ACCESS_MEMORY_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
      for(auto [offset,size]:inputs) {VkBufferCopy c{offset*4,offset*4,size*4};vkCmdCopyBuffer(replay_,staging_.buffer,arena_.buffer,1,&c);}
      barrier(replay_,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
      vkCmdBindDescriptorSets(replay_,VK_PIPELINE_BIND_POINT_COMPUTE,layout_,0,1,&set_,0,nullptr);
      for(size_t k=0;k<kernels.size();++k) {
        auto groups=(kernels[k].count+63)/64;
        if(groups>props_.limits.maxComputeWorkGroupCount[0])throw std::runtime_error("dispatch exceeds device workgroup-count limit");
        vkCmdBindPipeline(replay_,VK_PIPELINE_BIND_POINT_COMPUTE,pipelines_[k]);vkCmdDispatch(replay_,groups,1,1);
        // Include read->write hazards introduced by reusing temporary storage.
        barrier(replay_,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
      }
      barrier(replay_,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT);
      for(auto [dst,source]:updates) {auto [src,size]=source;VkBufferCopy c{src*4,dst*4,size*4};vkCmdCopyBuffer(replay_,arena_.buffer,arena_.buffer,1,&c);}
      barrier(replay_,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT);
      check(vkEndCommandBuffer(replay_),"end replay commands");
    } catch(...) {cleanup();throw;}
  }
  ~Vulkan()override{cleanup();}
  void synchronize()override {
    if(pending_) {check(vkWaitForFences(device_,1,&fence_,VK_TRUE,UINT64_MAX),"wait for execution");pending_=false;++stats_.waits;}
  }
  void run(const std::vector<std::pair<size_t,std::vector<float>>>& feeds)override {
    synchronize(); // One in-flight replay in v0; never wait between operators.
    for(auto& [offset,data]:feeds){std::memcpy(static_cast<float*>(staging_.mapped)+offset,data.data(),data.size()*4);stats_.upload_bytes+=data.size()*4;}
    submit(replay_);
  }
  std::vector<float> read(size_t offset,size_t count)override {
    begin_transfer();VkBufferCopy c{offset*4,offset*4,count*4};vkCmdCopyBuffer(transfer_,arena_.buffer,staging_.buffer,1,&c);
    barrier(transfer_,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_HOST_READ_BIT);
    check(vkEndCommandBuffer(transfer_),"end readback commands");submit(transfer_);synchronize();
    auto p=static_cast<float*>(staging_.mapped)+offset;stats_.download_bytes+=count*4;return {p,p+count};
  }
  void write(size_t offset,const std::vector<float>& data)override {
    begin_transfer();std::memcpy(static_cast<float*>(staging_.mapped)+offset,data.data(),data.size()*4);
    VkBufferCopy c{offset*4,offset*4,data.size()*4};vkCmdCopyBuffer(transfer_,staging_.buffer,arena_.buffer,1,&c);
    barrier(transfer_,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
    check(vkEndCommandBuffer(transfer_),"end upload commands");submit(transfer_);synchronize();stats_.upload_bytes+=data.size()*4;
  }
  std::string name()const override{return props_.deviceName;}
};
}
bool vulkan_built(){return true;}
std::vector<std::string> vulkan_devices(){
  VkInstance instance=instance_create();std::vector<std::string> result;
  try {for(auto d:enumerate(instance)){VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(d,&p);result.push_back(p.deviceName);}}
  catch(...){vkDestroyInstance(instance,nullptr);throw;}
  vkDestroyInstance(instance,nullptr);return result;
}
std::map<std::string,uint64_t> vulkan_device_limits(unsigned index){
  VkInstance instance=instance_create();std::map<std::string,uint64_t> result;
  try {
    auto devices=enumerate(instance);
    if(index>=devices.size())throw std::invalid_argument("Vulkan device index out of range");
    VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(devices[index],&p);
    VkPhysicalDeviceMemoryProperties m;vkGetPhysicalDeviceMemoryProperties(devices[index],&m);
    uint64_t heap=0;
    for(uint32_t i=0;i<m.memoryHeapCount;++i)
      if(m.memoryHeaps[i].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)heap=std::max(heap,uint64_t(m.memoryHeaps[i].size));
    result={{"max_buffer_bytes",p.limits.maxStorageBufferRange},{"device_heap_bytes",heap},{"api_version",p.apiVersion}};
  } catch(...){vkDestroyInstance(instance,nullptr);throw;}
  vkDestroyInstance(instance,nullptr);return result;
}
std::unique_ptr<DeviceExecutor> make_vulkan(const std::vector<float>& initial,const std::vector<Kernel>& kernels,
  const std::vector<std::vector<uint32_t>>& spirv,const std::vector<std::pair<size_t,size_t>>& inputs,
  const std::vector<std::pair<size_t,std::pair<size_t,size_t>>>& updates,unsigned index,Counters& stats) {
  return std::make_unique<Vulkan>(initial,kernels,spirv,inputs,updates,index,stats);
}
}
#else
namespace munet {
bool vulkan_built(){return false;}
std::vector<std::string> vulkan_devices(){throw std::runtime_error("MuNet was built without Vulkan; rebuild with MUNET_VULKAN=ON");}
std::map<std::string,uint64_t> vulkan_device_limits(unsigned){throw std::runtime_error("MuNet was built without Vulkan");}
std::unique_ptr<DeviceExecutor> make_vulkan(const std::vector<float>&,const std::vector<Kernel>&,
  const std::vector<std::vector<uint32_t>>&,const std::vector<std::pair<size_t,size_t>>&,
  const std::vector<std::pair<size_t,std::pair<size_t,size_t>>>&,unsigned,Counters&) {
  throw std::runtime_error("MuNet was built without Vulkan; rebuild with MUNET_VULKAN=ON");
}
}
#endif
