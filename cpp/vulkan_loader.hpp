#pragma once
#include <cstdlib>
#include <mutex>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace munet { namespace {
static PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers=nullptr;
static PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets=nullptr;
static PFN_vkAllocateMemory vkAllocateMemory=nullptr;
static PFN_vkBeginCommandBuffer vkBeginCommandBuffer=nullptr;
static PFN_vkBindBufferMemory vkBindBufferMemory=nullptr;
static PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets=nullptr;
static PFN_vkCmdBindPipeline vkCmdBindPipeline=nullptr;
static PFN_vkCmdCopyBuffer vkCmdCopyBuffer=nullptr;
static PFN_vkCmdDispatch vkCmdDispatch=nullptr;
static PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier=nullptr;
static PFN_vkCreateBuffer vkCreateBuffer=nullptr;
static PFN_vkCreateCommandPool vkCreateCommandPool=nullptr;
static PFN_vkCreateComputePipelines vkCreateComputePipelines=nullptr;
static PFN_vkCreateDescriptorPool vkCreateDescriptorPool=nullptr;
static PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout=nullptr;
static PFN_vkCreateDevice vkCreateDevice=nullptr;
static PFN_vkCreateFence vkCreateFence=nullptr;
static PFN_vkCreateInstance vkCreateInstance=nullptr;
static PFN_vkCreatePipelineCache vkCreatePipelineCache=nullptr;
static PFN_vkCreatePipelineLayout vkCreatePipelineLayout=nullptr;
static PFN_vkCreateShaderModule vkCreateShaderModule=nullptr;
static PFN_vkDestroyBuffer vkDestroyBuffer=nullptr;
static PFN_vkDestroyCommandPool vkDestroyCommandPool=nullptr;
static PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool=nullptr;
static PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout=nullptr;
static PFN_vkDestroyDevice vkDestroyDevice=nullptr;
static PFN_vkDestroyFence vkDestroyFence=nullptr;
static PFN_vkDestroyInstance vkDestroyInstance=nullptr;
static PFN_vkDestroyPipeline vkDestroyPipeline=nullptr;
static PFN_vkDestroyPipelineCache vkDestroyPipelineCache=nullptr;
static PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout=nullptr;
static PFN_vkDestroyShaderModule vkDestroyShaderModule=nullptr;
static PFN_vkDeviceWaitIdle vkDeviceWaitIdle=nullptr;
static PFN_vkEndCommandBuffer vkEndCommandBuffer=nullptr;
static PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties=nullptr;
static PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties=nullptr;
static PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices=nullptr;
static PFN_vkFreeMemory vkFreeMemory=nullptr;
static PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements=nullptr;
static PFN_vkGetDeviceQueue vkGetDeviceQueue=nullptr;
static PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties=nullptr;
static PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties=nullptr;
static PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties=nullptr;
static PFN_vkMapMemory vkMapMemory=nullptr;
static PFN_vkQueueSubmit vkQueueSubmit=nullptr;
static PFN_vkResetCommandBuffer vkResetCommandBuffer=nullptr;
static PFN_vkResetFences vkResetFences=nullptr;
static PFN_vkUnmapMemory vkUnmapMemory=nullptr;
static PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets=nullptr;
static PFN_vkWaitForFences vkWaitForFences=nullptr;

// Keep the driver loader alive until process exit. CPU execution never opens it.
void load_vulkan() {
  static std::once_flag once;
  std::call_once(once, [] {
    const char* override_path=std::getenv("MUNET_VULKAN_LIBRARY");
#ifdef _WIN32
    auto library=LoadLibraryA(override_path?override_path:"vulkan-1.dll");
    auto symbol=[&](const char* name){return GetProcAddress(library,name);};
    auto close=[&](){FreeLibrary(library);};
#else
#ifdef __APPLE__
    const char* default_name="libvulkan.1.dylib";
#else
    const char* default_name="libvulkan.so.1";
#endif
    auto library=dlopen(override_path?override_path:default_name,RTLD_NOW|RTLD_LOCAL);
    auto symbol=[&](const char* name){return dlsym(library,name);};
    auto close=[&](){dlclose(library);};
#endif
    if(!library)throw std::runtime_error("Vulkan loader unavailable; install a Vulkan driver/loader or select device=cpu explicitly");
    try {
#define MUNET_LOAD(name) name=reinterpret_cast<PFN_##name>(symbol(#name)); if(!name)throw std::runtime_error("Vulkan loader missing " #name)

      MUNET_LOAD(vkAllocateCommandBuffers);
      MUNET_LOAD(vkAllocateDescriptorSets);
      MUNET_LOAD(vkAllocateMemory);
      MUNET_LOAD(vkBeginCommandBuffer);
      MUNET_LOAD(vkBindBufferMemory);
      MUNET_LOAD(vkCmdBindDescriptorSets);
      MUNET_LOAD(vkCmdBindPipeline);
      MUNET_LOAD(vkCmdCopyBuffer);
      MUNET_LOAD(vkCmdDispatch);
      MUNET_LOAD(vkCmdPipelineBarrier);
      MUNET_LOAD(vkCreateBuffer);
      MUNET_LOAD(vkCreateCommandPool);
      MUNET_LOAD(vkCreateComputePipelines);
      MUNET_LOAD(vkCreateDescriptorPool);
      MUNET_LOAD(vkCreateDescriptorSetLayout);
      MUNET_LOAD(vkCreateDevice);
      MUNET_LOAD(vkCreateFence);
      MUNET_LOAD(vkCreateInstance);
      MUNET_LOAD(vkCreatePipelineCache);
      MUNET_LOAD(vkCreatePipelineLayout);
      MUNET_LOAD(vkCreateShaderModule);
      MUNET_LOAD(vkDestroyBuffer);
      MUNET_LOAD(vkDestroyCommandPool);
      MUNET_LOAD(vkDestroyDescriptorPool);
      MUNET_LOAD(vkDestroyDescriptorSetLayout);
      MUNET_LOAD(vkDestroyDevice);
      MUNET_LOAD(vkDestroyFence);
      MUNET_LOAD(vkDestroyInstance);
      MUNET_LOAD(vkDestroyPipeline);
      MUNET_LOAD(vkDestroyPipelineCache);
      MUNET_LOAD(vkDestroyPipelineLayout);
      MUNET_LOAD(vkDestroyShaderModule);
      MUNET_LOAD(vkDeviceWaitIdle);
      MUNET_LOAD(vkEndCommandBuffer);
      MUNET_LOAD(vkEnumerateDeviceExtensionProperties);
      MUNET_LOAD(vkEnumerateInstanceExtensionProperties);
      MUNET_LOAD(vkEnumeratePhysicalDevices);
      MUNET_LOAD(vkFreeMemory);
      MUNET_LOAD(vkGetBufferMemoryRequirements);
      MUNET_LOAD(vkGetDeviceQueue);
      MUNET_LOAD(vkGetPhysicalDeviceMemoryProperties);
      MUNET_LOAD(vkGetPhysicalDeviceProperties);
      MUNET_LOAD(vkGetPhysicalDeviceQueueFamilyProperties);
      MUNET_LOAD(vkMapMemory);
      MUNET_LOAD(vkQueueSubmit);
      MUNET_LOAD(vkResetCommandBuffer);
      MUNET_LOAD(vkResetFences);
      MUNET_LOAD(vkUnmapMemory);
      MUNET_LOAD(vkUpdateDescriptorSets);
      MUNET_LOAD(vkWaitForFences);
#undef MUNET_LOAD
    } catch(...) {close();throw;}
  });
}
} } // namespace munet, anonymous
