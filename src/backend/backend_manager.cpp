#include "backend.hpp"
#include "backend/cpu_backend.hpp"
#include "backend/debug_backend.hpp"
#include "util.hpp"

#ifdef MUNET_USE_CUDA
#include "backend/cuda_backend.hpp"
#endif

#ifdef MUNET_USE_VULKAN
#include "backend/vulkan_backend.hpp"
#endif

#include <functional>
#include <mutex>
#include <unordered_map>

namespace munet {

namespace {
std::unordered_map<int, std::shared_ptr<Backend>> &backend_cache() {
  static std::unordered_map<int, std::shared_ptr<Backend>> cache;
  return cache;
}

std::unordered_map<DeviceType, BackendManager::BackendFactory> &backend_factories() {
  static std::unordered_map<DeviceType, BackendManager::BackendFactory> factories;
  return factories;
}

std::mutex &backend_mutex() {
  static std::mutex m;
  return m;
}

void register_default_backends() {
  static bool initialized = false;
  if (initialized)
    return;

  std::lock_guard<std::mutex> lock(backend_mutex());
  auto &factories = backend_factories();

  factories.try_emplace(DeviceType::CPU,
                        [](Device) { return std::make_shared<CPUBackend>(); });

#ifdef MUNET_USE_CUDA
  factories.try_emplace(DeviceType::CUDA, [](Device device) {
    return std::make_shared<CUDABackend>(device.index);
  });
#endif

#ifdef MUNET_USE_VULKAN
  factories.try_emplace(DeviceType::VULKAN,
                        [](Device device) {
                          return std::make_shared<VulkanBackend>(device.index);
                        });
#endif

  initialized = true;
}

} // namespace

void BackendManager::register_backend(DeviceType type,
                                      BackendFactory factory) {
  std::lock_guard<std::mutex> lock(backend_mutex());
  backend_factories()[type] = std::move(factory);

  for (auto it = backend_cache().begin(); it != backend_cache().end();) {
    int cached_type = it->first / 1000;
    if (cached_type == static_cast<int>(type)) {
      it = backend_cache().erase(it);
    } else {
      ++it;
    }
  }
}

std::shared_ptr<Backend> BackendManager::get(Device device) {
  register_default_backends();

  std::lock_guard<std::mutex> lock(backend_mutex());
  int key = static_cast<int>(device.type) * 1000 + device.index;

  auto cache_it = backend_cache().find(key);
  if (cache_it != backend_cache().end())
    return cache_it->second;

  auto factory_it = backend_factories().find(device.type);
  if (factory_it == backend_factories().end()) {
    throw std::runtime_error("Requested backend not compiled or registered.");
  }

  auto backend = factory_it->second(device);
  if (is_debug_enabled() || is_profile_enabled()) {
    backend = wrap_with_debug_backend(std::move(backend));
  }

  backend_cache()[key] = backend;
  return backend;
}

} // namespace munet
