#include "core/backend.hpp"
#include "backend/cpu_backend.hpp"
#include "backend/debug_backend.hpp"
#include "core/util.hpp"

#ifdef MUNET_USE_CUDA
#include "backend/cuda_backend.hpp"
#endif

#ifdef MUNET_USE_VULKAN
#include "backend/vulkan_backend.hpp"
#endif

#include <mutex>

namespace munet {
namespace {

void register_default_backends(BackendRegistry &registry) {
  static std::once_flag once;
  std::call_once(once, [&registry]() {
    registry.register_backend(DeviceType::CPU,
                             [](Device) { return std::make_shared<CPUBackend>(); });

#ifdef MUNET_USE_CUDA
    registry.register_backend(DeviceType::CUDA, [](Device device) {
      return std::make_shared<CUDABackend>(device.index);
    });
#endif

#ifdef MUNET_USE_VULKAN
    registry.register_backend(DeviceType::VULKAN, [](Device device) {
      return std::make_shared<VulkanBackend>(device.index);
    });
#endif

    registry.set_decorator([](std::shared_ptr<Backend> backend) {
      if (is_debug_enabled() || is_profile_enabled()) {
        return wrap_with_debug_backend(std::move(backend));
      }
      return backend;
    });
  });
}

int backend_cache_key(Device device) {
  return static_cast<int>(device.type) * 1000 + device.index;
}

} // namespace

void BackendRegistry::register_backend(DeviceType type, BackendFactory factory) {
  std::lock_guard<std::mutex> lock(mutex_);
  factories_[type] = std::move(factory);

  for (auto it = cache_.begin(); it != cache_.end();) {
    const int cached_type = it->first / 1000;
    if (cached_type == static_cast<int>(type)) {
      it = cache_.erase(it);
    } else {
      ++it;
    }
  }
}

std::shared_ptr<Backend> BackendRegistry::get(Device device) {
  std::lock_guard<std::mutex> lock(mutex_);
  const int key = backend_cache_key(device);

  auto cache_it = cache_.find(key);
  if (cache_it != cache_.end()) {
    return cache_it->second;
  }

  auto factory_it = factories_.find(device.type);
  if (factory_it == factories_.end()) {
    throw std::runtime_error("Requested backend not compiled or registered.");
  }

  auto backend = factory_it->second(device);
  if (decorator_) {
    backend = decorator_(std::move(backend));
  }

  cache_[key] = backend;
  return backend;
}

void BackendRegistry::clear_cache(DeviceType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = cache_.begin(); it != cache_.end();) {
    const int cached_type = it->first / 1000;
    if (cached_type == static_cast<int>(type)) {
      it = cache_.erase(it);
    } else {
      ++it;
    }
  }
}

void BackendRegistry::clear_all() {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
  factories_.clear();
  decorator_ = nullptr;
}

void BackendRegistry::set_decorator(BackendDecorator decorator) {
  std::lock_guard<std::mutex> lock(mutex_);
  decorator_ = std::move(decorator);
  cache_.clear();
}

BackendRegistry &default_backend_registry() {
  static BackendRegistry registry;
  return registry;
}

BackendRegistry &BackendManager::registry() {
  auto &registry = default_backend_registry();
  register_default_backends(registry);
  return registry;
}

void BackendManager::register_backend(DeviceType type, BackendFactory factory) {
  registry().register_backend(type, std::move(factory));
}

std::shared_ptr<Backend> BackendManager::get(Device device) {
  return registry().get(device);
}

} // namespace munet
