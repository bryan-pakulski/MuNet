#include "backend/debug_backend.hpp"
#include "backend/host_staging_runtime.hpp"
#include "core/backend.hpp"
#include "core/util.hpp"

#include <algorithm>
#include <mutex>

namespace munet {
namespace {

bool vulkan_probe_ok(Device device, BackendRegistry::BackendFactory factory,
                     std::string *detail = nullptr) {
  try {
    auto backend = factory(device);
    if (backend) {
      backend->synchronize();
    }
    if (detail) {
      *detail = "Vulkan runtime probe succeeded using built-in host staging.";
    }
    return true;
  } catch (const std::exception &e) {
    if (detail) {
      *detail = e.what();
    }
    MUNET_WARNING << "Vulkan runtime unavailable: " << e.what() << std::endl;
    return false;
  } catch (...) {
    if (detail) {
      *detail = "Vulkan runtime probe failure.";
    }
    MUNET_WARNING << "Vulkan runtime unavailable due to unknown Vulkan error."
                  << std::endl;
    return false;
  }
}

BackendRegistry::BackendFactory vulkan_factory() {
  return [](Device) { return std::make_shared<HostStagingRuntime>(); };
}

void register_default_runtime(BackendRegistry &registry) {
  static std::once_flag once;
  std::call_once(once, [&registry]() {
    registry.register_backend(DeviceType::VULKAN, vulkan_factory());
    registry.set_decorator([](std::shared_ptr<Backend> backend) {
      if (is_debug_enabled() || is_profile_enabled()) {
        return wrap_with_debug_backend(std::move(backend));
      }
      return backend;
    });
  });
}

int backend_cache_key(Device device) { return device.index; }

} // namespace

void BackendRegistry::register_backend(DeviceType type, BackendFactory factory) {
  std::lock_guard<std::mutex> lock(mutex_);
  factories_[type] = std::move(factory);
  cache_.clear();
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
    throw std::runtime_error("Vulkan runtime is not registered.");
  }

  auto backend = factory_it->second(device);
  if (decorator_) {
    backend = decorator_(std::move(backend));
  }

  cache_[key] = backend;
  return backend;
}

void BackendRegistry::clear_cache(DeviceType) {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
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
  register_default_runtime(registry);
  return registry;
}

void BackendManager::register_backend(DeviceType type, BackendFactory factory) {
  registry().register_backend(type, std::move(factory));
}

std::shared_ptr<Backend> BackendManager::get(Device device) {
  return registry().get(device);
}

std::vector<std::string> BackendManager::list_available_backends() {
  std::string detail;
  if (vulkan_probe_ok(Device{DeviceType::VULKAN, 0}, vulkan_factory(), &detail)) {
    return {"vulkan"};
  }
  return {};
}

std::vector<BackendRuntimeStatus> BackendManager::backend_status() {
  BackendRuntimeStatus status;
  status.name = "vulkan";
  status.source = "builtin";
  status.discovered = true;

  std::string detail;
  if (vulkan_probe_ok(Device{DeviceType::VULKAN, 0}, vulkan_factory(), &detail)) {
    status.loadable = true;
    status.active = true;
    status.reason_code = "ok";
    status.detail = detail;
  } else {
    status.reason_code = "runtime_dependency_missing";
    status.detail = detail;
  }

  return {std::move(status)};
}

} // namespace munet
