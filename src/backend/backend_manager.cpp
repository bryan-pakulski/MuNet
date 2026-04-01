#include "backend/cpu_backend.hpp"
#include "backend/debug_backend.hpp"
#include "backend/plugin_loader.hpp"
#include "core/backend.hpp"
#include "core/util.hpp"

#ifdef MUNET_USE_CUDA
#include "backend/cuda_backend.hpp"
#endif

#ifdef MUNET_USE_VULKAN
#include "backend/vulkan_backend.hpp"
#endif

#include <algorithm>
#include <mutex>

namespace munet {
namespace {

bool backend_probe_ok(DeviceType type, const char *backend_name,
                      BackendRegistry::BackendFactory factory) {
  try {
    auto backend = factory(Device{type, 0});
    if (backend) {
      backend->synchronize();
    }
    return true;
  } catch (const std::exception &e) {
    MUNET_WARNING << backend_name
                  << " backend compiled but unavailable at runtime; disabling "
                     "acceleration for this backend. Reason: "
                  << e.what() << std::endl;
    return false;
  } catch (...) {
    MUNET_WARNING << backend_name
                  << " backend compiled but unavailable at runtime; disabling "
                     "acceleration for this backend due to unknown error."
                  << std::endl;
    return false;
  }
}

void register_default_backends(BackendRegistry &registry) {
  static std::once_flag once;
  std::call_once(once, [&registry]() {
    registry.register_backend(
        DeviceType::CPU, [](Device) { return std::make_shared<CPUBackend>(); });

#ifdef MUNET_USE_CUDA
    BackendRegistry::BackendFactory cuda_factory = [](Device device) {
      return std::make_shared<CUDABackend>(device.index);
    };
    if (backend_probe_ok(DeviceType::CUDA, "CUDA", cuda_factory)) {
      registry.register_backend(DeviceType::CUDA, std::move(cuda_factory));
    }
#endif

#ifdef MUNET_USE_VULKAN
    BackendRegistry::BackendFactory vulkan_factory = [](Device device) {
      return std::make_shared<VulkanBackend>(device.index);
    };
    if (backend_probe_ok(DeviceType::VULKAN, "Vulkan", vulkan_factory)) {
      registry.register_backend(DeviceType::VULKAN, std::move(vulkan_factory));
    }
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

void BackendRegistry::register_backend(DeviceType type,
                                       BackendFactory factory) {
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


std::vector<std::string> BackendManager::list_available_backends() {
  std::vector<std::string> out{"cpu"};

#ifdef MUNET_USE_CUDA
  if (backend_probe_ok(DeviceType::CUDA, "CUDA", [](Device device) {
        return std::make_shared<CUDABackend>(device.index);
      })) {
    out.push_back("cuda");
  }
#endif

#ifdef MUNET_USE_VULKAN
  if (backend_probe_ok(DeviceType::VULKAN, "Vulkan", [](Device device) {
        return std::make_shared<VulkanBackend>(device.index);
      })) {
    out.push_back("vulkan");
  }
#endif

  for (const auto &plugin_status : plugin::discover_backend_plugins()) {
    if (plugin_status.active) {
      out.push_back(plugin_status.name);
    }
  }

  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<BackendRuntimeStatus> BackendManager::backend_status() {
  std::vector<BackendRuntimeStatus> statuses;

  BackendRuntimeStatus cpu;
  cpu.name = "cpu";
  cpu.source = "builtin";
  cpu.discovered = true;
  cpu.loadable = true;
  cpu.active = true;
  cpu.reason_code = "ok";
  cpu.detail = "CPU backend is always available.";
  statuses.push_back(std::move(cpu));

  auto add_builtin_status = [&](const std::string &name, bool compiled,
                                DeviceType type,
                                BackendRegistry::BackendFactory factory) {
    BackendRuntimeStatus status;
    status.name = name;
    status.source = "builtin";
    status.discovered = compiled;
    if (!compiled) {
      status.reason_code = "not_compiled";
      status.detail = "Backend not compiled into this build.";
      statuses.push_back(std::move(status));
      return;
    }

    try {
      auto backend = factory(Device{type, 0});
      backend->synchronize();
      status.loadable = true;
      status.active = true;
      status.reason_code = "ok";
      status.detail = "Backend compiled and runtime probe succeeded.";
    } catch (const std::exception &e) {
      status.reason_code = "runtime_dependency_missing";
      status.detail = e.what();
    } catch (...) {
      status.reason_code = "runtime_dependency_missing";
      status.detail = "Unknown runtime probe failure.";
    }
    statuses.push_back(std::move(status));
  };

#ifdef MUNET_USE_CUDA
  add_builtin_status("cuda", true, DeviceType::CUDA,
                     [](Device device) {
                       return std::make_shared<CUDABackend>(device.index);
                     });
#else
  add_builtin_status("cuda", false, DeviceType::CUDA, nullptr);
#endif

#ifdef MUNET_USE_VULKAN
  add_builtin_status("vulkan", true, DeviceType::VULKAN,
                     [](Device device) {
                       return std::make_shared<VulkanBackend>(device.index);
                     });
#else
  add_builtin_status("vulkan", false, DeviceType::VULKAN, nullptr);
#endif

  for (const auto &plugin_status : plugin::discover_backend_plugins()) {
    BackendRuntimeStatus status;
    status.name = plugin_status.name;
    status.source = "plugin";
    status.discovered = plugin_status.discovered;
    status.loadable = plugin_status.loadable;
    status.active = plugin_status.active;
    status.reason_code = plugin_status.reason_code;
    status.detail = plugin_status.detail;
    status.plugin_path = plugin_status.path;
    status.plugin_abi_version = plugin_status.plugin_abi_version;
    status.core_abi_version = plugin::kBackendPluginAbiVersion;
    status.capability_flags = plugin_status.capability_flags;
    statuses.push_back(std::move(status));
  }

  return statuses;
}

} // namespace munet
