#include "backend/plugin_loader.hpp"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace munet::plugin {
namespace {

using AbiVersionFn = uint32_t (*)();
using NameFn = const char *(*)();
using DeviceFn = const char *(*)();
using CapabilityFlagsFn = uint64_t (*)();
using ProbeFn = const char *(*)();


std::filesystem::path core_module_dir() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(&discover_backend_plugins), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  return std::filesystem::path(info.dli_fname).parent_path();
}

std::vector<std::filesystem::path> plugin_search_paths() {
  std::vector<std::filesystem::path> paths;

  if (const char *env = std::getenv("MUNET_BACKEND_PLUGIN_PATH")) {
    std::string raw(env);
    size_t start = 0;
    while (start <= raw.size()) {
      size_t end = raw.find(':', start);
      std::string token = raw.substr(start, end - start);
      if (!token.empty()) {
        paths.emplace_back(token);
      }
      if (end == std::string::npos) {
        break;
      }
      start = end + 1;
    }
  }

  if (const char *env_dir = std::getenv("MUNET_BACKEND_PLUGIN_DIR")) {
    if (std::strlen(env_dir) > 0) {
      paths.emplace_back(env_dir);
    }
  }

  const auto module_dir = core_module_dir();
  if (!module_dir.empty()) {
    paths.emplace_back(module_dir);
    paths.emplace_back(module_dir / "plugins");
  }

  paths.emplace_back(std::filesystem::current_path());
  paths.emplace_back(std::filesystem::current_path() / "plugins");
  paths.emplace_back("/usr/local/lib/munet");

  return paths;
}

std::vector<std::filesystem::path> discover_candidate_plugin_files() {
  std::vector<std::filesystem::path> files;
  for (const auto &dir : plugin_search_paths()) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) {
      continue;
    }

    for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
      if (ec || !entry.is_regular_file()) {
        continue;
      }
      const auto filename = entry.path().filename().string();
      if (filename.rfind("libmunet_backend_", 0) == 0 &&
          entry.path().extension() == ".so") {
        files.push_back(entry.path());
      }
    }
  }
  return files;
}

BackendPluginStatus default_status(const std::string &name,
                                   const std::string &device) {
  BackendPluginStatus status;
  status.name = name;
  status.device = device;
  status.reason_code = "plugin_not_found";
  status.detail = "No plugin binary discovered.";
  return status;
}

} // namespace

std::vector<BackendPluginStatus> discover_backend_plugins() {
  std::unordered_map<std::string, BackendPluginStatus> by_name;
  by_name.emplace("cuda", default_status("cuda", "cuda"));
  by_name.emplace("vulkan", default_status("vulkan", "vulkan"));

  for (const auto &path : discover_candidate_plugin_files()) {
    BackendPluginStatus status;
    status.discovered = true;
    status.path = path.string();
    status.reason_code = "invalid_plugin_binary";
    status.detail = "Missing required plugin symbols.";

    void *handle = dlopen(status.path.c_str(), RTLD_LOCAL | RTLD_NOW);
    if (!handle) {
      status.reason_code = "plugin_dlopen_failed";
      status.detail = dlerror() ? dlerror() : "dlopen failed";
      status.name = path.stem().string();
      by_name[status.name] = status;
      continue;
    }

    auto abi_fn = reinterpret_cast<AbiVersionFn>(
        dlsym(handle, "munet_backend_plugin_abi_version"));
    auto name_fn =
        reinterpret_cast<NameFn>(dlsym(handle, "munet_backend_plugin_name"));
    auto device_fn = reinterpret_cast<DeviceFn>(
        dlsym(handle, "munet_backend_plugin_device_type"));
    auto probe_fn =
        reinterpret_cast<ProbeFn>(dlsym(handle, "munet_backend_plugin_probe"));
    auto flags_fn = reinterpret_cast<CapabilityFlagsFn>(
        dlsym(handle, "munet_backend_plugin_capability_flags"));

    if (!abi_fn || !name_fn || !device_fn || !probe_fn) {
      dlclose(handle);
      status.name = path.stem().string();
      by_name[status.name] = status;
      continue;
    }

    status.name = name_fn() ? name_fn() : path.stem().string();
    status.device = device_fn() ? device_fn() : "unknown";
    status.plugin_abi_version = abi_fn();
    status.capability_flags = flags_fn ? flags_fn() : 0;

    if (status.plugin_abi_version != kBackendPluginAbiVersion) {
      status.reason_code = "abi_mismatch";
      status.detail = "Plugin ABI version mismatch.";
      dlclose(handle);
      by_name[status.name] = status;
      continue;
    }

    const char *probe_error = probe_fn();
    if (probe_error != nullptr) {
      status.reason_code = "runtime_dependency_missing";
      status.detail = probe_error;
      dlclose(handle);
      by_name[status.name] = status;
      continue;
    }

    status.loadable = true;
    status.active = true;
    status.reason_code = "ok";
    status.detail = "Plugin loaded and probe succeeded.";
    dlclose(handle);
    by_name[status.name] = status;
  }

  std::vector<BackendPluginStatus> out;
  out.reserve(by_name.size());
  for (auto &kv : by_name) {
    out.push_back(std::move(kv.second));
  }
  return out;
}

bool has_active_plugin_for_device(const std::string &device) {
  for (const auto &status : discover_backend_plugins()) {
    if (status.active && status.device == device) {
      return true;
    }
  }
  return false;
}

} // namespace munet::plugin
