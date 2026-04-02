#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace munet::plugin {

constexpr uint32_t kBackendPluginAbiVersion = 1;

struct BackendPluginStatus {
  std::string name;
  std::string path;
  std::string device;
  bool discovered = false;
  bool loadable = false;
  bool active = false;
  uint32_t plugin_abi_version = 0;
  uint64_t capability_flags = 0;
  std::string reason_code;
  std::string detail;
};

std::vector<BackendPluginStatus> discover_backend_plugins();
bool has_active_plugin_for_device(const std::string &device);
std::vector<BackendPluginStatus> plugins_for_device(const std::string &device);

} // namespace munet::plugin
