#include "backend/plugins/plugin_common.hpp"

extern "C" uint32_t munet_backend_plugin_abi_version(void) { return 1u; }
extern "C" const char *munet_backend_plugin_name(void) { return "vulkan"; }
extern "C" const char *munet_backend_plugin_device_type(void) { return "vulkan"; }
extern "C" uint64_t munet_backend_plugin_capability_flags(void) { return 0x1Fu; }
extern "C" const char *munet_backend_plugin_probe(void) {
  return munet::plugin::probe_required_library("libvulkan.so.1");
}
