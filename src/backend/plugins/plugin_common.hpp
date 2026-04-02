#pragma once

#include <dlfcn.h>
#include <stdint.h>

namespace munet::plugin {

inline const char *probe_required_library(const char *library_name) {
  void *handle = dlopen(library_name, RTLD_LOCAL | RTLD_NOW);
  if (!handle) {
    return dlerror();
  }
  dlclose(handle);
  return nullptr;
}

} // namespace munet::plugin
