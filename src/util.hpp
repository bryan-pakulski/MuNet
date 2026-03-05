#pragma once

#include <cstdlib>

namespace munet {

// Logging Macros
#define MUNET_C_RESET "\033[0m"
#define MUNET_C_RED "\033[31m"
#define MUNET_C_ORANGE "\033[33m"
#define MUNET_C_GREEN "\033[32m"
#define MUNET_C_CYAN "\033[36m"

inline bool is_debug_enabled() {
  static const bool debug = (std::getenv("MUNET_DEBUG") != nullptr);
  return debug;
}

#define MUNET_DEBUG                                                            \
  if (!munet::is_debug_enabled()) {                                            \
  } else                                                                       \
    std::cerr << MUNET_C_CYAN "[DEBUG] " MUNET_C_RESET
#define MUNET_INFO                                                             \
  if (!munet::is_debug_enabled()) {                                            \
  } else                                                                       \
    std::cerr << MUNET_C_GREEN "[INFO] " MUNET_C_RESET
#define MUNET_ERROR                                                            \
  if (!munet::is_debug_enabled()) {                                            \
  } else                                                                       \
    std::cerr << MUNET_C_RED "[INFO] " MUNET_C_RESET
#define MUNET_WARNING                                                          \
  if (!munet::is_debug_enabled()) {                                            \
  } else                                                                       \
    std::cerr << MUNET_C_ORANGE "[WARN] " MUNET_C_RESET

}; // namespace munet
