#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace munet {

#define MUNET_C_RESET "\033[0m"
#define MUNET_C_RED "\033[31m"
#define MUNET_C_ORANGE "\033[33m"
#define MUNET_C_GREEN "\033[32m"
#define MUNET_C_CYAN "\033[36m"

inline bool env_flag_enabled(const char *name) {
  return std::getenv(name) != nullptr;
}

inline bool is_debug_enabled() { return env_flag_enabled("MUNET_DEBUG"); }

inline bool is_profile_enabled() { return env_flag_enabled("MUNET_PROFILE"); }

inline int log_level() {
  const char *env = std::getenv("MUNET_LOG_LEVEL");
  if (env)
    return std::max(0, std::min(3, std::atoi(env)));
  return is_debug_enabled() ? 3 : 1;
}

inline std::ostream &log_stream(const char *label, const char *color,
                                int min_level) {
  if (log_level() < min_level) {
    static std::ostream null_stream(nullptr);
    return null_stream;
  }
  std::cerr << color << label << MUNET_C_RESET;
  return std::cerr;
}

#define MUNET_DEBUG munet::log_stream("[DEBUG] ", MUNET_C_CYAN, 3)
#define MUNET_INFO munet::log_stream("[INFO] ", MUNET_C_GREEN, 2)
#define MUNET_WARNING munet::log_stream("[WARN] ", MUNET_C_ORANGE, 1)
#define MUNET_ERROR munet::log_stream("[ERROR] ", MUNET_C_RED, 0)

#define MUNET_PROFILE_LOG                                                      \
  if (!munet::is_profile_enabled()) {                                          \
  } else                                                                       \
    std::cerr << MUNET_C_ORANGE "[PROFILE] " MUNET_C_RESET

} // namespace munet
