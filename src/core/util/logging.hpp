#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <optional>

namespace munet {

#define MUNET_C_RESET "\033[0m"
#define MUNET_C_RED "\033[31m"
#define MUNET_C_ORANGE "\033[33m"
#define MUNET_C_GREEN "\033[32m"
#define MUNET_C_CYAN "\033[36m"

inline bool env_flag_enabled(const char *name) {
  return std::getenv(name) != nullptr;
}

constexpr int kRuntimeFlagUseEnv = -1;
constexpr int kRuntimeFlagDisabled = 0;
constexpr int kRuntimeFlagEnabled = 1;

inline std::atomic<int> &debug_enabled_override() {
  static std::atomic<int> value{kRuntimeFlagUseEnv};
  return value;
}

inline std::atomic<int> &profile_enabled_override() {
  static std::atomic<int> value{kRuntimeFlagUseEnv};
  return value;
}

inline std::atomic<int> &log_level_override() {
  static std::atomic<int> value{kRuntimeFlagUseEnv};
  return value;
}

inline bool cached_env_debug_enabled() {
  static const bool enabled = env_flag_enabled("MUNET_DEBUG");
  return enabled;
}

inline bool cached_env_profile_enabled() {
  static const bool enabled = env_flag_enabled("MUNET_PROFILE");
  return enabled;
}

inline std::optional<int> cached_env_log_level() {
  static const std::optional<int> level = []() -> std::optional<int> {
    const char *env = std::getenv("MUNET_LOG_LEVEL");
    if (!env)
      return std::nullopt;
    return std::max(0, std::min(3, std::atoi(env)));
  }();
  return level;
}

inline void set_debug_enabled_override(std::optional<bool> enabled) {
  debug_enabled_override().store(
      enabled.has_value()
          ? (*enabled ? kRuntimeFlagEnabled : kRuntimeFlagDisabled)
          : kRuntimeFlagUseEnv);
}

inline void set_profile_enabled_override(std::optional<bool> enabled) {
  profile_enabled_override().store(
      enabled.has_value()
          ? (*enabled ? kRuntimeFlagEnabled : kRuntimeFlagDisabled)
          : kRuntimeFlagUseEnv);
}

inline void set_log_level_override(std::optional<int> level) {
  log_level_override().store(level.has_value()
                                 ? std::max(0, std::min(3, *level))
                                 : kRuntimeFlagUseEnv);
}

inline bool is_debug_enabled() {
  const int override_value = debug_enabled_override().load();
  if (override_value != kRuntimeFlagUseEnv)
    return override_value == kRuntimeFlagEnabled;
  return cached_env_debug_enabled();
}

inline bool is_profile_enabled() {
  const int override_value = profile_enabled_override().load();
  if (override_value != kRuntimeFlagUseEnv)
    return override_value == kRuntimeFlagEnabled;
  return cached_env_profile_enabled();
}

inline int log_level() {
  const int override_value = log_level_override().load();
  if (override_value != kRuntimeFlagUseEnv)
    return override_value;
  if (const auto env_level = cached_env_log_level())
    return *env_level;
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
