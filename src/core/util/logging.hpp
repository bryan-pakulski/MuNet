#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

struct TraceContextState {
  uint64_t trace_id = 0;
  std::vector<std::string> span_stack{};
};

inline TraceContextState &trace_context_state() {
  thread_local TraceContextState state;
  return state;
}

inline uint64_t next_trace_id() {
  static std::atomic<uint64_t> next_id{1};
  return next_id.fetch_add(1);
}

inline uint64_t current_trace_id() {
  return trace_context_state().trace_id;
}

inline std::string join_trace_spans(const std::vector<std::string> &spans,
                                    size_t limit = std::string::npos) {
  if (spans.empty()) {
    return "";
  }
  const size_t count = std::min(limit, spans.size());
  std::ostringstream joined;
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      joined << ".";
    }
    joined << spans[i];
  }
  return joined.str();
}

inline std::string current_trace_span() {
  return join_trace_spans(trace_context_state().span_stack);
}

inline std::string current_trace_parent_span() {
  const auto &state = trace_context_state();
  if (state.span_stack.size() <= 1) {
    return "";
  }
  return join_trace_spans(state.span_stack, state.span_stack.size() - 1);
}

inline std::string current_trace_context_string() {
  const uint64_t trace_id = current_trace_id();
  if (trace_id == 0) {
    return "";
  }

  std::ostringstream detail;
  detail << "trace_id=" << trace_id;
  const std::string span = current_trace_span();
  if (!span.empty()) {
    detail << " span=" << span;
  }
  const std::string parent = current_trace_parent_span();
  if (!parent.empty()) {
    detail << " parent=" << parent;
  }
  return detail.str();
}

inline std::string append_trace_context(std::string detail) {
  const std::string trace = current_trace_context_string();
  if (trace.empty()) {
    return detail;
  }
  if (detail.empty()) {
    return trace;
  }
  return detail + " " + trace;
}

class ScopedTraceContext {
public:
  explicit ScopedTraceContext(std::string span,
                              std::optional<uint64_t> trace_id = std::nullopt)
      : previous_id_(trace_context_state().trace_id),
        previous_spans_(trace_context_state().span_stack) {
    auto &state = trace_context_state();
    if (trace_id.has_value()) {
      state.trace_id = *trace_id;
      state.span_stack.clear();
      active_ = true;
    } else if (state.trace_id != 0) {
      active_ = true;
    } else {
      return;
    }

    if (!span.empty()) {
      state.span_stack.push_back(std::move(span));
    }
  }

  ~ScopedTraceContext() {
    if (!active_) {
      return;
    }
    auto &state = trace_context_state();
    state.trace_id = previous_id_;
    state.span_stack = previous_spans_;
  }

private:
  uint64_t previous_id_ = 0;
  std::vector<std::string> previous_spans_{};
  bool active_ = false;
};

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
  const std::string trace_prefix = current_trace_context_string();
  if (!trace_prefix.empty()) {
    std::cerr << "[" << trace_prefix << "] ";
  }
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
