#pragma once

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

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

inline bool is_profile_enabled() {
  static const bool profile = (std::getenv("MUNET_PROFILE") != nullptr);
  return profile;
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

#define MUNET_PROFILE_LOG                                                      \
  if (!munet::is_profile_enabled()) {                                          \
  } else                                                                       \
    std::cerr << MUNET_C_ORANGE "[PROFILE] " MUNET_C_RESET

struct OpStats {
  double cpu_us = 0;
  double gpu_us = 0;
  size_t bytes_processed = 0;
  int count = 0;
  std::string last_shape = ""; // Track the most recent shape seen
};

class Profiler {
  std::map<std::string, OpStats> stats;
  size_t peak_memory = 0;
  size_t current_memory = 0;
  std::mutex mtx;

public:
  static Profiler &get() {
    static Profiler p;
    return p;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mtx);
    stats.clear();
    peak_memory = 0;
    current_memory = 0;
    MUNET_PROFILE_LOG << "Profiler stats and memory counters reset."
                      << std::endl;
  }

  // Track memory allocations
  void record_alloc(size_t bytes) {
    std::lock_guard<std::mutex> lock(mtx);
    current_memory += bytes;
    if (current_memory > peak_memory)
      peak_memory = current_memory;
  }
  void record_free(size_t bytes) {
    std::lock_guard<std::mutex> lock(mtx);
    current_memory -= bytes;
  }

  // Include shape string in record
  void record(std::string name, double cpu_us, double gpu_us, size_t bytes = 0,
              std::string shape = "") {
    std::lock_guard<std::mutex> lock(mtx);
    auto &s = stats[name];
    s.cpu_us += cpu_us;
    s.gpu_us += gpu_us;
    s.bytes_processed += bytes;
    s.count++;
    if (!shape.empty())
      s.last_shape = shape;
  }

  void print_summary() {
    std::cerr << MUNET_C_GREEN "\n--- MuNet Memory Report ---\n" MUNET_C_RESET;
    std::cerr << "Peak Memory Usage: " << std::fixed << std::setprecision(2)
              << (peak_memory / 1024.0 / 1024.0) << " MB\n";

    std::cerr << MUNET_C_GREEN
        "\n--- MuNet Performance Summary ---\n" MUNET_C_RESET;
    std::cerr << std::left << std::setw(40) << "Op [Last Shape]" << std::setw(8)
              << "Count" << std::setw(12) << "Kernel(us)" << std::setw(12)
              << "Avg CPU(us)" << std::setw(12) << "GB/s (Write)" << std::endl;

    for (auto const &stat_pair : stats) {
      auto const &[name, s] = stat_pair;
      double avg_gpu = s.gpu_us / s.count;
      double avg_cpu = s.cpu_us / s.count;

      // Total time across all calls in microseconds
      double total_time_us = s.gpu_us + s.cpu_us;

      // GB/s = (Total Bytes / 1e9) / (Total Time us / 1e6)
      // Simplified: GB/s = Total Bytes / (Total Time us * 1000)
      double bandwidth =
          (total_time_us > 0)
              ? ((double)s.bytes_processed / (total_time_us * 1000.0))
              : 0;

      std::string label =
          name + (s.last_shape.empty() ? "" : " " + s.last_shape);
      std::cerr << std::left << std::setw(40) << label.substr(0, 39)
                << std::setw(8) << s.count << std::setw(12) << std::fixed
                << std::setprecision(1) << avg_gpu << std::setw(12)
                << std::fixed << std::setprecision(1) << avg_cpu
                << std::setw(12) << std::fixed << std::setprecision(3)
                << bandwidth << std::endl;
    }
  }
};

struct Timer {
  std::chrono::high_resolution_clock::time_point start;
  Timer() { start = std::chrono::high_resolution_clock::now(); }

  double elapsed_ms() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  double elapsed_us() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
  }
};

}; // namespace munet
