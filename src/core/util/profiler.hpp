#pragma once

#include "logging.hpp"
#include <algorithm>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace munet {

struct OpStats {
  double cpu_us = 0;
  double gpu_us = 0;
  double min_cpu_us = 1e30;
  double min_gpu_us = 1e30;
  double max_cpu_us = 0;
  double max_gpu_us = 0;
  size_t bytes_processed = 0;
  int count = 0;
  std::string last_shape = "";
};

class Profiler {
  std::map<std::string, OpStats> stats;
  size_t peak_memory = 0;
  size_t current_memory = 0;
  mutable std::mutex mtx;
  bool printed_at_exit_ = false;

  Profiler() = default;

public:
  static Profiler &get() {
    static Profiler p;
    return p;
  }

  ~Profiler() {
    if (is_profile_enabled()) {
      std::lock_guard<std::mutex> lock(mtx);
      if (!printed_at_exit_ && !stats.empty()) {
        printed_at_exit_ = true;
        print_summary_locked("Auto Summary (process exit)");
      }
    }
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mtx);
    stats.clear();
    peak_memory = 0;
    current_memory = 0;
    MUNET_PROFILE_LOG << "Profiler stats and memory counters reset."
                      << std::endl;
  }

  void record_alloc(size_t bytes) {
    std::lock_guard<std::mutex> lock(mtx);
    current_memory += bytes;
    if (current_memory > peak_memory)
      peak_memory = current_memory;
  }

  void record_free(size_t bytes) {
    std::lock_guard<std::mutex> lock(mtx);
    current_memory = (bytes > current_memory) ? 0 : (current_memory - bytes);
  }

  void record(std::string name, double cpu_us, double gpu_us, size_t bytes = 0,
              std::string shape = "") {
    std::lock_guard<std::mutex> lock(mtx);
    auto &s = stats[name];
    s.cpu_us += cpu_us;
    s.gpu_us += gpu_us;
    s.min_cpu_us = std::min(s.min_cpu_us, cpu_us);
    s.min_gpu_us = std::min(s.min_gpu_us, gpu_us);
    s.max_cpu_us = std::max(s.max_cpu_us, cpu_us);
    s.max_gpu_us = std::max(s.max_gpu_us, gpu_us);
    s.bytes_processed += bytes;
    s.count++;
    if (!shape.empty())
      s.last_shape = shape;
  }

  void print_summary(const std::string &title = "MuNet Performance Summary") {
    std::lock_guard<std::mutex> lock(mtx);
    print_summary_locked(title);
    printed_at_exit_ = true;
  }

  size_t current_memory_bytes() const {
    std::lock_guard<std::mutex> lock(mtx);
    return current_memory;
  }

  size_t peak_memory_bytes() const {
    std::lock_guard<std::mutex> lock(mtx);
    return peak_memory;
  }

private:
  void print_summary_locked(const std::string &title) {
    std::cerr << MUNET_C_GREEN "\n--- MuNet Memory Report ---\n" MUNET_C_RESET;
    std::cerr << "Current Memory Usage: " << std::fixed << std::setprecision(2)
              << (current_memory / 1024.0 / 1024.0) << " MB\n";
    std::cerr << "Peak Memory Usage: " << std::fixed << std::setprecision(2)
              << (peak_memory / 1024.0 / 1024.0) << " MB\n";

    struct Row {
      std::string name;
      OpStats stats;
      double total_us = 0;
    };

    std::vector<Row> rows;
    rows.reserve(stats.size());
    double total_cpu = 0;
    double total_gpu = 0;

    for (const auto &entry : stats) {
      Row r;
      r.name = entry.first;
      r.stats = entry.second;
      r.total_us = entry.second.cpu_us + entry.second.gpu_us;
      total_cpu += entry.second.cpu_us;
      total_gpu += entry.second.gpu_us;
      rows.push_back(std::move(r));
    }

    std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) {
      return a.total_us > b.total_us;
    });

    std::cerr << MUNET_C_GREEN << "\n--- " << title << " ---\n" << MUNET_C_RESET;
    std::cerr << "Total CPU(us): " << std::fixed << std::setprecision(1)
              << total_cpu << " | Total GPU(us): " << total_gpu << "\n";
    std::cerr << std::left << std::setw(32) << "Op [Last Shape]" << std::setw(8)
              << "Count" << std::setw(12) << "AvgCPU" << std::setw(12)
              << "AvgGPU" << std::setw(12) << "MaxGPU" << std::setw(12)
              << "GB/s" << std::setw(10) << "%Total" << std::endl;

    const double grand_total = total_cpu + total_gpu;
    for (const auto &r : rows) {
      const auto &s = r.stats;
      double avg_gpu = (s.count > 0) ? s.gpu_us / s.count : 0;
      double avg_cpu = (s.count > 0) ? s.cpu_us / s.count : 0;
      double bandwidth =
          (r.total_us > 0) ? ((double)s.bytes_processed / (r.total_us * 1000.0))
                           : 0;
      double pct = (grand_total > 0) ? (100.0 * r.total_us / grand_total) : 0;

      std::string label =
          r.name + (s.last_shape.empty() ? "" : " " + s.last_shape);
      std::cerr << std::left << std::setw(32) << label.substr(0, 31)
                << std::setw(8) << s.count << std::setw(12) << std::fixed
                << std::setprecision(1) << avg_cpu << std::setw(12)
                << avg_gpu << std::setw(12) << s.max_gpu_us << std::setw(12)
                << std::setprecision(3) << bandwidth << std::setw(10)
                << std::setprecision(1) << pct << std::endl;
    }
  }
};

} // namespace munet
