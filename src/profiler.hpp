#pragma once
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace munet {

struct ProfileEvent {
  std::string name;
  std::string cat;
  long long ts; // microseconds
  double dur;   // microseconds
};

class Profiler {
public:
  static Profiler &get() {
    static Profiler instance;
    return instance;
  }

  Profiler() {
#ifdef ENABLE_PROFILING
    std::cerr
        << "[Profiler] -D ENABLE_PROFILING flag is set. Profiling is enabled."
        << std::endl;
    events_.reserve(10000); // Pre-allocate to minimize overhead
#endif
  }

  void log(const std::string &name, const std::string &cat,
           double duration_us) {
#ifdef ENABLE_PROFILING
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    long long ts = std::chrono::duration_cast<std::chrono::microseconds>(
                       now.time_since_epoch())
                       .count();
    // Approximate start time by subtracting duration (since we log at the end)
    events_.push_back({name, cat, ts - (long long)duration_us, duration_us});
#endif
  }

  ~Profiler() {
#ifdef ENABLE_PROFILING
    if (events_.empty())
      return;

    std::string filename = "munet_profile.json";
    std::ofstream out(filename);
    out << "{\"traceEvents\":[\n";
    for (size_t i = 0; i < events_.size(); ++i) {
      const auto &e = events_[i];
      // tid: 1 for CPU, 2 for CUDA, 3 for Vulkan (just for visualization
      // grouping)
      int tid = (e.cat == "cpu") ? 1 : ((e.cat == "cuda") ? 2 : 3);

      out << "{\"name\":\"" << e.name << "\", \"cat\":\"" << e.cat
          << "\", \"ph\":\"X\", \"ts\":" << e.ts << ", \"dur\":" << e.dur
          << ", \"pid\":1, \"tid\": " << tid << "}";
      if (i < events_.size() - 1)
        out << ",\n";
    }
    out << "\n]}\n";
    out.close();
    std::cout << "[Profiler] Saved " << events_.size() << " events to "
              << filename << std::endl;
#endif
  }

private:
  std::vector<ProfileEvent> events_;
  std::mutex mutex_;
};

} // namespace munet
