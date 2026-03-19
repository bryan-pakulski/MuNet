#pragma once

#include <chrono>

namespace munet {

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

} // namespace munet
