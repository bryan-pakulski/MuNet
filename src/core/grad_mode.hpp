#pragma once

namespace munet {

class GradMode {
public:
  static bool is_enabled() { return enabled_; }
  static void set_enabled(bool enabled) { enabled_ = enabled; }

private:
  inline static thread_local bool enabled_ = true;
};

} // namespace munet
