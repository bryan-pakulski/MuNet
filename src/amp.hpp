#pragma once

#include "optim.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include "types.hpp"
#include <cmath>
#include <vector>

namespace munet {
namespace amp {

class AutocastMode {
public:
  static bool is_enabled() { return enabled_; }
  static void set_enabled(bool enabled) { enabled_ = enabled; }

  static DataType dtype() { return dtype_; }
  static void set_dtype(DataType dt) { dtype_ = dt; }

private:
  inline static thread_local bool enabled_ = false;
  inline static thread_local DataType dtype_ = DataType::Float16;
};

class AutoCastGuard {
public:
  explicit AutoCastGuard(DataType dtype = DataType::Float16)
      : prev_enabled_(AutocastMode::is_enabled()),
        prev_dtype_(AutocastMode::dtype()) {
    AutocastMode::set_enabled(true);
    AutocastMode::set_dtype(dtype);
  }

  ~AutoCastGuard() {
    AutocastMode::set_enabled(prev_enabled_);
    AutocastMode::set_dtype(prev_dtype_);
  }

private:
  bool prev_enabled_;
  DataType prev_dtype_;
};

class GradScaler {
public:
  explicit GradScaler(float init_scale = 65536.0f, float growth_factor = 2.0f,
                      float backoff_factor = 0.5f, int growth_interval = 2000)
      : scale_(init_scale), growth_factor_(growth_factor),
        backoff_factor_(backoff_factor), growth_interval_(growth_interval) {
    if (init_scale <= 0.0f)
      throw std::runtime_error("GradScaler init_scale must be > 0");
  }

  Tensor scale(const Tensor &loss) const {
    Tensor s({1}, loss.device(), loss.dtype(), false);
    s.uniform_(scale_, scale_);
    return loss * s;
  }

  bool unscale_(const std::vector<Tensor> &params) const {
    bool found_inf = false;

    for (const auto &p : params) {
      if (!p.has_grad())
        continue;

      Tensor g = p.grad();
      Tensor g_cpu = (g.device().type == DeviceType::CPU)
                         ? g
                         : g.to(Device{DeviceType::CPU, 0});

      // Check finite before unscaling
      for (size_t i = 0; i < g_cpu.size(); ++i) {
        float v = static_cast<float *>(g_cpu.data())[i];
        if (!std::isfinite(v)) {
          found_inf = true;
          break;
        }
      }
      if (found_inf)
        break;

      Tensor inv({1}, g.device(), g.dtype(), false);
      inv.uniform_(1.0f / scale_, 1.0f / scale_);
      Tensor unscaled = g * inv;

      // Replace grad storage with unscaled grad
      g.impl_ = unscaled.impl_;
    }

    return found_inf;
  }

  bool step(optim::Optimizer &optimizer, const std::vector<Tensor> &params) {
    bool found_inf = unscale_(params);
    if (!found_inf) {
      optimizer.step();
      growth_tracker_++;
    } else {
      growth_tracker_ = 0;
    }
    update(found_inf);
    return !found_inf;
  }

  void update(bool found_inf) {
    if (found_inf) {
      scale_ = std::max(1.0f, scale_ * backoff_factor_);
      growth_tracker_ = 0;
      return;
    }

    if (growth_tracker_ >= growth_interval_) {
      scale_ *= growth_factor_;
      growth_tracker_ = 0;
    }
  }

  float current_scale() const { return scale_; }

private:
  float scale_;
  float growth_factor_;
  float backoff_factor_;
  int growth_interval_;
  int growth_tracker_ = 0;
};

} // namespace amp
} // namespace munet
