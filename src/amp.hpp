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
    Tensor s({1}, loss.device(), accumulation_dtype(loss.dtype()), false);
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

      for (size_t i = 0; i < g_cpu.size(); ++i) {
        double v = load_scalar_as_double(g_cpu.data(), g_cpu.dtype(), i);
        if (!std::isfinite(v)) {
          found_inf = true;
          break;
        }
      }
      if (found_inf)
        break;

      Tensor inv({1}, g.device(), accumulation_dtype(g.dtype()), false);
      inv.uniform_(1.0f / scale_, 1.0f / scale_);
      Tensor unscaled = g * inv;
      p.impl_->grad = unscaled.impl_;
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



class FP32MasterSGD {
public:
  FP32MasterSGD(std::vector<Tensor> params, float lr)
      : params_(std::move(params)), lr_(lr) {
    for (auto &p : params_) {
      master_params_.push_back(p.to_dtype(DataType::Float32));
    }
  }

  void step() {
    for (size_t i = 0; i < params_.size(); ++i) {
      auto &p = params_[i];
      if (!p.has_grad())
        continue;

      Tensor grad_fp32 = p.grad().to_dtype(DataType::Float32);
      Tensor lr_t({1}, master_params_[i].device(), DataType::Float32, false);
      lr_t.uniform_(lr_, lr_);

      Tensor updated = master_params_[i] - (grad_fp32 * lr_t);
      master_params_[i].impl_ = updated.impl_;

      Tensor model_copy = master_params_[i].to_dtype(p.dtype());
      p.impl_->backend().copy(model_copy.data(), p.data(), p.bytes(), p.device(),
                              p.device());
    }
  }

  void zero_grad() {
    for (auto &p : params_) {
      p.zero_grad();
    }
  }

private:
  std::vector<Tensor> params_;
  std::vector<Tensor> master_params_;
  float lr_;
};

class FP32MasterAdam {
public:
  FP32MasterAdam(std::vector<Tensor> params, float lr = 1e-3,
                 float beta1 = 0.9f, float beta2 = 0.999f,
                 float eps = 1e-8f)
      : params_(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
        eps_(eps) {
    for (auto &p : params_) {
      Tensor master = p.to_dtype(DataType::Float32);
      master_params_.push_back(master);

      Tensor m(p.shape(), p.device(), DataType::Float32, false);
      Tensor v(p.shape(), p.device(), DataType::Float32, false);
      m.impl_->storage->zero_();
      v.impl_->storage->zero_();
      exp_avg_.push_back(m);
      exp_avg_sq_.push_back(v);
    }
  }

  void step() {
    step_count_++;
    float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
    float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);

    for (size_t i = 0; i < params_.size(); ++i) {
      auto &p = params_[i];
      if (!p.has_grad())
        continue;

      Tensor grad_fp32 = p.grad().to_dtype(DataType::Float32);
      auto *g = static_cast<float *>(grad_fp32.data());
      auto *w = static_cast<float *>(master_params_[i].data());
      auto *m = static_cast<float *>(exp_avg_[i].data());
      auto *v = static_cast<float *>(exp_avg_sq_[i].data());

      for (size_t j = 0; j < p.size(); ++j) {
        float gj = g[j];
        m[j] = beta1_ * m[j] + (1.0f - beta1_) * gj;
        v[j] = beta2_ * v[j] + (1.0f - beta2_) * (gj * gj);

        float m_hat = m[j] / bias_correction1;
        float v_hat = v[j] / bias_correction2;
        w[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
      }

      Tensor model_copy = master_params_[i].to_dtype(p.dtype());
      p.impl_->backend().copy(model_copy.data(), p.data(), p.bytes(), p.device(),
                              p.device());
    }
  }

  void zero_grad() {
    for (auto &p : params_) {
      p.zero_grad();
    }
  }

private:
  std::vector<Tensor> params_;
  std::vector<Tensor> master_params_;
  std::vector<Tensor> exp_avg_;
  std::vector<Tensor> exp_avg_sq_;
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;
  int step_count_ = 0;
};

} // namespace amp
} // namespace munet
