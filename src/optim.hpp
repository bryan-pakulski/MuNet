#pragma once
#include "ops.hpp"
#include "tensor.hpp"
#include "core/util.hpp"
#include <cmath>
#include <cstring>
#include <vector>

namespace munet {
namespace optim {

class Optimizer {
public:
  Optimizer(std::vector<Tensor> params, float lr) : params_(params), lr_(lr) {}
  virtual ~Optimizer() = default;

  virtual void step() = 0;

  void zero_grad() {
    for (auto &p : params_) {
      p.zero_grad();
    }

    // Avoid unconditional GPU queue stalls in normal/profile mode.
    // In-order execution on a single backend queue is sufficient.
    // Keep explicit synchronization only in debug mode.
    if (!params_.empty() && is_debug_enabled()) {
      params_[0].impl_->backend().synchronize();
    }
  }

protected:
  std::vector<Tensor> params_;
  float lr_;
};

class Adam : public Optimizer {
public:
  Adam(std::vector<Tensor> params, float lr = 1e-3, float beta1 = 0.9,
       float beta2 = 0.999, float eps = 1e-8)
      : Optimizer(params, lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    for (auto &p : params_) {
      exp_avg_.push_back(
          munet::ops::zeros(p.shape(), p.device(), false,
                            optimizer_state_type(p.dtype())));
      exp_avg_sq_.push_back(
          munet::ops::zeros(p.shape(), p.device(), false,
                            optimizer_state_type(p.dtype())));
    }
  }

  void step() override {
    step_count_++;
    for (size_t i = 0; i < params_.size(); ++i) {
      auto &p = params_[i];
      if (!p.has_grad())
        continue;

      if (p.dtype() == DataType::Float32 &&
          exp_avg_[i].dtype() == DataType::Float32 &&
          exp_avg_sq_[i].dtype() == DataType::Float32) {
        p.impl_->backend().adam_step(*p.impl_->storage, *p.grad().impl_->storage,
                                     *exp_avg_[i].impl_->storage,
                                     *exp_avg_sq_[i].impl_->storage, lr_,
                                     beta1_, beta2_, eps_, step_count_,
                                     p.size());
        continue;
      }

      apply_typed_adam_step(p, p.grad(), exp_avg_[i], exp_avg_sq_[i]);
    }
  }

private:
  void apply_typed_adam_step(Tensor &param, const Tensor &grad, Tensor &exp_avg,
                             Tensor &exp_avg_sq) {
    if (!is_floating(param.dtype())) {
      throw std::runtime_error("Adam only supports floating-point parameters");
    }

    Device cpu{DeviceType::CPU, 0};
    Tensor param_cpu = (param.device().type == DeviceType::CPU) ? param
                                                                : param.to(cpu);
    Tensor grad_cpu = (grad.device().type == DeviceType::CPU) ? grad
                                                              : grad.to(cpu);
    Tensor exp_avg_cpu =
        (exp_avg.device().type == DeviceType::CPU) ? exp_avg : exp_avg.to(cpu);
    Tensor exp_avg_sq_cpu = (exp_avg_sq.device().type == DeviceType::CPU)
                                ? exp_avg_sq
                                : exp_avg_sq.to(cpu);

    char *param_bytes = static_cast<char *>(param_cpu.data());
    const char *grad_bytes = static_cast<const char *>(grad_cpu.data());
    char *exp_avg_bytes = static_cast<char *>(exp_avg_cpu.data());
    char *exp_avg_sq_bytes = static_cast<char *>(exp_avg_sq_cpu.data());

    const size_t param_stride = dtype_size(param_cpu.dtype());
    const size_t grad_stride = dtype_size(grad_cpu.dtype());
    const size_t exp_avg_stride = dtype_size(exp_avg_cpu.dtype());
    const size_t exp_avg_sq_stride = dtype_size(exp_avg_sq_cpu.dtype());
    const float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
    const float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);

    for (size_t j = 0; j < param.size(); ++j) {
      const float p_val =
          read_scalar_from_buffer(param_bytes + j * param_stride, param_cpu.dtype())
              .as_float();
      const float g_val =
          read_scalar_from_buffer(grad_bytes + j * grad_stride, grad_cpu.dtype())
              .as_float();
      float m_val =
          read_scalar_from_buffer(exp_avg_bytes + j * exp_avg_stride,
                                  exp_avg_cpu.dtype())
              .as_float();
      float v_val =
          read_scalar_from_buffer(exp_avg_sq_bytes + j * exp_avg_sq_stride,
                                  exp_avg_sq_cpu.dtype())
              .as_float();

      m_val = beta1_ * m_val + (1.0f - beta1_) * g_val;
      v_val = beta2_ * v_val + (1.0f - beta2_) * g_val * g_val;

      const float m_hat = m_val / bias_correction1;
      const float v_hat = v_val / bias_correction2;
      const float updated =
          p_val - lr_ * m_hat / (std::sqrt(v_hat) + eps_);

      write_scalar_to_buffer(param_bytes + j * param_stride, param_cpu.dtype(),
                             updated);
      write_scalar_to_buffer(exp_avg_bytes + j * exp_avg_stride,
                             exp_avg_cpu.dtype(), m_val);
      write_scalar_to_buffer(exp_avg_sq_bytes + j * exp_avg_sq_stride,
                             exp_avg_sq_cpu.dtype(), v_val);
    }

    if (param.device().type == DeviceType::CPU) {
      std::memcpy(param.data(), param_cpu.data(), param.bytes());
    } else {
      BackendManager::get(param.device())
          ->copy(param_cpu.data(), param.data(), param.bytes(), cpu,
                 param.device());
    }

    if (exp_avg.device().type == DeviceType::CPU) {
      std::memcpy(exp_avg.data(), exp_avg_cpu.data(), exp_avg.bytes());
    } else {
      BackendManager::get(exp_avg.device())
          ->copy(exp_avg_cpu.data(), exp_avg.data(), exp_avg.bytes(), cpu,
                 exp_avg.device());
    }

    if (exp_avg_sq.device().type == DeviceType::CPU) {
      std::memcpy(exp_avg_sq.data(), exp_avg_sq_cpu.data(), exp_avg_sq.bytes());
    } else {
      BackendManager::get(exp_avg_sq.device())
          ->copy(exp_avg_sq_cpu.data(), exp_avg_sq.data(), exp_avg_sq.bytes(),
                 cpu, exp_avg_sq.device());
    }
  }

  float beta1_, beta2_, eps_;
  int step_count_ = 0;
  std::vector<Tensor> exp_avg_;
  std::vector<Tensor> exp_avg_sq_;
};

class SGD : public Optimizer {
public:
  using Optimizer::Optimizer;

  void step() override {
    for (auto &p : params_) {
      // p.step() uses the backend update kernel (w = w - lr * g)
      p.step(lr_);
    }
    // Avoid unconditional GPU queue stalls in normal/profile mode.
    // Keep explicit synchronization only in debug mode.
    if (!params_.empty() && is_debug_enabled()) {
      params_[0].impl_->backend().synchronize();
    }
  }
};

} // namespace optim
} // namespace munet
