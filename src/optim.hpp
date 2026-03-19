#pragma once
#include "ops.hpp"
#include "tensor.hpp"
#include "core/util.hpp"
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
      exp_avg_.push_back(munet::ops::zeros(p.shape(), p.device()));
      exp_avg_sq_.push_back(munet::ops::zeros(p.shape(), p.device()));
    }
  }

  void step() override {
    step_count_++;
    for (size_t i = 0; i < params_.size(); ++i) {
      auto &p = params_[i];
      if (!p.has_grad())
        continue;

      p.impl_->backend().adam_step(*p.impl_->storage, *p.grad().impl_->storage,
                                   *exp_avg_[i].impl_->storage,
                                   *exp_avg_sq_[i].impl_->storage, lr_, beta1_,
                                   beta2_, eps_, step_count_, p.size());
    }
  }

private:
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
