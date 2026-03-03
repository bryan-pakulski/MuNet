#pragma once

#include "tensor.hpp"
#include <unordered_map>

namespace munet {

class Optimizer {
public:
  Optimizer(const std::vector<Tensor *> &parameters)
      : parameters_(parameters) {}
  virtual ~Optimizer() = default;

  virtual void step() = 0;

  virtual void zero_grad() {
    for (auto *p : parameters_) {
      if (p->grad()) {
        p->grad()->zero();
      }
    }
  }

protected:
  std::vector<Tensor *> parameters_;
};

class SGD : public Optimizer {
public:
  SGD(const std::vector<Tensor *> &parameters, float lr)
      : Optimizer(parameters), lr_(lr) {}

  void step() override {
    for (auto *p : parameters_) {
      if (!p->grad())
        continue;

      float *w_ptr = static_cast<float *>(p->data());
      const float *g_ptr = static_cast<const float *>(p->grad()->data());
      int size = p->size();

#ifdef MUNET_USE_CUDA
      if (p->device_ == Device::CUDA) {
        cuda_kernels::sgd_step(w_ptr, g_ptr, lr_, size);
        continue;
      }
#endif

      for (int i = 0; i < size; ++i) {
        w_ptr[i] -= lr_ * g_ptr[i];
      }
    }
  }

private:
  float lr_;
};

class Adam : public Optimizer {
public:
  Adam(const std::vector<Tensor *> &parameters, float lr = 0.001f,
       float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
       float weight_decay = 0.0f)
      : Optimizer(parameters), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
        weight_decay_(weight_decay), step_(0) {

    // Initialize state buffers (m and v) for each parameter
    for (auto *p : parameters_) {
      // m and v are same shape as p, initialized to 0
      m_states_[p] =
          std::make_shared<Tensor>(p->shape(), p->device_, p->dtype_);
      v_states_[p] =
          std::make_shared<Tensor>(p->shape(), p->device_, p->dtype_);
      m_states_[p]->zero();
      v_states_[p]->zero();
    }
  }

  void step() override {
    step_++;
    for (auto *p : parameters_) {
      if (!p->grad())
        continue;

      auto m_tensor = m_states_[p];
      auto v_tensor = v_states_[p];

      float *w_ptr = static_cast<float *>(p->data());
      const float *g_ptr = static_cast<const float *>(p->grad()->data());
      float *m_ptr = static_cast<float *>(m_tensor->data());
      float *v_ptr = static_cast<float *>(v_tensor->data());
      int size = p->size();

#ifdef MUNET_USE_CUDA
      if (p->device_ == Device::CUDA) {
        cuda_kernels::adam_step(w_ptr, g_ptr, m_ptr, v_ptr, lr_, beta1_, beta2_,
                                eps_, weight_decay_, step_, size);
        continue;
      }
#endif
      // CPU Implementation
      for (int i = 0; i < size; ++i) {
        float grad = g_ptr[i];
        if (weight_decay_ > 0.0f)
          grad += w_ptr[i] * weight_decay_;

        m_ptr[i] = beta1_ * m_ptr[i] + (1.0f - beta1_) * grad;
        v_ptr[i] = beta2_ * v_ptr[i] + (1.0f - beta2_) * grad * grad;

        float m_hat = m_ptr[i] / (1.0f - std::pow(beta1_, step_));
        float v_hat = v_ptr[i] / (1.0f - std::pow(beta2_, step_));

        w_ptr[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
      }
    }
  }

private:
  float lr_, beta1_, beta2_, eps_, weight_decay_;
  int step_;
  // Map parameter pointer to its state tensors
  std::unordered_map<Tensor *, std::shared_ptr<Tensor>> m_states_;
  std::unordered_map<Tensor *, std::shared_ptr<Tensor>> v_states_;
};

} // namespace munet
