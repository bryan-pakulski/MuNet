#pragma once
#include "tensor.hpp"
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
  }

protected:
  std::vector<Tensor> params_;
  float lr_;
};

class SGD : public Optimizer {
public:
  using Optimizer::Optimizer;

  void step() override {
    for (auto &p : params_) {
      // p.step() uses the backend update kernel (w = w - lr * g)
      p.step(lr_);
    }
  }
};

} // namespace optim
} // namespace munet
