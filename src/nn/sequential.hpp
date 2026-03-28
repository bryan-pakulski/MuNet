#pragma once

#include "module.hpp"

namespace munet {
namespace nn {

class Sequential : public Module {
public:
  explicit Sequential(TensorOptions options = TensorOptions{})
      : Module(options) {}

  void add(std::shared_ptr<Module> m) {
    register_module(std::to_string(modules_.size()), m);
    ordered_modules_.push_back(m);
  }

  Tensor forward_impl(Tensor x) override {
    for (size_t i = 0; i < ordered_modules_.size(); ++i) {
      x = enforce_offload_boundary(std::to_string(i), std::move(x));
      auto &m = ordered_modules_[i];
      x = m->forward(x);
    }
    return x;
  }

  std::vector<std::shared_ptr<Module>> ordered_modules_;
};

} // namespace nn
} // namespace munet
