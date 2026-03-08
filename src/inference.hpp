#pragma once

#include "core/module.hpp"

namespace munet {
namespace inference {

// Inference modules share the same model graph core, but always stay in eval
// mode to avoid training-time behavior/state updates.
class Module : public core::Module {
public:
  virtual ~Module() = default;

  void train(bool mode = true) override {
    (void)mode;
    core::Module::train(false);
  }
};

class Sequential : public Module {
public:
  Sequential() = default;

  void add(std::shared_ptr<Module> m) {
    register_module(std::to_string(modules_.size()), m);
    ordered_modules_.push_back(m);
  }

  Tensor forward(Tensor x) override {
    for (auto &m : ordered_modules_) {
      x = m->forward(x);
    }
    return x;
  }

  std::vector<std::shared_ptr<Module>> ordered_modules_;
};

} // namespace inference
} // namespace munet
