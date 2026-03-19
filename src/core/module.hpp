#pragma once

#include "tensor.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace munet {
namespace core {

class Module {
public:
  virtual ~Module() = default;

  // Generic forward for single-tensor modules
  virtual Tensor forward(Tensor x) = 0;

  virtual std::vector<Tensor> parameters() {
    std::vector<Tensor> params;
    for (auto &[name, p] : parameters_) {
      params.push_back(*p);
    }
    for (auto &[name, m] : modules_) {
      auto sub_params = m->parameters();
      params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
  }

  virtual std::map<std::string, Tensor>
  named_parameters(std::string prefix = "") {
    std::map<std::string, Tensor> params;
    for (auto &[name, p] : parameters_)
      params[prefix + name] = *p;
    for (auto &[name, b] : buffers_)
      params[prefix + name] = *b;
    for (auto &[name, m] : modules_) {
      auto sub_params = m->named_parameters(prefix + name + ".");
      params.insert(sub_params.begin(), sub_params.end());
    }
    return params;
  }

  virtual std::map<std::string, std::shared_ptr<Module>>
  named_modules(std::string prefix = "") {
    std::map<std::string, std::shared_ptr<Module>> mods;
    for (auto &[name, m] : modules_) {
      mods[prefix + name] = m;
      auto sub = m->named_modules(prefix + name + ".");
      mods.insert(sub.begin(), sub.end());
    }
    return mods;
  }

  virtual void train(bool mode = true) {
    training_ = mode;
    for (auto &[name, m] : modules_) {
      m->train(mode);
    }
  }

  void eval() { train(false); }

  virtual void to(Device device) {
    for (auto &[name, p] : parameters_) {
      *p = p->to(device);
      if (p->requires_grad()) {
        p->set_requires_grad(true);
      }
    }
    for (auto &[name, b] : buffers_) {
      *b = b->to(device);
    }
    for (auto &[name, m] : modules_) {
      m->to(device);
    }
  }

  virtual void to(DataType dtype) {
    for (auto &[name, p] : parameters_) {
      *p = p->to(dtype);
      if (p->requires_grad()) {
        p->set_requires_grad(true);
      }
    }
    for (auto &[name, b] : buffers_) {
      *b = b->to(dtype);
    }
    for (auto &[name, m] : modules_) {
      m->to(dtype);
    }
  }

  virtual void to(const TensorOptions &options) {
    for (auto &[name, p] : parameters_) {
      TensorOptions parameter_options = options;
      parameter_options.requires_grad = true;
      *p = p->to(parameter_options);
      p->set_requires_grad(true);
    }
    for (auto &[name, b] : buffers_) {
      TensorOptions buffer_options = options;
      buffer_options.requires_grad = false;
      *b = b->to(buffer_options);
      b->set_requires_grad(false);
    }
    for (auto &[name, m] : modules_) {
      m->to(options);
    }
  }

  void zero_grad() {
    for (auto &p : parameters()) {
      p.zero_grad();
    }
  }

  Tensor &register_parameter(std::string name, Tensor &t) {
    if (t.name().empty())
      t.set_name(name);
    parameters_[name] = &t;
    return t;
  }

  Tensor &register_buffer(std::string name, Tensor &t) {
    if (t.name().empty())
      t.set_name(name);
    buffers_[name] = &t;
    return t;
  }

  std::shared_ptr<Module> register_module(std::string name,
                                          std::shared_ptr<Module> m) {
    modules_[name] = m;
    return m;
  }

protected:
  bool training_ = true;
  std::map<std::string, Tensor *> parameters_;
  std::map<std::string, Tensor *> buffers_;
  std::map<std::string, std::shared_ptr<Module>> modules_;
};

} // namespace core
} // namespace munet
