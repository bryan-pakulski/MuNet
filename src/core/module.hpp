#pragma once

#include "tensor.hpp"
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace munet {
namespace core {

struct BufferRegistration {
  std::optional<DataType> fixed_dtype;
  std::optional<AccumulationOp> accumulation_op;

  static BufferRegistration fixed(DataType dtype) {
    BufferRegistration registration;
    registration.fixed_dtype = dtype;
    return registration;
  }

  static BufferRegistration accumulation(AccumulationOp op) {
    BufferRegistration registration;
    registration.accumulation_op = op;
    return registration;
  }
};

class Module {
public:
  explicit Module(TensorOptions default_options = TensorOptions{})
      : default_options_(default_options) {}
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

  const TensorOptions &default_options() const { return default_options_; }

  virtual void to(Device device) {
    default_options_.device = device;
    for (auto &[name, p] : parameters_) {
      *p = p->to(device);
      if (p->requires_grad()) {
        p->set_requires_grad(true);
      }
    }
    for (auto &[name, b] : buffers_) {
      TensorOptions buffer_options = b->options();
      buffer_options.device = device;
      buffer_options.dtype = resolve_buffer_dtype(name, default_options_.dtype);
      buffer_options.requires_grad = false;
      *b = b->to(buffer_options);
    }
    for (auto &[name, m] : modules_) {
      m->to(device);
    }
  }

  virtual void to(DataType dtype) {
    default_options_.dtype = dtype;
    for (auto &[name, p] : parameters_) {
      *p = p->to(dtype);
      if (p->requires_grad()) {
        p->set_requires_grad(true);
      }
    }
    for (auto &[name, b] : buffers_) {
      TensorOptions buffer_options = b->options();
      buffer_options.dtype = resolve_buffer_dtype(name, dtype);
      buffer_options.requires_grad = false;
      *b = b->to(buffer_options);
    }
    for (auto &[name, m] : modules_) {
      m->to(dtype);
    }
  }

  virtual void to(const TensorOptions &options) {
    default_options_ = options;
    for (auto &[name, p] : parameters_) {
      TensorOptions parameter_options = options;
      parameter_options.requires_grad = true;
      *p = p->to(parameter_options);
      p->set_requires_grad(true);
    }
    for (auto &[name, b] : buffers_) {
      TensorOptions buffer_options = options;
      buffer_options.dtype = resolve_buffer_dtype(name, options.dtype);
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

  Tensor &register_buffer(std::string name, Tensor &t,
                          BufferRegistration registration = {}) {
    if (t.name().empty())
      t.set_name(name);
    buffers_[name] = &t;
    buffer_registrations_[name] = registration;
    return t;
  }

  std::shared_ptr<Module> register_module(std::string name,
                                          std::shared_ptr<Module> m) {
    if (m) {
      const TensorOptions unresolved_defaults;
      const TensorOptions &child_defaults = m->default_options();
      if (child_defaults.device == unresolved_defaults.device &&
          child_defaults.dtype == unresolved_defaults.dtype &&
          child_defaults.requires_grad == unresolved_defaults.requires_grad) {
        m->to(default_options_);
      }
    }
    modules_[name] = m;
    return m;
  }

protected:
  DataType resolve_buffer_dtype(const std::string &name,
                                DataType requested_dtype) const {
    auto it = buffer_registrations_.find(name);
    if (it == buffer_registrations_.end()) {
      return requested_dtype;
    }

    const BufferRegistration &registration = it->second;
    if (registration.fixed_dtype.has_value()) {
      return registration.fixed_dtype.value();
    }
    if (registration.accumulation_op.has_value()) {
      return accumulation_type(registration.accumulation_op.value(),
                               requested_dtype);
    }
    return requested_dtype;
  }

  bool training_ = true;
  TensorOptions default_options_;
  std::map<std::string, Tensor *> parameters_;
  std::map<std::string, Tensor *> buffers_;
  std::map<std::string, BufferRegistration> buffer_registrations_;
  std::map<std::string, std::shared_ptr<Module>> modules_;
};

} // namespace core
} // namespace munet
