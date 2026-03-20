#pragma once
#include "training_guard.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include "core/util.hpp"
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace munet {
namespace optim {

enum class MasterWeightDTypePolicy { None, MatchParameter, Float32 };
enum class OptimizerStateTensorDTypePolicy {
  MatchParameter,
  MatchModel,
  OptimizerStateType,
  Float32,
};

struct OptimizerStatePolicy {
  std::optional<DataType> model_dtype;
  MasterWeightDTypePolicy master_weight_dtype =
      MasterWeightDTypePolicy::None;
  OptimizerStateTensorDTypePolicy state_tensor_dtype =
      OptimizerStateTensorDTypePolicy::OptimizerStateType;
};

inline DataType resolve_model_dtype(DataType parameter_dtype,
                                    const OptimizerStatePolicy &policy) {
  return policy.model_dtype.value_or(parameter_dtype);
}

inline std::optional<DataType>
resolve_master_weight_dtype(DataType parameter_dtype,
                            const OptimizerStatePolicy &policy) {
  switch (policy.master_weight_dtype) {
  case MasterWeightDTypePolicy::None:
    return std::nullopt;
  case MasterWeightDTypePolicy::MatchParameter:
    return parameter_dtype;
  case MasterWeightDTypePolicy::Float32:
    return DataType::Float32;
  default:
    return std::nullopt;
  }
}

inline DataType resolve_optimizer_state_dtype(
    DataType parameter_dtype, const OptimizerStatePolicy &policy) {
  switch (policy.state_tensor_dtype) {
  case OptimizerStateTensorDTypePolicy::MatchParameter:
    return parameter_dtype;
  case OptimizerStateTensorDTypePolicy::MatchModel:
    return resolve_model_dtype(parameter_dtype, policy);
  case OptimizerStateTensorDTypePolicy::Float32:
    return DataType::Float32;
  case OptimizerStateTensorDTypePolicy::OptimizerStateType:
  default:
    return optimizer_state_type(resolve_model_dtype(parameter_dtype, policy));
  }
}

struct ParameterGroup {
  std::vector<Tensor> params;
  std::optional<float> lr;
  OptimizerStatePolicy state_policy;
  std::string name;

  ParameterGroup() = default;
  explicit ParameterGroup(std::vector<Tensor> parameters, std::optional<float> lr_ = std::nullopt,
                          OptimizerStatePolicy policy = {}, std::string group_name = "")
      : params(std::move(parameters)), lr(lr_), state_policy(policy),
        name(std::move(group_name)) {}
};

class Optimizer {
public:
  Optimizer(std::vector<Tensor> params, float lr)
      : Optimizer(std::vector<ParameterGroup>{ParameterGroup(std::move(params))},
                  lr) {}

  Optimizer(std::vector<ParameterGroup> parameter_groups, float lr)
      : parameter_groups_(std::move(parameter_groups)), lr_(lr) {}
  virtual ~Optimizer() = default;

  virtual void step() = 0;

  float lr() const { return lr_; }
  void set_lr(float value) { lr_ = value; }

  void zero_grad() {
    for_each_parameter([&](Tensor &p, size_t, size_t) { p.zero_grad(); });

    auto *first = first_parameter();
    if (first && is_debug_enabled()) {
      first->impl_->backend().synchronize();
    }
  }

  void scale_gradients(float factor) {
    if (factor == 1.0f) {
      return;
    }

    for_each_parameter([&](Tensor &p, size_t, size_t) {
      if (!p.has_grad()) {
        return;
      }
      scale_tensor_in_place(p.grad(), factor);
    });
  }

  float grad_global_norm() const {
    double total_sq = 0.0;
    for_each_parameter([&](Tensor p, size_t, size_t) {
      if (!p.has_grad()) {
        return;
      }
      Device cpu{DeviceType::CPU, 0};
      Tensor grad_cpu =
          (p.grad().device().type == DeviceType::CPU) ? p.grad() : p.grad().to(cpu);
      const char *bytes = static_cast<const char *>(grad_cpu.data());
      const size_t stride = dtype_size(grad_cpu.dtype());
      for (size_t i = 0; i < grad_cpu.size(); ++i) {
        const double value =
            read_scalar_from_buffer(bytes + i * stride, grad_cpu.dtype()).value;
        total_sq += value * value;
      }
    });
    return static_cast<float>(std::sqrt(total_sq));
  }

  float clip_grad_norm(float max_norm) {
    const float total_norm = grad_global_norm();
    if (max_norm > 0.0f && total_norm > max_norm && total_norm > 0.0f) {
      scale_gradients(max_norm / total_norm);
    }
    return total_norm;
  }

  void apply_weight_decay(float weight_decay) {
    if (weight_decay <= 0.0f) {
      return;
    }
    for_each_parameter([&](Tensor &p, size_t group_index, size_t) {
      if (!is_floating(p.dtype())) {
        return;
      }
      const float factor = 1.0f - group_lr(group_index) * weight_decay;
      scale_tensor_in_place(p, factor);
    });
  }

  const std::vector<ParameterGroup> &parameter_groups() const {
    return parameter_groups_;
  }

  size_t parameter_count() const {
    size_t total = 0;
    for (const auto &group : parameter_groups_) {
      total += group.params.size();
    }
    return total;
  }

protected:
  template <typename Fn> void for_each_parameter(Fn &&fn) {
    for (size_t group_index = 0; group_index < parameter_groups_.size();
         ++group_index) {
      auto &group = parameter_groups_[group_index];
      for (size_t param_index = 0; param_index < group.params.size();
           ++param_index) {
        fn(group.params[param_index], group_index, param_index);
      }
    }
  }

  template <typename Fn> void for_each_parameter(Fn &&fn) const {
    for (size_t group_index = 0; group_index < parameter_groups_.size();
         ++group_index) {
      const auto &group = parameter_groups_[group_index];
      for (size_t param_index = 0; param_index < group.params.size();
           ++param_index) {
        fn(group.params[param_index], group_index, param_index);
      }
    }
  }

  float group_lr(size_t group_index) const {
    return parameter_groups_[group_index].lr.value_or(lr_);
  }

  const OptimizerStatePolicy &group_state_policy(size_t group_index) const {
    return parameter_groups_[group_index].state_policy;
  }

  Tensor *first_parameter() {
    for (auto &group : parameter_groups_) {
      if (!group.params.empty()) {
        return &group.params.front();
      }
    }
    return nullptr;
  }

  static void copy_tensor_storage(const Tensor &src, Tensor &dst) {
    if (src.bytes() != dst.bytes()) {
      throw std::runtime_error("Tensor copy requires matching storage size");
    }

    Device cpu{DeviceType::CPU, 0};
    Tensor src_cpu = (src.device().type == DeviceType::CPU) ? src : src.to(cpu);
    if (dst.device().type == DeviceType::CPU) {
      std::memcpy(dst.data(), src_cpu.data(), dst.bytes());
      return;
    }

    BackendManager::get(dst.device())
        ->copy(src_cpu.data(), dst.data(), dst.bytes(), cpu, dst.device());
  }

  static void scale_tensor_in_place(Tensor tensor, float factor) {
    if (!tensor.impl_) {
      return;
    }
    if (!is_floating(tensor.dtype())) {
      throw std::runtime_error(
          "Gradient scaling only supports floating-point tensors");
    }

    Device cpu{DeviceType::CPU, 0};
    Tensor tensor_cpu = (tensor.device().type == DeviceType::CPU) ? tensor
                                                                 : tensor.to(cpu);
    char *bytes = static_cast<char *>(tensor_cpu.data());
    const size_t stride = dtype_size(tensor_cpu.dtype());
    for (size_t i = 0; i < tensor_cpu.size(); ++i) {
      const float value =
          read_scalar_from_buffer(bytes + i * stride, tensor_cpu.dtype())
              .as_float();
      write_scalar_to_buffer(bytes + i * stride, tensor_cpu.dtype(),
                             value * factor);
    }

    if (tensor.device().type == DeviceType::CPU) {
      std::memcpy(tensor.data(), tensor_cpu.data(), tensor.bytes());
    } else {
      BackendManager::get(tensor.device())
          ->copy(tensor_cpu.data(), tensor.data(), tensor.bytes(), cpu,
                 tensor.device());
    }
    tensor.bump_version();
  }

  std::vector<ParameterGroup> parameter_groups_;
  float lr_;
};

class Adam : public Optimizer {
public:
  Adam(std::vector<Tensor> params, float lr = 1e-3, float beta1 = 0.9,
       float beta2 = 0.999, float eps = 1e-8)
      : Adam(std::vector<ParameterGroup>{ParameterGroup(std::move(params))}, lr,
             beta1, beta2, eps) {}

  Adam(std::vector<ParameterGroup> parameter_groups, float lr = 1e-3,
       float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8)
      : Optimizer(std::move(parameter_groups), lr), beta1_(beta1),
        beta2_(beta2), eps_(eps) {
    initialize_state();
  }

  void step() override {
    step_count_++;
    size_t flat_index = 0;
    for (size_t group_index = 0; group_index < parameter_groups_.size();
         ++group_index) {
      const float lr = group_lr(group_index);
      for (size_t param_index = 0;
           param_index < parameter_groups_[group_index].params.size();
           ++param_index, ++flat_index) {
        auto &p = parameter_groups_[group_index].params[param_index];
        if (!p.has_grad())
          continue;

        if (!master_weights_[flat_index].impl_ &&
            p.dtype() == DataType::Float32 &&
            exp_avg_[flat_index].dtype() == DataType::Float32 &&
            exp_avg_sq_[flat_index].dtype() == DataType::Float32) {
          p.impl_->backend().adam_step(*p.impl_->storage,
                                       *p.grad().impl_->storage,
                                       *exp_avg_[flat_index].impl_->storage,
                                       *exp_avg_sq_[flat_index].impl_->storage,
                                       lr, beta1_, beta2_, eps_, step_count_,
                                       p.size());
          continue;
        }

        apply_typed_adam_step(p, p.grad(), exp_avg_[flat_index],
                              exp_avg_sq_[flat_index], master_weights_[flat_index],
                              lr);
      }
    }
  }

  DataType state_dtype_for_parameter(size_t flat_index) const {
    return exp_avg_.at(flat_index).dtype();
  }

  bool has_master_weight_for_parameter(size_t flat_index) const {
    return master_weights_.at(flat_index).impl_ != nullptr;
  }

  DataType master_weight_dtype_for_parameter(size_t flat_index) const {
    if (!has_master_weight_for_parameter(flat_index)) {
      throw std::runtime_error("Requested master weight dtype for parameter without master weight");
    }
    return master_weights_.at(flat_index).dtype();
  }

private:
  void initialize_state() {
    for (const auto &group : parameter_groups_) {
      for (const auto &p : group.params) {
        const DataType state_dtype =
            resolve_optimizer_state_dtype(p.dtype(), group.state_policy);
        exp_avg_.push_back(
            munet::ops::zeros(p.shape(), p.device(), false, state_dtype));
        exp_avg_sq_.push_back(
            munet::ops::zeros(p.shape(), p.device(), false, state_dtype));

        const auto master_dtype =
            resolve_master_weight_dtype(p.dtype(), group.state_policy);
        if (master_dtype.has_value()) {
          Tensor master = p.to(master_dtype.value());
          master.set_requires_grad(false);
          master_weights_.push_back(master);
        } else {
          master_weights_.push_back(Tensor());
        }
      }
    }
  }

  void apply_typed_adam_step(Tensor &param, const Tensor &grad, Tensor &exp_avg,
                             Tensor &exp_avg_sq, Tensor &master_weight,
                             float lr) {
    if (!is_floating(param.dtype())) {
      throw std::runtime_error("Adam only supports floating-point parameters");
    }

    Device cpu{DeviceType::CPU, 0};
    Tensor param_cpu;
    Tensor *target_param = &param;
    if (master_weight.impl_) {
      target_param = &master_weight;
      param_cpu = (master_weight.device().type == DeviceType::CPU)
                      ? master_weight
                      : master_weight.to(cpu);
    } else {
      param_cpu = (param.device().type == DeviceType::CPU) ? param
                                                           : param.to(cpu);
    }
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
      const float updated = p_val - lr * m_hat / (std::sqrt(v_hat) + eps_);

      write_scalar_to_buffer(param_bytes + j * param_stride, param_cpu.dtype(),
                             updated);
      write_scalar_to_buffer(exp_avg_bytes + j * exp_avg_stride,
                             exp_avg_cpu.dtype(), m_val);
      write_scalar_to_buffer(exp_avg_sq_bytes + j * exp_avg_sq_stride,
                             exp_avg_sq_cpu.dtype(), v_val);
    }

    copy_tensor_storage(exp_avg_cpu, exp_avg);
    copy_tensor_storage(exp_avg_sq_cpu, exp_avg_sq);

    if (master_weight.impl_) {
      copy_tensor_storage(param_cpu, *target_param);
      Tensor updated_param = param_cpu.to(param.dtype());
      copy_tensor_storage(updated_param, param);
    } else {
      copy_tensor_storage(param_cpu, param);
    }
  }

  float beta1_, beta2_, eps_;
  int step_count_ = 0;
  std::vector<Tensor> exp_avg_;
  std::vector<Tensor> exp_avg_sq_;
  std::vector<Tensor> master_weights_;
};

class SGD : public Optimizer {
public:
  using Optimizer::Optimizer;

  void step() override {
    size_t flat_index = 0;
    for (size_t group_index = 0; group_index < parameter_groups_.size();
         ++group_index) {
      const float lr = group_lr(group_index);
      const auto &policy = group_state_policy(group_index);
      for (size_t param_index = 0;
           param_index < parameter_groups_[group_index].params.size();
           ++param_index, ++flat_index) {
        auto &p = parameter_groups_[group_index].params[param_index];
        if (!p.has_grad()) {
          continue;
        }
        const auto master_dtype = resolve_master_weight_dtype(p.dtype(), policy);
        if (!master_dtype.has_value()) {
          p.step(lr);
          continue;
        }
        apply_master_weight_sgd_step(p, p.grad(), lr, master_dtype.value());
      }
    }

    auto *first = first_parameter();
    if (first && is_debug_enabled()) {
      first->impl_->backend().synchronize();
    }
  }

private:
  static void apply_master_weight_sgd_step(Tensor &param, const Tensor &grad,
                                           float lr, DataType master_dtype) {
    Device cpu{DeviceType::CPU, 0};
    Tensor master = param.to(master_dtype);
    Tensor master_cpu =
        (master.device().type == DeviceType::CPU) ? master : master.to(cpu);
    Tensor grad_cpu = (grad.device().type == DeviceType::CPU) ? grad
                                                              : grad.to(cpu);

    char *param_bytes = static_cast<char *>(master_cpu.data());
    const char *grad_bytes = static_cast<const char *>(grad_cpu.data());
    const size_t param_stride = dtype_size(master_cpu.dtype());
    const size_t grad_stride = dtype_size(grad_cpu.dtype());

    for (size_t j = 0; j < param.size(); ++j) {
      const float p_val =
          read_scalar_from_buffer(param_bytes + j * param_stride, master_cpu.dtype())
              .as_float();
      const float g_val =
          read_scalar_from_buffer(grad_bytes + j * grad_stride, grad_cpu.dtype())
              .as_float();
      write_scalar_to_buffer(param_bytes + j * param_stride, master_cpu.dtype(),
                             p_val - lr * g_val);
    }

    Tensor updated_param = master_cpu.to(param.dtype());
    copy_tensor_storage(updated_param, param);
  }
};

} // namespace optim

namespace amp {

struct LossScaleUpdate {
  bool found_inf = false;
};

class GradScaler {
public:
  explicit GradScaler(bool enabled = true, float initial_scale = 65536.0f,
                      float growth_factor = 2.0f,
                      float backoff_factor = 0.5f,
                      int growth_interval = 2000)
      : enabled_(enabled), scale_(initial_scale),
        growth_factor_(growth_factor), backoff_factor_(backoff_factor),
        growth_interval_(growth_interval) {}

  bool enabled() const { return enabled_; }
  float scale_value() const { return scale_; }

  Tensor scale(const Tensor &loss) const {
    if (!enabled_) {
      return loss;
    }
    if (!is_floating(loss.dtype())) {
      throw std::runtime_error("GradScaler expects floating-point loss tensors");
    }

    Tensor factor({1}, loss.device(), loss.dtype(), false);
    factor.fill_(make_scalar(scale_, loss.dtype()));
    return loss * factor;
  }

  void unscale_(optim::Optimizer &optimizer) const {
    if (!enabled_) {
      return;
    }
    optimizer.scale_gradients(1.0f / scale_);
  }

  void update(LossScaleUpdate update = {}) {
    if (!enabled_) {
      return;
    }
    if (update.found_inf) {
      scale_ *= backoff_factor_;
      growth_tracker_ = 0;
      return;
    }
    ++growth_tracker_;
    if (growth_tracker_ >= growth_interval_) {
      scale_ *= growth_factor_;
      growth_tracker_ = 0;
    }
  }

private:
  bool enabled_ = true;
  float scale_ = 65536.0f;
  float growth_factor_ = 2.0f;
  float backoff_factor_ = 0.5f;
  int growth_interval_ = 2000;
  int growth_tracker_ = 0;
};

enum class AutocastConversionPolicy {
  Strict,
  PromoteInputs,
  PromoteInputsAndOutputs,
};

struct AutocastOptions {
  bool enabled = false;
  DeviceType device_type = DeviceType::CPU;
  DataType compute_dtype = DataType::Float16;
  AutocastConversionPolicy conversion_policy =
      AutocastConversionPolicy::PromoteInputs;
};

inline AutocastOptions &autocast_state() {
  static thread_local AutocastOptions state;
  return state;
}

inline const AutocastOptions &current_autocast_options() {
  return autocast_state();
}

inline bool autocast_enabled() { return autocast_state().enabled; }

inline bool allows_implicit_conversion(DataType from, DataType to,
                                       const AutocastOptions &options =
                                           current_autocast_options()) {
  if (!options.enabled || from == to) {
    return from == to;
  }
  if (!is_floating(from) || !is_floating(to)) {
    return false;
  }
  if (to != options.compute_dtype) {
    return false;
  }
  return options.conversion_policy == AutocastConversionPolicy::PromoteInputs ||
         options.conversion_policy ==
             AutocastConversionPolicy::PromoteInputsAndOutputs;
}

inline bool allows_output_conversion(
    DataType from, DataType to,
    const AutocastOptions &options = current_autocast_options()) {
  if (!options.enabled || from == to) {
    return from == to;
  }
  if (!is_floating(from) || !is_floating(to)) {
    return false;
  }
  if (to != options.compute_dtype) {
    return false;
  }
  return options.conversion_policy ==
         AutocastConversionPolicy::PromoteInputsAndOutputs;
}

class AutocastGuard {
public:
  explicit AutocastGuard(AutocastOptions options)
      : previous_(autocast_state()) {
    autocast_state() = options;
  }

  ~AutocastGuard() { autocast_state() = previous_; }

private:
  AutocastOptions previous_;
};

} // namespace amp
} // namespace munet
