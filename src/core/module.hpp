#pragma once

#include "tensor.hpp"
#include "backend.hpp"
#include "util/profiler.hpp"
#include "util/timer.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
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

struct OffloadValidationReport {
  bool valid = true;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
  int estimated_boundaries = 0;
  int estimated_ping_pong_boundaries = 0;
};

struct OffloadTransferTelemetry {
  size_t boundary_transfer_count = 0;
  size_t boundary_transfer_bytes = 0;
  std::map<std::string, size_t> direction_counts;
};

struct OffloadPlannerRationale {
  std::string strategy = "manual";
  size_t compute_cost = 0;
  size_t param_bytes = 0;
  size_t activation_bytes = 0;
  size_t transfer_cost = 0;
  size_t projected_mem_bytes = 0;
  std::optional<size_t> budget_bytes;
  std::string source = "planner";
};

class Module {
public:
  explicit Module(TensorOptions default_options = TensorOptions{})
      : default_options_(default_options) {}
  virtual ~Module() = default;

  // Generic forward for single-tensor modules
  Tensor forward(Tensor x) {
    if (!is_profile_enabled()) {
      return forward_impl(std::move(x));
    }

    const Shape input_shape = x.shape();
    const size_t input_bytes = x.bytes();
    Timer timer;
    Tensor out = forward_impl(std::move(x));
    Profiler::get().record(module_profile_label(), timer.elapsed_us(), 0.0,
                           input_bytes, to_string(input_shape));
    return out;
  }

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

  bool is_training() const { return training_; }

  const TensorOptions &default_options() const { return default_options_; }

  virtual bool is_on(Device device) const {
    for (const auto &[name, p] : parameters_) {
      (void)name;
      if (p && p->impl_ && p->device() != device) {
        return false;
      }
    }
    for (const auto &[name, b] : buffers_) {
      (void)name;
      if (b && b->impl_ && b->device() != device) {
        return false;
      }
    }
    for (const auto &[name, m] : modules_) {
      (void)name;
      if (m && !m->is_on(device)) {
        return false;
      }
    }
    return true;
  }

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

  void offload(Device device, const std::vector<std::string> &layers) {
    Module *root = root_module();
    auto mods = const_cast<Module *>(root)->named_modules("");
    for (const auto &layer : layers) {
      auto it = mods.find(layer);
      if (it == mods.end() || !it->second) {
        throw std::runtime_error("offload: unknown layer path '" + layer + "'");
      }
      it->second->to(device);
      root->offload_plan_[layer] = device;
    }
  }

  void clear_offload() {
    Module *root = root_module();
    root->offload_plan_.clear();
    root->offload_plan_rationale_.clear();
  }

  std::map<std::string, Device> offload_plan() const {
    return root_module_const()->offload_plan_;
  }

  std::map<std::string, std::string> offload_plan_rationale() const {
    std::map<std::string, std::string> out;
    const Module *root = root_module_const();
    for (const auto &[layer, rationale] : root->offload_plan_rationale_) {
      out[layer] = serialize_rationale(rationale);
    }
    return out;
  }

  std::map<std::string, OffloadPlannerRationale> offload_plan_rationale_typed()
      const {
    return root_module_const()->offload_plan_rationale_;
  }

  std::map<std::string, std::string> freeze_offload_plan() const {
    std::map<std::string, std::string> frozen;
    for (const auto &[layer, device] : root_module_const()->offload_plan_) {
      frozen[layer] = device.to_string();
    }
    return frozen;
  }

  void apply_offload_plan(const std::map<std::string, std::string> &plan) {
    Module *root = root_module();
    root->clear_offload();
    std::vector<std::string> sorted_layers;
    for (const auto &[layer, _] : plan) {
      sorted_layers.push_back(layer);
    }
    std::sort(sorted_layers.begin(), sorted_layers.end());
    for (const auto &layer : sorted_layers) {
      const std::string &device_spec = plan.at(layer);
      root->offload(parse_device_string(device_spec), {layer});
      OffloadPlannerRationale rationale;
      rationale.strategy = "manual-frozen";
      rationale.source = "manual-frozen";
      root->offload_plan_rationale_[layer] = rationale;
    }
    root->planner_last_strategy_ = "manual-frozen";
  }

  std::map<std::string, Device> auto_offload(
      const std::vector<Device> &devices, const std::string &strategy,
      const Tensor &sample_input,
      const std::map<std::string, size_t> &memory_budgets_bytes = {}) {
    if (devices.empty()) {
      throw std::runtime_error("auto_offload: devices list must not be empty");
    }
    if (strategy != "balanced" && strategy != "memory-first" &&
        strategy != "transfer-minimized") {
      throw std::runtime_error("auto_offload: unsupported strategy '" +
                               strategy + "'");
    }

    Module *root = root_module();
    auto mods = root->named_modules("");
    std::vector<std::string> layers;
    for (const auto &[name, m] : mods) {
      if (!m) {
        continue;
      }
      bool numeric = !name.empty();
      for (char c : name) {
        if (c < '0' || c > '9') {
          numeric = false;
          break;
        }
      }
      if (numeric) {
        layers.push_back(name);
      }
    }
    std::sort(layers.begin(), layers.end(),
              [](const std::string &a, const std::string &b) {
                return std::stoi(a) < std::stoi(b);
              });
    if (layers.empty()) {
      for (const auto &[name, _] : mods) {
        layers.push_back(name);
      }
      std::sort(layers.begin(), layers.end());
    }
    if (layers.empty()) {
      throw std::runtime_error("auto_offload: no layers found");
    }

    root->clear_offload();

    std::vector<double> device_scores(devices.size(), 0.0);
    std::vector<size_t> device_param_bytes(devices.size(), 0);
    std::optional<size_t> previous_device_index;
    size_t activation_bytes = sample_input.bytes();

    for (const auto &layer : layers) {
      auto it = mods.find(layer);
      if (it == mods.end() || !it->second) {
        continue;
      }

      size_t param_bytes = 0;
      DataType dominant_dtype = DataType::Float32;
      auto params = it->second->named_parameters("");
      for (const auto &[_, t] : params) {
        param_bytes += t.bytes();
        dominant_dtype = t.dtype();
      }
      if (param_bytes == 0) {
        param_bytes = activation_bytes;
      }

      std::vector<size_t> candidates;
      for (size_t i = 0; i < devices.size(); ++i) {
        bool supported =
            BackendManager::get(devices[i])->supports(BackendFeature::Matmul,
                                                      dominant_dtype);
        if (supported) {
          candidates.push_back(i);
        }
      }
      if (candidates.empty()) {
        throw std::runtime_error("auto_offload: no candidate device supports layer '" +
                                 layer + "'");
      }

      size_t chosen = candidates.front();
      double chosen_score = std::numeric_limits<double>::infinity();
      size_t chosen_transfer = 0;
      size_t chosen_projected_mem = 0;
      size_t chosen_compute = 0;

      for (size_t idx : candidates) {
        const size_t projected_mem = device_param_bytes[idx] + param_bytes;
        size_t budget = std::numeric_limits<size_t>::max();
        auto budget_it = memory_budgets_bytes.find(devices[idx].to_string());
        if (budget_it != memory_budgets_bytes.end()) {
          budget = budget_it->second;
        }
        if (projected_mem > budget) {
          continue;
        }

        const bool boundary_transfer =
            previous_device_index.has_value() && previous_device_index.value() != idx;
        const size_t transfer_cost = boundary_transfer ? activation_bytes : 0;
        const size_t compute_cost = param_bytes + (activation_bytes / 2);

        double score = 0.0;
        if (strategy == "memory-first") {
          score = static_cast<double>(projected_mem) +
                  static_cast<double>(transfer_cost) * 0.10;
        } else if (strategy == "transfer-minimized") {
          score = static_cast<double>(transfer_cost) * 10.0 +
                  static_cast<double>(projected_mem) * 0.05 +
                  static_cast<double>(device_scores[idx]) * 0.01;
        } else {
          // balanced
          score = device_scores[idx] + static_cast<double>(compute_cost) +
                  static_cast<double>(transfer_cost) * 1.5;
        }

        if (score < chosen_score) {
          chosen = idx;
          chosen_score = score;
          chosen_transfer = transfer_cost;
          chosen_projected_mem = projected_mem;
          chosen_compute = compute_cost;
        }
      }

      if (!std::isfinite(chosen_score)) {
        throw std::runtime_error("auto_offload: layer '" + layer +
                                 "' cannot satisfy device memory budgets");
      }

      root->offload(devices[chosen], {layer});
      OffloadPlannerRationale rationale;
      rationale.strategy = strategy;
      rationale.compute_cost = chosen_compute;
      rationale.param_bytes = param_bytes;
      rationale.activation_bytes = activation_bytes;
      rationale.transfer_cost = chosen_transfer;
      rationale.projected_mem_bytes = chosen_projected_mem;
      rationale.source = "planner";
      auto budget_it = memory_budgets_bytes.find(devices[chosen].to_string());
      if (budget_it != memory_budgets_bytes.end()) {
        rationale.budget_bytes = budget_it->second;
      }
      root->offload_plan_rationale_[layer] = rationale;
      device_scores[chosen] += static_cast<double>(chosen_compute + chosen_transfer);
      device_param_bytes[chosen] += param_bytes;
      previous_device_index = chosen;
      activation_bytes = std::max<size_t>(activation_bytes, param_bytes);
    }
    root->planner_last_strategy_ = strategy;
    return root->offload_plan_;
  }

  OffloadValidationReport validate_offload_plan(const Tensor &sample_input) const {
    const Module *root = root_module_const();
    OffloadValidationReport report;

    auto mods = const_cast<Module *>(root)->named_modules("");
    for (const auto &[layer, device] : root->offload_plan_) {
      auto it = mods.find(layer);
      if (it == mods.end() || !it->second) {
        report.errors.push_back("unknown layer path: " + layer);
        continue;
      }

      auto params = it->second->named_parameters("");
      for (const auto &[name, tensor] : params) {
        (void)name;
        if (!BackendManager::get(device)->supports(BackendFeature::Matmul, tensor.dtype())) {
          report.errors.push_back("backend '" + device.to_string() +
                                  "' does not support dtype for layer '" +
                                  layer + "' (matmul check)");
          break;
        }
      }
    }

    // Estimate boundaries and synthetic ping-pong for numeric Sequential names.
    std::vector<std::pair<int, Device>> ordered;
    for (const auto &[layer, device] : root->offload_plan_) {
      bool numeric = !layer.empty();
      for (char c : layer) {
        if (c < '0' || c > '9') {
          numeric = false;
          break;
        }
      }
      if (numeric) {
        ordered.push_back({std::stoi(layer), device});
      }
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    std::optional<Device> prev_device;
    std::optional<Device> prev_prev_device;
    for (const auto &[idx, dev] : ordered) {
      (void)idx;
      if (prev_device.has_value() && prev_device.value() != dev) {
        report.estimated_boundaries++;
        if (prev_prev_device.has_value() && prev_prev_device.value() == dev) {
          report.estimated_ping_pong_boundaries++;
        }
      }
      prev_prev_device = prev_device;
      prev_device = dev;
    }
    if (report.estimated_ping_pong_boundaries > 0) {
      report.warnings.push_back("plan contains potential ping-pong boundaries");
    }

    try {
      Tensor probe = sample_input;
      (void)const_cast<Module *>(this)->forward(std::move(probe));
    } catch (const std::exception &e) {
      report.errors.push_back(std::string("shape/runtime continuity check failed: ") + e.what());
    }

    report.valid = report.errors.empty();
    return report;
  }

  void set_offload_warnings(bool enabled = true) {
    root_module()->offload_warnings_enabled_ = enabled;
  }

  void set_offload_warning_threshold_bytes(size_t threshold_bytes) {
    root_module()->offload_warning_threshold_bytes_ = threshold_bytes;
  }

  OffloadTransferTelemetry offload_telemetry_snapshot() const {
    return root_module_const()->offload_telemetry_;
  }

  void reset_offload_telemetry() {
    root_module()->offload_telemetry_ = OffloadTransferTelemetry{};
    root_module()->last_offload_direction_.clear();
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
      m->parent_ = this;
      m->registered_name_ = name;
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
  virtual Tensor forward_impl(Tensor x) = 0;

  Tensor enforce_offload_boundary(const std::string &child_name,
                                  Tensor x) const {
    const auto expected = lookup_offload_device_for_child(child_name);
    if (expected.has_value() && x.impl_ && x.device() != expected.value()) {
      Module *root = const_cast<Module *>(root_module_const());
      auto &telemetry = root->offload_telemetry_;
      telemetry.boundary_transfer_count += 1;
      telemetry.boundary_transfer_bytes += x.bytes();
      const std::string direction =
          x.device().to_string() + "->" + expected.value().to_string();
      telemetry.direction_counts[direction] += 1;
      if (root->offload_warnings_enabled_ &&
          x.bytes() <= root->offload_warning_threshold_bytes_) {
        std::cerr << "[WARN] offload small transfer bytes=" << x.bytes()
                  << " direction=" << direction << std::endl;
      }
      const std::string reverse_direction =
          expected.value().to_string() + "->" + x.device().to_string();
      if (root->offload_warnings_enabled_ && !root->last_offload_direction_.empty() &&
          root->last_offload_direction_ == reverse_direction) {
        std::cerr << "[WARN] offload potential ping-pong previous="
                  << root->last_offload_direction_ << " current=" << direction
                  << std::endl;
      }
      root->last_offload_direction_ = direction;
      return x.to(expected.value());
    }
    return x;
  }

  std::string module_profile_label() const {
    const std::string path = module_path();
    return path.empty() ? "module.root.forward" : "module." + path + ".forward";
  }

  std::string module_path() const {
    if (!parent_) {
      return registered_name_;
    }

    const std::string parent_path = parent_->module_path();
    if (parent_path.empty()) {
      return registered_name_;
    }
    if (registered_name_.empty()) {
      return parent_path;
    }
    return parent_path + "." + registered_name_;
  }

  std::optional<Device>
  lookup_offload_device_for_child(const std::string &child_name) const {
    const Module *root = root_module_const();
    std::string full = module_path();
    if (!full.empty()) {
      full += ".";
    }
    full += child_name;
    auto it = root->offload_plan_.find(full);
    if (it == root->offload_plan_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  Module *root_module() {
    Module *m = this;
    while (m->parent_) {
      m = m->parent_;
    }
    return m;
  }

  const Module *root_module_const() const {
    const Module *m = this;
    while (m->parent_) {
      m = m->parent_;
    }
    return m;
  }

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

  static Device parse_device_string(const std::string &device_spec) {
    const auto pos = device_spec.find(':');
    if (pos == std::string::npos) {
      throw std::runtime_error("invalid device spec '" + device_spec +
                               "' (expected '<type>:<index>')");
    }
    const std::string type_str = device_spec.substr(0, pos);
    const std::string idx_str = device_spec.substr(pos + 1);
    if (idx_str.empty()) {
      throw std::runtime_error("invalid device spec '" + device_spec +
                               "' (missing index)");
    }
    for (char c : idx_str) {
      if (!std::isdigit(static_cast<unsigned char>(c))) {
        throw std::runtime_error("invalid device spec '" + device_spec +
                                 "' (index is not numeric)");
      }
    }
    DeviceType type = DeviceType::UNKNOWN;
    if (type_str == "cpu") {
      type = DeviceType::CPU;
    } else if (type_str == "cuda") {
      type = DeviceType::CUDA;
    } else if (type_str == "vulkan") {
      type = DeviceType::VULKAN;
    } else {
      throw std::runtime_error("invalid device spec '" + device_spec +
                               "' (unknown type)");
    }
    return Device{type, std::stoi(idx_str)};
  }

  static std::string serialize_rationale(const OffloadPlannerRationale &r) {
    std::ostringstream out;
    out << "source=" << r.source << ",strategy=" << r.strategy
        << ",compute_cost=" << r.compute_cost
        << ",param_bytes=" << r.param_bytes
        << ",activation_bytes=" << r.activation_bytes
        << ",transfer_cost=" << r.transfer_cost
        << ",projected_mem_bytes=" << r.projected_mem_bytes;
    if (r.budget_bytes.has_value()) {
      out << ",budget_bytes=" << r.budget_bytes.value();
    }
    return out.str();
  }

  bool training_ = true;
  TensorOptions default_options_;
  std::map<std::string, Tensor *> parameters_;
  std::map<std::string, Tensor *> buffers_;
  std::map<std::string, BufferRegistration> buffer_registrations_;
  std::map<std::string, std::shared_ptr<Module>> modules_;
  std::map<std::string, Device> offload_plan_;
  std::map<std::string, OffloadPlannerRationale> offload_plan_rationale_;
  std::string planner_last_strategy_ = "manual";
  OffloadTransferTelemetry offload_telemetry_;
  bool offload_warnings_enabled_ = true;
  size_t offload_warning_threshold_bytes_ = 64 * 1024;
  std::string last_offload_direction_;
  Module *parent_ = nullptr;
  std::string registered_name_;
};

} // namespace core
} // namespace munet
