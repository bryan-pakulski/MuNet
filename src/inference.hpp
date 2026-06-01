#pragma once

#include "core/grad_mode.hpp"
#include "core/module.hpp"
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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

struct EngineConfig {
  Device device{DeviceType::VULKAN, 0};
  bool strict_shape_check = true;
  bool allow_autograd_inputs = false;
};

struct EngineStats {
  bool loaded = false;
  bool prepared = false;
  bool compiled = false;
  size_t runs = 0;
  size_t batch_runs = 0;
  double last_run_ms = 0.0;
  double compile_ms = 0.0;
  std::vector<int> compiled_input_shape{};
  std::vector<int> compiled_output_shape{};
};

std::shared_ptr<core::Module>
load_serialized(const std::string &path,
                std::optional<Device> device = std::nullopt);

void load_weights_serialized(const std::shared_ptr<core::Module> &module,
                             const std::string &path,
                             std::optional<Device> device = std::nullopt);

namespace detail {
class InferenceModeGuard {
public:
  InferenceModeGuard() : previous_(GradMode::is_enabled()) {
    GradMode::set_enabled(false);
  }

  ~InferenceModeGuard() { GradMode::set_enabled(previous_); }

private:
  bool previous_;
};

class ShapeContract {
public:
  void reset() {
    active_ = false;
    rank_ = 0;
    constrained_indices_.clear();
    constrained_dims_.clear();
  }

  void compile_exact(const std::vector<int> &shape) {
    active_ = true;
    rank_ = shape.size();
    constrained_indices_.resize(shape.size());
    constrained_dims_ = shape;
    for (size_t i = 0; i < shape.size(); ++i) {
      constrained_indices_[i] = i;
    }
  }

  void compile_expected(const std::vector<int> &expected) {
    active_ = true;
    rank_ = expected.size();
    constrained_indices_.clear();
    constrained_dims_.clear();
    constrained_indices_.reserve(expected.size());
    constrained_dims_.reserve(expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] == -1) {
        continue;
      }
      constrained_indices_.push_back(i);
      constrained_dims_.push_back(expected[i]);
    }
  }

  std::string mismatch_reason(const std::vector<int> &actual) const {
    if (!active_) {
      return {};
    }
    if (actual.size() != rank_) {
      return "rank mismatch";
    }
    for (size_t i = 0; i < constrained_indices_.size(); ++i) {
      const size_t dim_index = constrained_indices_[i];
      if (actual[dim_index] != constrained_dims_[i]) {
        return "dim mismatch at index " + std::to_string(dim_index);
      }
    }
    return {};
  }

private:
  bool active_ = false;
  size_t rank_ = 0;
  std::vector<size_t> constrained_indices_{};
  std::vector<int> constrained_dims_{};
};
} // namespace detail

class Engine {
public:
  explicit Engine(EngineConfig cfg = {}) : config_(cfg) {}

  void set_device(Device device) {
    config_.device = device;
    input_shape_contract_.reset();
    output_shape_contract_.reset();
    compiled_ = false;
    prepared_ = false;
    sync_stats_state();
  }
  Device device() const { return config_.device; }

  void set_strict_shape_check(bool enabled) {
    config_.strict_shape_check = enabled;
  }
  void set_allow_autograd_inputs(bool enabled) {
    config_.allow_autograd_inputs = enabled;
  }
  bool allow_autograd_inputs() const { return config_.allow_autograd_inputs; }

  void load(const std::string &serialized_path) {
    load(load_serialized(serialized_path));
  }

  void load(const std::shared_ptr<core::Module> &module) {
    if (!module) {
      throw std::runtime_error("Engine::load received null module");
    }

    module_ = module;
    stats_ = {};
    input_shape_contract_.reset();
    output_shape_contract_.reset();
    compiled_input_shape_.clear();
    compiled_output_shape_.clear();

    if (!module_->is_on(config_.device)) {
      module_->to(config_.device);
    }
    module_->eval();

    loaded_ = true;
    prepared_ = false;
    compiled_ = false;
    sync_stats_state();
  }

  Tensor prepare(const Tensor &input) {
    ensure_loaded();
    Tensor prepared = prepare_input(input, "prepare");
    if (config_.strict_shape_check && compiled_) {
      validate_shape_contract(
          input_shape_contract_, prepared.shape(),
          "Engine: input shape mismatch with compiled shape");
    }
    prepared_ = true;
    sync_stats_state();
    return prepared;
  }

  void compile(const Tensor &example_input,
               const std::vector<int> &expected_input_shape = {},
               const std::vector<int> &expected_output_shape = {}) {
    ensure_loaded();
    const auto start = std::chrono::high_resolution_clock::now();
    Tensor input = prepare_input(example_input, "compile");

    detail::InferenceModeGuard no_grad;
    compiled_input_shape_ = input.shape();

    if (!expected_input_shape.empty()) {
      validate_shape(expected_input_shape, compiled_input_shape_,
                     "Engine: compile expected_input_shape mismatch");
      input_shape_contract_.compile_expected(expected_input_shape);
    } else {
      input_shape_contract_.compile_exact(compiled_input_shape_);
    }

    Tensor output = module_->forward(input);
    ensure_inference_output(output, "compile");
    compiled_output_shape_ = output.shape();

    if (!expected_output_shape.empty()) {
      validate_shape(expected_output_shape, compiled_output_shape_,
                     "Engine: compile expected_output_shape mismatch");
      output_shape_contract_.compile_expected(expected_output_shape);
    } else {
      output_shape_contract_.compile_exact(compiled_output_shape_);
    }

    stats_.compile_ms = elapsed_ms(start);
    compiled_ = true;
    prepared_ = true;
    sync_stats_state();
  }

  Tensor run(const Tensor &input) {
    ensure_loaded();
    const auto start = std::chrono::high_resolution_clock::now();
    Tensor prepared = prepare_input(input, "run");

    if (config_.strict_shape_check && compiled_) {
      validate_shape_contract(
          input_shape_contract_, prepared.shape(),
          "Engine: input shape mismatch with compiled shape");
    }

    detail::InferenceModeGuard no_grad;
    Tensor output = module_->forward(prepared);
    ensure_inference_output(output, "run");

    if (config_.strict_shape_check && compiled_) {
      validate_shape_contract(
          output_shape_contract_, output.shape(),
          "Engine: output shape mismatch with compiled shape");
    }

    stats_.runs += 1;
    stats_.last_run_ms = elapsed_ms(start);
    sync_stats_state();
    return output;
  }

  std::vector<Tensor> run_batch(const std::vector<Tensor> &inputs) {
    ensure_loaded();
    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());
    for (const auto &input : inputs) {
      outputs.push_back(run(input));
    }
    stats_.batch_runs += 1;
    sync_stats_state();
    return outputs;
  }

  bool is_loaded() const { return loaded_; }
  bool is_prepared() const { return prepared_; }
  bool is_compiled() const { return compiled_; }
  const std::vector<int> &compiled_input_shape() const {
    return compiled_input_shape_;
  }
  const std::vector<int> &compiled_output_shape() const {
    return compiled_output_shape_;
  }
  EngineStats stats() const { return stats_; }

private:
  void validate_inference_input(const Tensor &input, const char *stage) const {
    if (!config_.allow_autograd_inputs && input.requires_grad()) {
      throw std::runtime_error(
          std::string("Engine: ") + stage +
          " received a tensor with requires_grad=true. "
          "Detach inputs or opt into allow_autograd_inputs for debugging.");
    }
  }

  Tensor prepare_input(const Tensor &input, const char *stage) const {
    validate_inference_input(input, stage);
    if (input.device() == config_.device) {
      return input;
    }
    return input.to(config_.device);
  }

  static void ensure_inference_output(const Tensor &output, const char *stage) {
    if (output.requires_grad()) {
      throw std::runtime_error(std::string("Engine: ") + stage +
                               " produced a gradient-tracked tensor. ");
    }
  }

  static void validate_shape(const std::vector<int> &expected,
                             const std::vector<int> &actual,
                             const std::string &err_prefix) {
    if (expected.size() != actual.size()) {
      throw std::runtime_error(err_prefix + ": rank mismatch");
    }

    for (size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] != -1 && expected[i] != actual[i]) {
        throw std::runtime_error(err_prefix + ": dim mismatch at index " +
                                 std::to_string(i));
      }
    }
  }

  static void validate_shape_contract(const detail::ShapeContract &contract,
                                      const std::vector<int> &actual,
                                      const std::string &err_prefix) {
    if (const std::string reason = contract.mismatch_reason(actual);
        !reason.empty()) {
      throw std::runtime_error(err_prefix + ": " + reason);
    }
  }

  void ensure_loaded() const {
    if (!loaded_ || !module_) {
      throw std::runtime_error("Engine: no module loaded");
    }
  }

  static double
  elapsed_ms(const std::chrono::high_resolution_clock::time_point &start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - start)
        .count();
  }

  void sync_stats_state() {
    stats_.loaded = loaded_;
    stats_.prepared = prepared_;
    stats_.compiled = compiled_;
    stats_.compiled_input_shape = compiled_input_shape_;
    stats_.compiled_output_shape = compiled_output_shape_;
  }

  EngineConfig config_;
  EngineStats stats_;
  std::shared_ptr<core::Module> module_ = nullptr;
  bool loaded_ = false;
  bool prepared_ = false;
  bool compiled_ = false;
  std::vector<int> compiled_input_shape_{};
  std::vector<int> compiled_output_shape_{};
  detail::ShapeContract input_shape_contract_{};
  detail::ShapeContract output_shape_contract_{};
};

class Sequential : public Module {
public:
  Sequential() = default;

  void add(std::shared_ptr<Module> m) {
    register_module(std::to_string(modules_.size()), m);
    ordered_modules_.push_back(m);
  }

  Tensor forward_impl(Tensor x) override {
    for (auto &m : ordered_modules_) {
      x = m->forward(x);
    }
    return x;
  }

  std::vector<std::shared_ptr<Module>> ordered_modules_;
};

} // namespace inference
} // namespace munet
