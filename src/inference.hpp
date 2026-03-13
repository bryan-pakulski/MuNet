#pragma once

#include "core/module.hpp"
#include <chrono>
#include <stdexcept>
#include <string>

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

enum class LossScaleMode { None, Dynamic, Static };
enum class PrecisionFallbackMode { Error, WarnAndUpcast };

struct PrecisionPolicy {
  DataType param_dtype = DataType::Float32;
  DataType activation_dtype = DataType::Float32;
  DataType gradient_dtype = DataType::Float32;
  DataType optimizer_state_dtype = DataType::Float32;
  DataType accumulation_dtype = DataType::Float32;
  LossScaleMode loss_scale_mode = LossScaleMode::None;
  PrecisionFallbackMode fallback_mode = PrecisionFallbackMode::WarnAndUpcast;
};

struct EngineConfig {
  Device device{DeviceType::CPU, 0};
  int warmup_runs = 0;
  bool strict_shape_check = true;
  PrecisionPolicy precision_policy{};
};

struct EngineStats {
  size_t runs = 0;
  double last_run_ms = 0.0;
  double compile_ms = 0.0;
  std::vector<int> compiled_input_shape{};
  std::vector<int> compiled_output_shape{};
};

class Engine {
public:
  explicit Engine(EngineConfig cfg = {}) : config_(cfg) {}

  void set_device(Device device) {
    config_.device = device;
    compiled_ = false;
    prepared_ = false;
  }
  Device device() const { return config_.device; }

  void set_warmup_runs(int warmup_runs) {
    if (warmup_runs < 0)
      throw std::runtime_error("Engine warmup_runs must be >= 0");
    config_.warmup_runs = warmup_runs;
  }

  void set_strict_shape_check(bool enabled) { config_.strict_shape_check = enabled; }

  void set_precision_policy(const PrecisionPolicy &policy) {
    config_.precision_policy = policy;
  }

  const PrecisionPolicy &precision_policy() const {
    return config_.precision_policy;
  }

  void load(const std::shared_ptr<core::Module> &module) {
    if (!module)
      throw std::runtime_error("Engine::load received null module");

    module_ = module;
    module_->to(config_.device);
    module_->eval();

    loaded_ = true;
    prepared_ = false;
    compiled_ = false;
    stats_ = {};
  }

  void compile(const Tensor &example_input,
               const std::vector<int> &expected_input_shape = {},
               const std::vector<int> &expected_output_shape = {}) {
    ensure_loaded();

    auto start = std::chrono::high_resolution_clock::now();
    Tensor input = to_engine_device(example_input);
    compiled_input_shape_ = input.shape();

    if (!expected_input_shape.empty()) {
      validate_shape(expected_input_shape, compiled_input_shape_,
                     "Engine: compile expected_input_shape mismatch");
      expected_input_shape_ = expected_input_shape;
      use_expected_input_shape_ = true;
    } else {
      expected_input_shape_.clear();
      use_expected_input_shape_ = false;
    }

    Tensor out = module_->forward(input);
    compiled_output_shape_ = out.shape();

    if (!expected_output_shape.empty()) {
      validate_shape(expected_output_shape, compiled_output_shape_,
                     "Engine: compile expected_output_shape mismatch");
      expected_output_shape_ = expected_output_shape;
      use_expected_output_shape_ = true;
    } else {
      expected_output_shape_.clear();
      use_expected_output_shape_ = false;
    }

    for (int i = 1; i < config_.warmup_runs; ++i) {
      (void)module_->forward(input);
    }

    auto end = std::chrono::high_resolution_clock::now();
    stats_.compile_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    stats_.compiled_input_shape = compiled_input_shape_;
    stats_.compiled_output_shape = compiled_output_shape_;

    compiled_ = true;
    prepared_ = true;
  }

  void prepare(const Tensor &example_input) { compile(example_input); }

  Tensor run(const Tensor &input) {
    ensure_loaded();

    Tensor in = to_engine_device(input);
    if (config_.strict_shape_check && compiled_) {
      if (use_expected_input_shape_) {
        validate_shape(expected_input_shape_, in.shape(),
                       "Engine: input shape mismatch with expected compiled shape");
      } else if (in.shape() != compiled_input_shape_) {
        throw std::runtime_error("Engine: input shape mismatch with compiled shape");
      }
    }

    auto start = std::chrono::high_resolution_clock::now();
    Tensor out = module_->forward(in);

    if (config_.strict_shape_check && compiled_ && use_expected_output_shape_) {
      validate_shape(expected_output_shape_, out.shape(),
                     "Engine: output shape mismatch with expected compiled shape");
    }
    auto end = std::chrono::high_resolution_clock::now();

    stats_.runs += 1;
    stats_.last_run_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    return out;
  }

  std::vector<Tensor> run_batch(const std::vector<Tensor> &inputs) {
    ensure_loaded();

    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());
    for (const auto &in : inputs) {
      outputs.push_back(run(in));
    }
    return outputs;
  }

  bool is_loaded() const { return loaded_; }
  bool is_prepared() const { return prepared_; }
  bool is_compiled() const { return compiled_; }
  const std::vector<int> &compiled_input_shape() const { return compiled_input_shape_; }
  const std::vector<int> &compiled_output_shape() const { return compiled_output_shape_; }
  EngineStats stats() const { return stats_; }

private:
  Tensor to_engine_device(const Tensor &input) const {
    return (input.device() == config_.device) ? input : input.to(config_.device);
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

  void ensure_loaded() const {
    if (!loaded_ || !module_)
      throw std::runtime_error("Engine: no module loaded");
  }

  EngineConfig config_;
  EngineStats stats_;
  std::shared_ptr<core::Module> module_ = nullptr;
  bool loaded_ = false;
  bool prepared_ = false;
  bool compiled_ = false;
  std::vector<int> compiled_input_shape_{};
  std::vector<int> compiled_output_shape_{};
  std::vector<int> expected_input_shape_{};
  std::vector<int> expected_output_shape_{};
  bool use_expected_input_shape_ = false;
  bool use_expected_output_shape_ = false;
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
