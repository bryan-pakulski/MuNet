#pragma once

#include "autograd/engine.hpp"
#include "core/module.hpp"
#include "core/util/profiler.hpp"
#include <chrono>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
  Device device{DeviceType::CPU, 0};
  int warmup_runs = 0;
  bool strict_shape_check = true;
  bool allow_autograd_inputs = false;
  bool capture_profiler_memory = true;
};

struct EngineStats {
  size_t runs = 0;
  double last_run_ms = 0.0;
  double compile_ms = 0.0;
  std::vector<int> compiled_input_shape{};
  std::vector<int> compiled_output_shape{};
  size_t current_memory_bytes = 0;
  size_t peak_memory_bytes = 0;
};

enum class EngineEventType {
  LoadStarted,
  LoadCompleted,
  CompileStarted,
  CompileCompleted,
  RunStarted,
  RunCompleted,
  Error,
};

struct EngineEvent {
  EngineEventType type{EngineEventType::LoadStarted};
  Device device{DeviceType::CPU, 0};
  size_t run_index = 0;
  double duration_ms = 0.0;
  std::vector<int> input_shape{};
  std::vector<int> output_shape{};
  size_t current_memory_bytes = 0;
  size_t peak_memory_bytes = 0;
  std::string message{};
};

using EngineObserver = std::function<void(const EngineEvent &)>;

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
} // namespace detail

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
  void set_allow_autograd_inputs(bool enabled) { config_.allow_autograd_inputs = enabled; }
  void set_capture_profiler_memory(bool enabled) { config_.capture_profiler_memory = enabled; }

  bool allow_autograd_inputs() const { return config_.allow_autograd_inputs; }
  bool capture_profiler_memory() const { return config_.capture_profiler_memory; }

  void set_observer(EngineObserver observer) { observer_ = std::move(observer); }
  void clear_observer() { observer_ = nullptr; }

  void load(const std::shared_ptr<core::Module> &module) {
    emit_event(EngineEventType::LoadStarted, 0.0, {}, {}, 0,
               "Loading module into inference engine");

    try {
      if (!module)
        throw std::runtime_error("Engine::load received null module");

      module_ = module;
      module_->to(config_.device);
      module_->eval();

      loaded_ = true;
      prepared_ = false;
      compiled_ = false;
      stats_ = {};
      refresh_memory_stats();

      const size_t trainable_count = count_requires_grad_parameters();
      std::ostringstream message;
      message << "Module loaded on " << config_.device.to_string()
              << "; gradients will remain disabled during compile/run";
      if (trainable_count > 0) {
        message << " (" << trainable_count
                << " parameter/buffer tensor(s) still marked requires_grad)";
      }
      emit_event(EngineEventType::LoadCompleted, 0.0, {}, {}, 0,
                 message.str());
    } catch (const std::exception &e) {
      emit_event(EngineEventType::Error, 0.0, {}, {}, 0,
                 std::string("load failed: ") + e.what());
      throw;
    }
  }

  void compile(const Tensor &example_input,
               const std::vector<int> &expected_input_shape = {},
               const std::vector<int> &expected_output_shape = {}) {
    ensure_loaded();
    auto start = std::chrono::high_resolution_clock::now();
    Tensor input;
    try {
      input = prepare_input(example_input, "compile");
      emit_event(EngineEventType::CompileStarted, 0.0, input.shape(), {},
                 stats_.runs, "Compiling inference shape contract");

      detail::InferenceModeGuard no_grad;
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
      ensure_inference_output(out, "compile");
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
        Tensor warmup_out = module_->forward(input);
        ensure_inference_output(warmup_out, "warmup");
      }

      auto end = std::chrono::high_resolution_clock::now();
      stats_.compile_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      stats_.compiled_input_shape = compiled_input_shape_;
      stats_.compiled_output_shape = compiled_output_shape_;
      refresh_memory_stats();

      compiled_ = true;
      prepared_ = true;
      emit_event(EngineEventType::CompileCompleted, stats_.compile_ms,
                 compiled_input_shape_, compiled_output_shape_, stats_.runs,
                 "Inference compile completed");
    } catch (const std::exception &e) {
      emit_event(EngineEventType::Error, elapsed_ms(start),
                 shape_or_empty(input), {}, stats_.runs,
                 std::string("compile failed: ") + e.what());
      throw;
    }
  }

  void prepare(const Tensor &example_input) { compile(example_input); }

  Tensor run(const Tensor &input) {
    ensure_loaded();

    auto start = std::chrono::high_resolution_clock::now();
    Tensor in;
    try {
      in = prepare_input(input, "run");
      if (config_.strict_shape_check && compiled_) {
        if (use_expected_input_shape_) {
          validate_shape(expected_input_shape_, in.shape(),
                         "Engine: input shape mismatch with expected compiled shape");
        } else if (in.shape() != compiled_input_shape_) {
          throw std::runtime_error("Engine: input shape mismatch with compiled shape");
        }
      }

      emit_event(EngineEventType::RunStarted, 0.0, in.shape(), {}, stats_.runs + 1,
                 "Starting inference run");

      detail::InferenceModeGuard no_grad;
      Tensor out = module_->forward(in);
      ensure_inference_output(out, "run");

      if (config_.strict_shape_check && compiled_ && use_expected_output_shape_) {
        validate_shape(expected_output_shape_, out.shape(),
                       "Engine: output shape mismatch with expected compiled shape");
      }
      auto end = std::chrono::high_resolution_clock::now();

      stats_.runs += 1;
      stats_.last_run_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      refresh_memory_stats();
      emit_event(EngineEventType::RunCompleted, stats_.last_run_ms, in.shape(),
                 out.shape(), stats_.runs, "Inference run completed");

      return out;
    } catch (const std::exception &e) {
      emit_event(EngineEventType::Error, elapsed_ms(start),
                 shape_or_empty(in), {}, stats_.runs + 1,
                 std::string("run failed: ") + e.what());
      throw;
    }
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

  void validate_inference_input(const Tensor &input, const char *stage) const {
    if (!config_.allow_autograd_inputs && input.requires_grad()) {
      throw std::runtime_error(std::string("Engine: ") + stage +
                               " received a tensor with requires_grad=true. "
                               "Detach inputs or opt into allow_autograd_inputs "
                               "for debugging-only inspection paths.");
    }
  }

  Tensor prepare_input(const Tensor &input, const char *stage) const {
    Tensor prepared = to_engine_device(input);
    validate_inference_input(prepared, stage);
    return prepared;
  }

  static void ensure_inference_output(const Tensor &output, const char *stage) {
    if (output.requires_grad()) {
      throw std::runtime_error(std::string("Engine: ") + stage +
                               " produced a gradient-tracked tensor. "
                               "Inference execution must remain detached from "
                               "autograd.");
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

  void ensure_loaded() const {
    if (!loaded_ || !module_)
      throw std::runtime_error("Engine: no module loaded");
  }

  static std::vector<int> shape_or_empty(const Tensor &tensor) {
    return tensor.impl_ ? tensor.shape() : std::vector<int>{};
  }

  static double elapsed_ms(
      const std::chrono::high_resolution_clock::time_point &start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - start)
        .count();
  }

  void refresh_memory_stats() {
    if (!config_.capture_profiler_memory)
      return;
    stats_.current_memory_bytes = Profiler::get().current_memory_bytes();
    stats_.peak_memory_bytes = Profiler::get().peak_memory_bytes();
  }

  size_t count_requires_grad_parameters() const {
    if (!module_) {
      return 0;
    }

    size_t count = 0;
    for (const auto &[name, tensor] : module_->named_parameters()) {
      (void)name;
      if (tensor.requires_grad()) {
        ++count;
      }
    }
    return count;
  }

  void emit_event(EngineEventType type, double duration_ms,
                  std::vector<int> input_shape, std::vector<int> output_shape,
                  size_t run_index, std::string message) const {
    if (!observer_) {
      return;
    }

    EngineEvent event;
    event.type = type;
    event.device = config_.device;
    event.run_index = run_index;
    event.duration_ms = duration_ms;
    event.input_shape = std::move(input_shape);
    event.output_shape = std::move(output_shape);
    event.message = std::move(message);
    if (config_.capture_profiler_memory) {
      event.current_memory_bytes = Profiler::get().current_memory_bytes();
      event.peak_memory_bytes = Profiler::get().peak_memory_bytes();
    }
    observer_(event);
  }

  EngineConfig config_;
  EngineStats stats_;
  std::shared_ptr<core::Module> module_ = nullptr;
  EngineObserver observer_;
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
