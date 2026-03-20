#pragma once

#include "core/grad_mode.hpp"
#include "core/module.hpp"
#include "core/util/logging.hpp"
#include "core/util/profiler.hpp"
#include "core/util/timer.hpp"
#include <chrono>
#include <functional>
#include <optional>
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
  bool capture_profiler_memory = false;
  bool lean_mode = false;
  size_t prepared_input_cache_entries = 8;
  size_t prepared_input_cache_max_bytes = 64 * 1024 * 1024;
};

struct EngineStats {
  size_t runs = 0;
  uint64_t last_compile_trace_id = 0;
  uint64_t last_run_trace_id = 0;
  double load_to_device_ms = 0.0;
  double load_eval_ms = 0.0;
  double last_run_ms = 0.0;
  double last_prepare_input_ms = 0.0;
  double last_forward_ms = 0.0;
  double last_output_validation_ms = 0.0;
  double compile_ms = 0.0;
  double compile_prepare_input_ms = 0.0;
  double compile_forward_ms = 0.0;
  double compile_warmup_ms = 0.0;
  std::vector<int> compiled_input_shape{};
  std::vector<int> compiled_output_shape{};
  size_t current_memory_bytes = 0;
  size_t peak_memory_bytes = 0;
  size_t prepared_input_cache_entries = 0;
  size_t prepared_input_cache_bytes = 0;
  size_t prepared_input_cache_hits = 0;
  size_t prepared_input_cache_misses = 0;
  size_t prepared_input_cache_evictions = 0;
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
  uint64_t trace_id = 0;
  size_t run_index = 0;
  double duration_ms = 0.0;
  std::vector<int> input_shape{};
  std::vector<int> output_shape{};
  size_t current_memory_bytes = 0;
  size_t peak_memory_bytes = 0;
  std::string span{};
  std::string message{};
};

using EngineObserver = std::function<void(const EngineEvent &)>;

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

class OptionalTraceScope {
public:
  OptionalTraceScope(bool enabled, std::string span,
                     std::optional<uint64_t> trace_id = std::nullopt) {
    if (enabled) {
      scope_.emplace(std::move(span), trace_id);
    }
  }

private:
  std::optional<ScopedTraceContext> scope_;
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

struct PreparedInputCacheEntry {
  const TensorImpl *input_impl = nullptr;
  uint64_t input_version = 0;
  Device source_device{DeviceType::UNKNOWN, 0};
  Device target_device{DeviceType::UNKNOWN, 0};
  Tensor prepared{};

  void clear() {
    input_impl = nullptr;
    input_version = 0;
    source_device = Device{DeviceType::UNKNOWN, 0};
    target_device = Device{DeviceType::UNKNOWN, 0};
    prepared = {};
  }

  bool matches(const Tensor &input, Device target) const {
    return prepared.impl_ && input.impl_.get() == input_impl &&
           input.version() == input_version &&
           input.device() == source_device && target == target_device;
  }

  bool matches_identity(const Tensor &input, Device target) const {
    return input.impl_.get() == input_impl && input.device() == source_device &&
           target == target_device;
  }

  void store(const Tensor &input, Device target, const Tensor &value) {
    input_impl = input.impl_.get();
    input_version = input.version();
    source_device = input.device();
    target_device = target;
    prepared = value;
  }
};

class PreparedInputCache {
public:
  void configure(size_t max_entries, size_t max_bytes) {
    max_entries_ = max_entries;
    max_bytes_ = max_bytes;
    reset();
    entries_.reserve(max_entries_);
  }

  const Tensor *find(const Tensor &input, Device target) const {
    for (const auto &entry : entries_) {
      if (entry.matches(input, target)) {
        return &entry.prepared;
      }
    }
    return nullptr;
  }

  void reset() {
    entries_.clear();
    current_bytes_ = 0;
  }

  size_t store(const Tensor &input, Device target, const Tensor &value) {
    if (!value.impl_ || max_entries_ == 0 || max_bytes_ == 0) {
      return 0;
    }

    const size_t bytes = value.bytes();
    if (bytes > max_bytes_) {
      return 0;
    }

    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->matches_identity(input, target)) {
        current_bytes_ -= it->prepared.impl_ ? it->prepared.bytes() : 0;
        entries_.erase(it);
        break;
      }
    }

    size_t evictions = 0;
    while (!entries_.empty() && (entries_.size() >= max_entries_ ||
                                 current_bytes_ + bytes > max_bytes_)) {
      current_bytes_ -= entries_.front().prepared.impl_
                            ? entries_.front().prepared.bytes()
                            : 0;
      entries_.erase(entries_.begin());
      ++evictions;
    }

    PreparedInputCacheEntry entry;
    entry.store(input, target, value);
    entries_.push_back(std::move(entry));
    current_bytes_ += bytes;
    return evictions;
  }

  size_t size() const { return entries_.size(); }
  size_t current_bytes() const { return current_bytes_; }
  size_t max_entries() const { return max_entries_; }
  size_t max_bytes() const { return max_bytes_; }

private:
  std::vector<PreparedInputCacheEntry> entries_{};
  size_t max_entries_ = 8;
  size_t max_bytes_ = 64 * 1024 * 1024;
  size_t current_bytes_ = 0;
};
} // namespace detail

class Engine {
public:
  explicit Engine(EngineConfig cfg = {}) : config_(cfg) {
    prepared_input_cache_.configure(config_.prepared_input_cache_entries,
                                    config_.prepared_input_cache_max_bytes);
    refresh_cache_stats();
  }

  void set_device(Device device) {
    config_.device = device;
    input_shape_contract_.reset();
    output_shape_contract_.reset();
    prepared_input_cache_.reset();
    refresh_cache_stats();
    compiled_ = false;
    prepared_ = false;
  }
  Device device() const { return config_.device; }

  void set_warmup_runs(int warmup_runs) {
    if (warmup_runs < 0)
      throw std::runtime_error("Engine warmup_runs must be >= 0");
    config_.warmup_runs = warmup_runs;
  }

  void set_strict_shape_check(bool enabled) {
    config_.strict_shape_check = enabled;
  }
  void set_allow_autograd_inputs(bool enabled) {
    config_.allow_autograd_inputs = enabled;
  }
  void set_capture_profiler_memory(bool enabled) {
    config_.capture_profiler_memory = enabled;
    if (enabled) {
      config_.lean_mode = false;
    }
  }

  void set_lean_mode(bool enabled) {
    config_.lean_mode = enabled;
    if (enabled) {
      config_.capture_profiler_memory = false;
    }
  }

  bool allow_autograd_inputs() const { return config_.allow_autograd_inputs; }
  bool capture_profiler_memory() const {
    return config_.capture_profiler_memory;
  }
  bool lean_mode() const { return config_.lean_mode; }
  size_t prepared_input_cache_entries_limit() const {
    return config_.prepared_input_cache_entries;
  }
  size_t prepared_input_cache_max_bytes_limit() const {
    return config_.prepared_input_cache_max_bytes;
  }

  void set_prepared_input_cache_entries(size_t entries) {
    config_.prepared_input_cache_entries = entries;
    prepared_input_cache_.configure(config_.prepared_input_cache_entries,
                                    config_.prepared_input_cache_max_bytes);
    refresh_cache_stats();
  }

  void set_prepared_input_cache_max_bytes(size_t bytes) {
    config_.prepared_input_cache_max_bytes = bytes;
    prepared_input_cache_.configure(config_.prepared_input_cache_entries,
                                    config_.prepared_input_cache_max_bytes);
    refresh_cache_stats();
  }

  void clear_prepared_input_cache() {
    prepared_input_cache_.reset();
    refresh_cache_stats();
  }

  void prepare_batch(const std::vector<Tensor> &inputs) {
    ensure_loaded();
    for (const auto &input : inputs) {
      Tensor prepared = prepare_input(input, "prepare_batch");
      if (config_.strict_shape_check && compiled_) {
        validate_shape_contract(
            input_shape_contract_, prepared.shape(),
            "Engine: input shape mismatch with compiled shape");
      }
    }
    if (!inputs.empty()) {
      prepared_ = true;
    }
    refresh_memory_stats();
  }

  void set_observer(EngineObserver observer) {
    observer_ = std::move(observer);
  }
  void clear_observer() { observer_ = nullptr; }

  void load(const std::string &serialized_path) {
    load(load_serialized(serialized_path));
  }

  void load(const std::shared_ptr<core::Module> &module) {
    emit_event(EngineEventType::LoadStarted, 0.0, {}, {}, 0,
               "Loading module into inference engine");

    try {
      if (!module)
        throw std::runtime_error("Engine::load received null module");

      module_ = module;
      stats_ = {};
      input_shape_contract_.reset();
      output_shape_contract_.reset();
      prepared_input_cache_.reset();
      refresh_cache_stats();
      {
        Timer timer;
        if (!module_->is_on(config_.device)) {
          module_->to(config_.device);
        }
        stats_.load_to_device_ms = timer.elapsed_ms();
        record_profile_phase("inference.load.to_device",
                             stats_.load_to_device_ms, {}, 0);
      }
      {
        Timer timer;
        module_->eval();
        stats_.load_eval_ms = timer.elapsed_ms();
        record_profile_phase("inference.load.eval", stats_.load_eval_ms, {}, 0);
      }

      loaded_ = true;
      prepared_ = false;
      compiled_ = false;
      refresh_memory_stats();

      std::ostringstream message;
      message << "Module loaded on " << config_.device.to_string()
              << "; gradients will remain disabled during compile/run";
      if (!config_.lean_mode) {
        const size_t trainable_count = count_requires_grad_parameters();
        if (trainable_count > 0) {
          message << " (" << trainable_count
                  << " parameter/buffer tensor(s) still marked requires_grad)";
        }
      }
      emit_event(EngineEventType::LoadCompleted, 0.0, {}, {}, 0, message.str());
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
    const bool trace_enabled = should_enable_trace_contexts();
    const uint64_t trace_id = trace_enabled ? next_trace_id() : 0;
    stats_.last_compile_trace_id = trace_id;
    detail::OptionalTraceScope compile_trace(trace_enabled, "compile",
                                             trace_id);
    Tensor input;
    try {
      {
        detail::OptionalTraceScope phase_trace(trace_enabled, "prepare_input");
        Timer timer;
        input = prepare_input(example_input, "compile");
        stats_.compile_prepare_input_ms = timer.elapsed_ms();
        record_profile_phase("inference.compile.prepare_input",
                             stats_.compile_prepare_input_ms, input.shape(),
                             input.bytes());
      }
      emit_event(EngineEventType::CompileStarted, 0.0, input.shape(), {},
                 stats_.runs, "Compiling inference shape contract");

      detail::InferenceModeGuard no_grad;
      compiled_input_shape_ = input.shape();

      if (!expected_input_shape.empty()) {
        validate_shape(expected_input_shape, compiled_input_shape_,
                       "Engine: compile expected_input_shape mismatch");
        expected_input_shape_ = expected_input_shape;
        input_shape_contract_.compile_expected(expected_input_shape_);
      } else {
        expected_input_shape_.clear();
        input_shape_contract_.compile_exact(compiled_input_shape_);
      }

      Tensor out;
      {
        detail::OptionalTraceScope phase_trace(trace_enabled, "forward");
        Timer timer;
        out = module_->forward(input);
        stats_.compile_forward_ms = timer.elapsed_ms();
        record_profile_phase("inference.compile.forward",
                             stats_.compile_forward_ms, input.shape(),
                             input.bytes());
      }
      ensure_inference_output(out, "compile");
      compiled_output_shape_ = out.shape();

      if (!expected_output_shape.empty()) {
        validate_shape(expected_output_shape, compiled_output_shape_,
                       "Engine: compile expected_output_shape mismatch");
        expected_output_shape_ = expected_output_shape;
        output_shape_contract_.compile_expected(expected_output_shape_);
      } else {
        expected_output_shape_.clear();
        output_shape_contract_.compile_exact(compiled_output_shape_);
      }

      stats_.compile_warmup_ms = 0.0;
      {
        detail::OptionalTraceScope phase_trace(trace_enabled, "warmup");
        for (int i = 1; i < config_.warmup_runs; ++i) {
          Timer timer;
          Tensor warmup_out = module_->forward(input);
          stats_.compile_warmup_ms += timer.elapsed_ms();
          ensure_inference_output(warmup_out, "warmup");
        }
      }
      record_profile_phase("inference.compile.warmup", stats_.compile_warmup_ms,
                           input.shape(), input.bytes());

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
    const bool trace_enabled = should_enable_trace_contexts();
    const uint64_t trace_id = trace_enabled ? next_trace_id() : 0;
    stats_.last_run_trace_id = trace_id;
    detail::OptionalTraceScope run_trace(trace_enabled, "run", trace_id);
    Tensor in;
    try {
      {
        detail::OptionalTraceScope phase_trace(trace_enabled, "prepare_input");
        Timer timer;
        in = prepare_input(input, "run");
        stats_.last_prepare_input_ms = timer.elapsed_ms();
        record_profile_phase("inference.run.prepare_input",
                             stats_.last_prepare_input_ms, in.shape(),
                             in.bytes());
      }
      if (config_.strict_shape_check && compiled_) {
        validate_shape_contract(
            input_shape_contract_, in.shape(),
            "Engine: input shape mismatch with compiled shape");
      }

      emit_event(EngineEventType::RunStarted, 0.0, in.shape(), {},
                 stats_.runs + 1, "Starting inference run");

      detail::InferenceModeGuard no_grad;
      Tensor out;
      {
        detail::OptionalTraceScope phase_trace(trace_enabled, "forward");
        Timer timer;
        out = module_->forward(in);
        stats_.last_forward_ms = timer.elapsed_ms();
        record_profile_phase("inference.run.forward", stats_.last_forward_ms,
                             in.shape(), in.bytes());
      }
      {
        detail::OptionalTraceScope phase_trace(trace_enabled,
                                               "validate_output");
        Timer timer;
        ensure_inference_output(out, "run");
        stats_.last_output_validation_ms = timer.elapsed_ms();
        record_profile_phase("inference.run.validate_output",
                             stats_.last_output_validation_ms, out.shape(),
                             out.bytes());
      }

      if (config_.strict_shape_check && compiled_) {
        validate_shape_contract(
            output_shape_contract_, out.shape(),
            "Engine: output shape mismatch with compiled shape");
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
      emit_event(EngineEventType::Error, elapsed_ms(start), shape_or_empty(in),
                 {}, stats_.runs + 1, std::string("run failed: ") + e.what());
      throw;
    }
  }

  std::vector<Tensor> run_batch(const std::vector<Tensor> &inputs) {
    std::vector<Tensor> outputs;
    run_batch_into(inputs, outputs);
    return outputs;
  }

  void run_batch_into(const std::vector<Tensor> &inputs,
                      std::vector<Tensor> &outputs) {
    ensure_loaded();

    outputs.clear();
    outputs.reserve(inputs.size());
    for (const auto &in : inputs) {
      outputs.push_back(run(in));
    }
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
          "Detach inputs or opt into allow_autograd_inputs "
          "for debugging-only inspection paths.");
    }
  }

  Tensor prepare_input(const Tensor &input, const char *stage) {
    validate_inference_input(input, stage);
    if (input.device() == config_.device) {
      return input;
    }
    if (const Tensor *cached =
            prepared_input_cache_.find(input, config_.device)) {
      stats_.prepared_input_cache_hits += 1;
      refresh_cache_stats();
      return *cached;
    }
    stats_.prepared_input_cache_misses += 1;

    Tensor prepared = input.to(config_.device);
    stats_.prepared_input_cache_evictions +=
        prepared_input_cache_.store(input, config_.device, prepared);
    refresh_cache_stats();
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

  static void validate_shape_contract(const detail::ShapeContract &contract,
                                      const std::vector<int> &actual,
                                      const std::string &err_prefix) {
    if (const std::string reason = contract.mismatch_reason(actual);
        !reason.empty()) {
      throw std::runtime_error(err_prefix + ": " + reason);
    }
  }

  void ensure_loaded() const {
    if (!loaded_ || !module_)
      throw std::runtime_error("Engine: no module loaded");
  }

  static std::vector<int> shape_or_empty(const Tensor &tensor) {
    return tensor.impl_ ? tensor.shape() : std::vector<int>{};
  }

  static double
  elapsed_ms(const std::chrono::high_resolution_clock::time_point &start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - start)
        .count();
  }

  bool should_enable_trace_contexts() const {
    return observer_ || is_profile_enabled() || is_debug_enabled();
  }

  static void record_profile_phase(const std::string &name, double ms,
                                   const std::vector<int> &shape,
                                   size_t bytes) {
    if (!is_profile_enabled()) {
      return;
    }
    Profiler::get().record(name, ms * 1000.0, 0.0, bytes, to_string(shape));
  }

  void refresh_memory_stats() {
    if (!config_.capture_profiler_memory) {
      stats_.current_memory_bytes = 0;
      stats_.peak_memory_bytes = 0;
      return;
    }
    stats_.current_memory_bytes = Profiler::get().current_memory_bytes();
    stats_.peak_memory_bytes = Profiler::get().peak_memory_bytes();
  }

  void refresh_cache_stats() {
    stats_.prepared_input_cache_entries = prepared_input_cache_.size();
    stats_.prepared_input_cache_bytes = prepared_input_cache_.current_bytes();
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
    event.trace_id = current_trace_id();
    event.run_index = run_index;
    event.duration_ms = duration_ms;
    event.input_shape = std::move(input_shape);
    event.output_shape = std::move(output_shape);
    event.span = current_trace_span();
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
  detail::ShapeContract input_shape_contract_{};
  detail::ShapeContract output_shape_contract_{};
  detail::PreparedInputCache prepared_input_cache_{};
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
