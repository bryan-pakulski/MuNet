#pragma once

#include "core/module.hpp"
#include <chrono>
#include <stdexcept>

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
};

struct EngineStats {
  size_t runs = 0;
  double last_run_ms = 0.0;
};

class Engine {
public:
  explicit Engine(EngineConfig cfg = {}) : config_(cfg) {}

  void set_device(Device device) { config_.device = device; }
  Device device() const { return config_.device; }

  void set_warmup_runs(int warmup_runs) {
    if (warmup_runs < 0)
      throw std::runtime_error("Engine warmup_runs must be >= 0");
    config_.warmup_runs = warmup_runs;
  }

  void load(const std::shared_ptr<core::Module> &module) {
    if (!module)
      throw std::runtime_error("Engine::load received null module");

    module_ = module;
    module_->to(config_.device);
    module_->eval();

    loaded_ = true;
    prepared_ = false;
    stats_ = {};
  }

  void prepare(const Tensor &example_input) {
    ensure_loaded();

    Tensor input = (example_input.device() == config_.device)
                       ? example_input
                       : example_input.to(config_.device);

    for (int i = 0; i < config_.warmup_runs; ++i) {
      (void)module_->forward(input);
    }

    prepared_ = true;
  }

  Tensor run(const Tensor &input) {
    ensure_loaded();

    Tensor in = (input.device() == config_.device) ? input : input.to(config_.device);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor out = module_->forward(in);
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
  EngineStats stats() const { return stats_; }

private:
  void ensure_loaded() const {
    if (!loaded_ || !module_)
      throw std::runtime_error("Engine: no module loaded");
  }

  EngineConfig config_;
  EngineStats stats_;
  std::shared_ptr<core::Module> module_ = nullptr;
  bool loaded_ = false;
  bool prepared_ = false;
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
