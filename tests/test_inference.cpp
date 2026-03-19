#include "inference.hpp"
#include "nn.hpp"
#include "core/util/profiler.hpp"
#include "test_utils.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <type_traits>

using namespace munet;

namespace {
class ScopedProfileOverride {
public:
  explicit ScopedProfileOverride(bool enabled) {
    set_profile_enabled_override(enabled);
  }

  ~ScopedProfileOverride() { set_profile_enabled_override(std::nullopt); }
};

class IdentityLayer : public inference::Module {
public:
  Tensor forward_impl(Tensor x) override { return x; }
};

class IdentityLayerWithState : public inference::Module {
public:
  explicit IdentityLayerWithState(const TensorOptions &options) {
    weight = Tensor({1}, options.device, options.dtype, true);
    weight.fill_(1.0f);
    register_parameter("weight", weight);

    running = Tensor({1}, options.device,
                     accumulation_type(AccumulationOp::Elementwise, options.dtype),
                     false);
    running.fill_(0.0f);
    register_buffer("running", running);
  }

  Tensor forward_impl(Tensor x) override { return x; }

  Tensor weight;
  Tensor running;
};
} // namespace

TEST(InferenceTest, ModuleInheritsCoreModule) {
  EXPECT_TRUE((std::is_base_of_v<core::Module, inference::Module>));
}

TEST(InferenceTest, TrainCallKeepsEvalMode) {
  auto m = std::make_shared<IdentityLayer>();
  m->train(true);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({1, 1}, cpu);
  x.uniform_(1.0f, 1.0f);

  auto y = m->forward(x);
  EXPECT_EQ(y.size(), x.size());
}


TEST(InferenceTest, EngineLoadPrepareRunAndStats) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.set_warmup_runs(2);
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 2}, cpu);
  x.uniform_(0.5f, 0.5f);

  engine.prepare(x);
  EXPECT_TRUE(engine.is_loaded());
  EXPECT_TRUE(engine.is_prepared());

  Tensor y = engine.run(x).to(cpu);
  EXPECT_EQ(y.shape(), x.shape());

  auto stats = engine.stats();
  EXPECT_EQ(stats.runs, 1u);
  EXPECT_GE(stats.last_run_ms, 0.0);
}

TEST(InferenceTest, EngineThrowsWithoutLoad) {
  inference::Engine engine;
  Device cpu{DeviceType::CPU, 0};
  Tensor x({1, 1}, cpu);

  EXPECT_THROW(engine.run(x), std::runtime_error);
  EXPECT_THROW(engine.prepare(x), std::runtime_error);
}

TEST(InferenceTest, EngineRunBatch) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor a({1, 2}, cpu);
  Tensor b({1, 2}, cpu);
  a.uniform_(1.0f, 1.0f);
  b.uniform_(2.0f, 2.0f);

  auto outs = engine.run_batch({a, b});
  EXPECT_EQ(outs.size(), 2u);
  EXPECT_EQ(outs[0].shape(), a.shape());
  EXPECT_EQ(outs[1].shape(), b.shape());
  EXPECT_EQ(engine.stats().runs, 2u);
}


TEST(InferenceTest, EngineCompileCapturesShapeAndStats) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.set_warmup_runs(1);
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 3}, cpu);
  x.uniform_(0.2f, 0.2f);

  engine.compile(x);
  EXPECT_TRUE(engine.is_compiled());
  EXPECT_TRUE(engine.is_prepared());
  EXPECT_EQ(engine.compiled_input_shape(), x.shape());
  EXPECT_GE(engine.stats().compile_ms, 0.0);
  EXPECT_GE(engine.stats().compile_prepare_input_ms, 0.0);
  EXPECT_GE(engine.stats().compile_forward_ms, 0.0);
}

TEST(InferenceTest, EngineStrictShapeCheckAfterCompile) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 3}, cpu);
  Tensor bad({2, 4}, cpu);
  x.uniform_(0.1f, 0.1f);
  bad.uniform_(0.1f, 0.1f);

  engine.compile(x);
  EXPECT_THROW(engine.run(bad), std::runtime_error);

  engine.set_strict_shape_check(false);
  EXPECT_NO_THROW((void)engine.run(bad));
}


TEST(InferenceTest, EngineCompileWithDynamicInputShape) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x_compile({1, 3, 64, 64}, cpu);
  x_compile.uniform_(0.1f, 0.1f);

  engine.compile(x_compile, {-1, 3, -1, -1}, {-1, 3, -1, -1});

  Tensor x_ok({4, 3, 128, 256}, cpu);
  x_ok.uniform_(0.2f, 0.2f);
  EXPECT_NO_THROW((void)engine.run(x_ok));

  Tensor x_bad({4, 1, 128, 256}, cpu);
  x_bad.uniform_(0.2f, 0.2f);
  EXPECT_THROW((void)engine.run(x_bad), std::runtime_error);
}

TEST(InferenceTest, EngineCompileWithInvalidExpectedOutputShapeThrows) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 3}, cpu);
  x.uniform_(0.3f, 0.3f);

  EXPECT_THROW(engine.compile(x, {-1, 3}, {-1, 4}), std::runtime_error);
}

TEST(InferenceTest, EngineLoadPreservesModelWideDTypeConversion) {
  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float16;

  auto m = std::make_shared<IdentityLayerWithState>(options);
  inference::Engine engine;
  engine.load(m);

  EXPECT_EQ(m->weight.dtype(), DataType::Float16);
  EXPECT_EQ(m->running.dtype(), DataType::Float32);

  Tensor x({2, 2}, Device{DeviceType::CPU, 0}, DataType::Float16, false);
  x.fill_(0.5f);
  Tensor y = engine.run(x);
  EXPECT_EQ(y.dtype(), DataType::Float16);
  EXPECT_EQ(y.shape(), x.shape());
}

TEST(InferenceTest, EnginePreservesFloat16LinearOutputsAcrossAvailableDevices) {
  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float16;

  auto linear = std::make_shared<nn::Linear>(2, 2, true, options);
  Tensor weight_cpu({2, 2}, Device{DeviceType::CPU, 0}, DataType::Float32);
  Tensor bias_cpu({2}, Device{DeviceType::CPU, 0}, DataType::Float32);
  float *w = static_cast<float *>(weight_cpu.data());
  float *b = static_cast<float *>(bias_cpu.data());
  w[0] = 1.0f; w[1] = 2.0f;
  w[2] = -1.0f; w[3] = 0.5f;
  b[0] = 0.25f; b[1] = -0.75f;
  linear->weight = weight_cpu.to(DataType::Float16);
  linear->bias = bias_cpu.to(DataType::Float16);

  Tensor x32({1, 2}, Device{DeviceType::CPU, 0}, DataType::Float32);
  float *x_ptr = static_cast<float *>(x32.data());
  x_ptr[0] = 3.0f;
  x_ptr[1] = -2.0f;
  Tensor x = x32.to(DataType::Float16);

  Tensor expected = linear->forward(x).to(Device{DeviceType::CPU, 0}).to(DataType::Float32);

  for (const Device &device : test::get_available_devices()) {
    inference::Engine engine;
    engine.set_device(device);
    engine.load(linear);

    Tensor y = engine.run(x.to(device)).to(Device{DeviceType::CPU, 0}).to(DataType::Float32);
    EXPECT_EQ(y.dtype(), DataType::Float32);
    EXPECT_TRUE(test::all_close(y, expected, 2e-1f)) << device.to_string();
  }
}

TEST(InferenceTest, EngineRunDisablesAutogradGraphConstruction) {
  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float32;

  auto linear = std::make_shared<nn::Linear>(2, 2, true, options);
  inference::Engine engine;
  engine.load(linear);

  EXPECT_TRUE(linear->weight.requires_grad());
  GradMode::set_enabled(true);

  Tensor x({1, 2}, options.device, options.dtype, false);
  x.fill_(1.0f);

  Tensor y = engine.run(x);
  EXPECT_FALSE(y.requires_grad());
  EXPECT_FALSE(y.impl_ && y.impl_->grad_fn);
  EXPECT_TRUE(GradMode::is_enabled());
}

TEST(InferenceTest, EngineRejectsAutogradInputsByDefault) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 2}, cpu, DataType::Float32, true);
  x.fill_(0.25f);

  EXPECT_THROW(engine.run(x), std::runtime_error);
}

TEST(InferenceTest, EngineObserverReceivesLifecycleAndErrorEvents) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  std::vector<inference::EngineEvent> events;
  engine.set_observer([&events](const inference::EngineEvent &event) {
    events.push_back(event);
  });
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 3}, cpu);
  Tensor bad({2, 4}, cpu);
  x.uniform_(0.1f, 0.1f);
  bad.uniform_(0.1f, 0.1f);

  engine.compile(x);
  (void)engine.run(x);
  EXPECT_THROW((void)engine.run(bad), std::runtime_error);

  ASSERT_GE(events.size(), 6u);
  EXPECT_EQ(events.front().type, inference::EngineEventType::LoadStarted);
  EXPECT_EQ(events[1].type, inference::EngineEventType::LoadCompleted);
  EXPECT_EQ(events[2].type, inference::EngineEventType::CompileStarted);
  EXPECT_EQ(events[3].type, inference::EngineEventType::CompileCompleted);
  EXPECT_EQ(events[4].type, inference::EngineEventType::RunStarted);
  EXPECT_EQ(events[5].type, inference::EngineEventType::RunCompleted);
  EXPECT_GT(events[2].trace_id, 0u);
  EXPECT_EQ(events[2].trace_id, events[3].trace_id);
  EXPECT_EQ(events[2].span, "compile");
  EXPECT_EQ(events[3].span, "compile");
  EXPECT_GT(events[4].trace_id, 0u);
  EXPECT_EQ(events[4].trace_id, events[5].trace_id);
  EXPECT_NE(events[2].trace_id, events[4].trace_id);
  EXPECT_EQ(events[4].span, "run");
  EXPECT_EQ(events[5].span, "run");
  EXPECT_EQ(events[3].output_shape, x.shape());
  EXPECT_EQ(events[5].input_shape, x.shape());
  EXPECT_GE(events[5].peak_memory_bytes, events[5].current_memory_bytes);
  EXPECT_EQ(events.back().type, inference::EngineEventType::Error);
  EXPECT_GT(events.back().trace_id, 0u);
  EXPECT_EQ(events.back().span, "run");
  EXPECT_NE(events.back().message.find("run failed"), std::string::npos);
}

TEST(InferenceTest, EngineProfilesLifecyclePhasesIntoStatsAndProfiler) {
  ScopedProfileOverride profile(true);
  Profiler::get().reset();

  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.set_warmup_runs(2);
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 2}, cpu);
  x.fill_(1.0f);

  engine.compile(x);
  (void)engine.run(x);

  const auto stats = engine.stats();
  EXPECT_GT(stats.last_compile_trace_id, 0u);
  EXPECT_GT(stats.last_run_trace_id, 0u);
  EXPECT_NE(stats.last_compile_trace_id, stats.last_run_trace_id);
  EXPECT_GE(stats.load_to_device_ms, 0.0);
  EXPECT_GE(stats.load_eval_ms, 0.0);
  EXPECT_GE(stats.compile_prepare_input_ms, 0.0);
  EXPECT_GE(stats.compile_forward_ms, 0.0);
  EXPECT_GE(stats.compile_warmup_ms, 0.0);
  EXPECT_GE(stats.last_prepare_input_ms, 0.0);
  EXPECT_GE(stats.last_forward_ms, 0.0);
  EXPECT_GE(stats.last_output_validation_ms, 0.0);

  const auto snapshot = Profiler::get().snapshot();
  EXPECT_NE(snapshot.stats.find("inference.load.to_device"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.load.eval"), snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.compile.prepare_input"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.compile.forward"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.compile.warmup"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.run.prepare_input"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.run.forward"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("inference.run.validate_output"),
            snapshot.stats.end());
  const auto compile_forward = snapshot.stats.find("inference.compile.forward");
  ASSERT_NE(compile_forward, snapshot.stats.end());
  EXPECT_NE(
      compile_forward->second.last_shape.find("trace_id=" +
                                              std::to_string(stats.last_compile_trace_id)),
      std::string::npos);
  EXPECT_NE(compile_forward->second.last_shape.find("span=compile.forward"),
            std::string::npos);
  const auto run_forward = snapshot.stats.find("inference.run.forward");
  ASSERT_NE(run_forward, snapshot.stats.end());
  EXPECT_NE(run_forward->second.last_shape.find(
                "trace_id=" + std::to_string(stats.last_run_trace_id)),
            std::string::npos);
  EXPECT_NE(run_forward->second.last_shape.find("span=run.forward"),
            std::string::npos);
  const auto module_forward = snapshot.stats.find("module.root.forward");
  ASSERT_NE(module_forward, snapshot.stats.end());
  EXPECT_NE(module_forward->second.last_shape.find(
                "trace_id=" + std::to_string(stats.last_run_trace_id)),
            std::string::npos);
  EXPECT_NE(module_forward->second.last_shape.find("span=run.forward"),
            std::string::npos);
}

TEST(InferenceTest, EngineDefaultHotPathLeavesTraceIdsZeroWithoutDiagnostics) {
  auto m = std::make_shared<IdentityLayer>();
  inference::Engine engine;
  engine.load(m);

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 2}, cpu);
  x.fill_(1.0f);

  engine.compile(x);
  (void)engine.run(x);

  const auto stats = engine.stats();
  EXPECT_EQ(stats.last_compile_trace_id, 0u);
  EXPECT_EQ(stats.last_run_trace_id, 0u);
  EXPECT_EQ(stats.current_memory_bytes, 0u);
  EXPECT_EQ(stats.peak_memory_bytes, 0u);
}

TEST(InferenceTest, EngineLeanModeDisablesProfilerMemoryByDefault) {
  auto m = std::make_shared<IdentityLayer>();
  inference::EngineConfig cfg;
  cfg.lean_mode = true;
  inference::Engine engine(cfg);
  engine.load(m);

  EXPECT_TRUE(engine.lean_mode());
  EXPECT_FALSE(engine.capture_profiler_memory());

  Device cpu{DeviceType::CPU, 0};
  Tensor x({2, 2}, cpu);
  x.fill_(1.0f);

  engine.compile(x);
  (void)engine.run(x);

  const auto stats = engine.stats();
  EXPECT_EQ(stats.current_memory_bytes, 0u);
  EXPECT_EQ(stats.peak_memory_bytes, 0u);
}
