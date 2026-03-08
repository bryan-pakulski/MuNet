#include "inference.hpp"
#include <gtest/gtest.h>
#include <type_traits>

using namespace munet;

namespace {
class IdentityLayer : public inference::Module {
public:
  Tensor forward(Tensor x) override { return x; }
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
