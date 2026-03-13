#include "amp.hpp"
#include "nn.hpp"
#include "optim.hpp"
#include <gtest/gtest.h>

using namespace munet;

TEST(AMPTest, AutocastGuardTogglesState) {
  EXPECT_FALSE(amp::AutocastMode::is_enabled());
  {
    amp::AutoCastGuard guard(DataType::BFloat16);
    EXPECT_TRUE(amp::AutocastMode::is_enabled());
    EXPECT_EQ(amp::AutocastMode::dtype(), DataType::BFloat16);
  }
  EXPECT_FALSE(amp::AutocastMode::is_enabled());
}

TEST(AMPTest, GradScalerScalesLossAndSteps) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float32, true);
  w.uniform_(2.0f, 2.0f);

  optim::SGD opt({w}, 0.1f);
  amp::GradScaler scaler(8.0f, 2.0f, 0.5f, 1);

  Tensor x({1}, cpu, DataType::Float32, false);
  x.uniform_(1.0f, 1.0f);

  Tensor y = w * x;
  Tensor target({1}, cpu, DataType::Float32, false);
  target.uniform_(0.0f, 0.0f);

  Tensor loss = y.mse_loss(target);
  Tensor scaled = scaler.scale(loss);
  scaled.backward();

  bool stepped = scaler.step(opt, {w});
  EXPECT_TRUE(stepped);
  EXPECT_LT(w.item(), 2.0f);
  EXPECT_GE(scaler.current_scale(), 8.0f);
}
