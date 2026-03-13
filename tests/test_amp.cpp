#include "amp.hpp"
#include "nn.hpp"
#include "optim.hpp"
#include <gtest/gtest.h>
#include <limits>

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



TEST(AMPTest, AutocastCastsCoreForwardOps) {
  Device cpu{DeviceType::CPU, 0};
  Tensor a({2, 2}, cpu, DataType::Float32, false);
  Tensor b({2, 2}, cpu, DataType::Float32, false);
  auto *ap = static_cast<float *>(a.data());
  auto *bp = static_cast<float *>(b.data());
  ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f; ap[3] = 4.0f;
  bp[0] = 1.0f; bp[1] = 1.0f; bp[2] = 1.0f; bp[3] = 1.0f;

  Tensor add_fp32 = a + b;
  EXPECT_EQ(add_fp32.dtype(), DataType::Float32);

  {
    amp::AutoCastGuard guard(DataType::Float16);
    Tensor add_out = a + b;
    Tensor mul_out = a * b;
    Tensor mm_out = a.matmul(b);
    Tensor relu_out = a.relu();
    Tensor loss = a.mse_loss(b);

    EXPECT_EQ(add_out.dtype(), DataType::Float16);
    EXPECT_EQ(mul_out.dtype(), DataType::Float16);
    EXPECT_EQ(mm_out.dtype(), DataType::Float16);
    EXPECT_EQ(relu_out.dtype(), DataType::Float16);
    EXPECT_EQ(loss.dtype(), DataType::Float32);
  }
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


TEST(AMPTest, FP32MasterSGDKeepsModelLowPrecision) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float16, true);
  // Float16 path currently uses fallback scalar conversion; integer proxy values.
  auto *wp = static_cast<int8_t *>(w.data());
  wp[0] = 4;

  amp::FP32MasterSGD opt({w}, 1.0f);

  Tensor grad({1}, cpu, DataType::Float16, false);
  auto *gp = static_cast<int8_t *>(grad.data());
  gp[0] = 2;
  w.impl_->grad = grad.impl_;

  opt.step();

  EXPECT_EQ(w.dtype(), DataType::Float16);
  auto *w_after = static_cast<int8_t *>(w.data());
  EXPECT_NE(w_after[0], 4);
}


TEST(AMPTest, FP32MasterAdamKeepsModelLowPrecision) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float16, true);
  auto *wp = static_cast<int8_t *>(w.data());
  wp[0] = 8;

  amp::FP32MasterAdam opt({w}, 4.0f);

  Tensor grad({1}, cpu, DataType::Float16, false);
  auto *gp = static_cast<int8_t *>(grad.data());
  gp[0] = 4;
  w.impl_->grad = grad.impl_;

  opt.step();

  EXPECT_EQ(w.dtype(), DataType::Float16);
  auto *w_after = static_cast<int8_t *>(w.data());
  EXPECT_NE(w_after[0], 8);
}

TEST(AMPTest, FP32MasterSGDStepParityAgainstFP32) {
  Device cpu{DeviceType::CPU, 0};

  Tensor w_fp32({1}, cpu, DataType::Float32, true);
  auto *w32 = static_cast<float *>(w_fp32.data());
  w32[0] = 10.0f;
  Tensor g_fp32({1}, cpu, DataType::Float32, false);
  auto *g32 = static_cast<float *>(g_fp32.data());
  g32[0] = 4.0f;
  w_fp32.impl_->grad = g_fp32.impl_;
  optim::SGD sgd_fp32({w_fp32}, 0.5f);
  sgd_fp32.step();

  Tensor w_lp({1}, cpu, DataType::Float16, true);
  auto *wlp = static_cast<int8_t *>(w_lp.data());
  wlp[0] = 10;
  Tensor g_lp({1}, cpu, DataType::Float16, false);
  auto *glp = static_cast<int8_t *>(g_lp.data());
  glp[0] = 4;
  w_lp.impl_->grad = g_lp.impl_;
  amp::FP32MasterSGD sgd_lp({w_lp}, 0.5f);
  sgd_lp.step();

  Tensor w_lp_fp32 = w_lp.to_dtype(DataType::Float32);
  EXPECT_NEAR(w_lp_fp32.item(), w_fp32.item(), 1e-5f);
}

TEST(AMPTest, FP32MasterAdamStepParityAgainstFP32) {
  Device cpu{DeviceType::CPU, 0};

  Tensor w_fp32({1}, cpu, DataType::Float32, true);
  auto *w32 = static_cast<float *>(w_fp32.data());
  w32[0] = 10.0f;
  Tensor g_fp32({1}, cpu, DataType::Float32, false);
  auto *g32 = static_cast<float *>(g_fp32.data());
  g32[0] = 4.0f;
  w_fp32.impl_->grad = g_fp32.impl_;
  optim::Adam adam_fp32({w_fp32}, 2.0f);
  adam_fp32.step();

  Tensor w_lp({1}, cpu, DataType::Float16, true);
  auto *wlp = static_cast<int8_t *>(w_lp.data());
  wlp[0] = 10;
  Tensor g_lp({1}, cpu, DataType::Float16, false);
  auto *glp = static_cast<int8_t *>(g_lp.data());
  glp[0] = 4;
  w_lp.impl_->grad = g_lp.impl_;
  amp::FP32MasterAdam adam_lp({w_lp}, 2.0f);
  adam_lp.step();

  Tensor w_lp_fp32 = w_lp.to_dtype(DataType::Float32);
  EXPECT_NEAR(w_lp_fp32.item(), w_fp32.item(), 1e-4f);
}


TEST(AMPTest, GradScalerSkipsStepAndBacksOffOnInfGradients) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float32, true);
  auto *wp = static_cast<float *>(w.data());
  wp[0] = 2.0f;

  optim::SGD opt({w}, 0.1f);
  amp::GradScaler scaler(8.0f, 2.0f, 0.5f, 2);

  Tensor grad({1}, cpu, DataType::Float32, false);
  auto *gp = static_cast<float *>(grad.data());
  gp[0] = std::numeric_limits<float>::infinity();
  w.impl_->grad = grad.impl_;

  bool stepped = scaler.step(opt, {w});
  EXPECT_FALSE(stepped);
  EXPECT_FLOAT_EQ(w.item(), 2.0f);
  EXPECT_FLOAT_EQ(scaler.current_scale(), 4.0f);
}

TEST(AMPTest, GradScalerGrowsScaleAfterConfiguredInterval) {
  Device cpu{DeviceType::CPU, 0};
  Tensor w({1}, cpu, DataType::Float32, true);
  auto *wp = static_cast<float *>(w.data());
  wp[0] = 2.0f;

  optim::SGD opt({w}, 0.1f);
  amp::GradScaler scaler(8.0f, 2.0f, 0.5f, 2);

  Tensor grad({1}, cpu, DataType::Float32, false);
  auto *gp = static_cast<float *>(grad.data());
  gp[0] = 1.0f;

  w.impl_->grad = grad.impl_;
  EXPECT_TRUE(scaler.step(opt, {w}));
  EXPECT_FLOAT_EQ(scaler.current_scale(), 8.0f);

  w.impl_->grad = grad.impl_;
  EXPECT_TRUE(scaler.step(opt, {w}));
  EXPECT_FLOAT_EQ(scaler.current_scale(), 16.0f);
}
