#include "../src/tensor.hpp"
#include <cstring>
#include <gtest/gtest.h>

using namespace munet;

TEST(LossTest, MSELossForwardBackward) {
  Tensor pred({4}, Device{DeviceType::CPU, 0});
  Tensor target({4}, Device{DeviceType::CPU, 0});

  float p_data[] = {0.0f, 1.0f, 2.0f, 3.0f};
  float t_data[] = {0.0f, 0.0f, 2.0f, 2.0f};
  std::memcpy(pred.data(), p_data, sizeof(p_data));
  std::memcpy(target.data(), t_data, sizeof(t_data));

  pred.set_requires_grad(true);

  // MSE = sum((pred - target)^2) / N = (0 + 1 + 0 + 1) / 4 = 0.5
  Tensor loss = pred.mse_loss(target);
  float loss_val = static_cast<float *>(loss.data())[0];
  EXPECT_FLOAT_EQ(loss_val, 0.5f);

  loss.backward();

  // dL/dp = 2 * (pred - target) / N = 2/4 * [0, 1, 0, 1] = [0, 0.5, 0, 0.5]
  const float *grad_p = static_cast<const float *>(pred.grad().data());
  EXPECT_FLOAT_EQ(grad_p[0], 0.0f);
  EXPECT_FLOAT_EQ(grad_p[1], 0.5f);
  EXPECT_FLOAT_EQ(grad_p[2], 0.0f);
  EXPECT_FLOAT_EQ(grad_p[3], 0.5f);
}

TEST(LossTest, CrossEntropyForwardBackward) {
  Tensor logits({2, 3}, Device{DeviceType::CPU, 0});
  Tensor targets({2, 3}, Device{DeviceType::CPU, 0});

  float l_data[] = {2.0f, 1.0f, 0.1f, 0.1f, 1.0f, 2.0f};
  float t_data[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  std::memcpy(logits.data(), l_data, sizeof(l_data));
  std::memcpy(targets.data(), t_data, sizeof(t_data));

  logits.set_requires_grad(true);

  Tensor loss = logits.cross_entropy(targets);
  float loss_val = static_cast<float *>(loss.data())[0];

  // Values verified against PyTorch equivalent
  EXPECT_NEAR(loss_val, 0.41702f, 1e-4);

  loss.backward();

  const float *grad_l = static_cast<const float *>(logits.grad().data());
  EXPECT_NEAR(grad_l[0], -0.17049f, 1e-4);
  EXPECT_NEAR(grad_l[1], 0.12101f, 1e-4);
  EXPECT_NEAR(grad_l[2], 0.04948f, 1e-4);

  EXPECT_NEAR(grad_l[3], 0.04948f, 1e-4);
  EXPECT_NEAR(grad_l[4], 0.12101f, 1e-4);
  EXPECT_NEAR(grad_l[5], -0.17049f, 1e-4);
}
