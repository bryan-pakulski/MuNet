#include "tensor.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>

using namespace munet;

class LossTest : public ::testing::TestWithParam<Device> {
protected:
  Device dev() { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(AllBackends, LossTest,
                         ::testing::ValuesIn(test::get_available_devices()),
                         [](const ::testing::TestParamInfo<Device> &info) {
                           std::string name = info.param.to_string();
                           std::replace(name.begin(), name.end(), ':', '_');
                           return name;
                         });

TEST_P(LossTest, MSELossForwardBackward) {
  Tensor pred({4}, dev());
  Tensor target({4}, dev());

  Tensor p_host({4}, {DeviceType::VULKAN, 0});
  Tensor t_host({4}, {DeviceType::VULKAN, 0});
  float p_vals[] = {0.0f, 1.0f, 2.0f, 3.0f};
  float t_vals[] = {0.0f, 0.0f, 2.0f, 2.0f};
  std::memcpy(p_host.data(), p_vals, sizeof(p_vals));
  std::memcpy(t_host.data(), t_vals, sizeof(t_vals));

  pred.impl_->backend().copy(p_host.data(), pred.data(), pred.bytes(),
                             p_host.device(), dev());
  target.impl_->backend().copy(t_host.data(), target.data(), target.bytes(),
                               t_host.device(), dev());

  pred.set_requires_grad(true);

  Tensor loss = pred.mse_loss(target);
  Tensor loss_host = loss.to({DeviceType::VULKAN, 0});
  EXPECT_FLOAT_EQ(static_cast<float *>(loss_host.data())[0], 0.5f);

  loss.backward();

  Tensor grad_host = pred.grad().to({DeviceType::VULKAN, 0});
  const float *grad_p = static_cast<const float *>(grad_host.data());
  EXPECT_FLOAT_EQ(grad_p[0], 0.0f);
  EXPECT_FLOAT_EQ(grad_p[1], 0.5f);
  EXPECT_FLOAT_EQ(grad_p[2], 0.0f);
  EXPECT_FLOAT_EQ(grad_p[3], 0.5f);
}

TEST_P(LossTest, CrossEntropyForwardBackward) {
  Tensor logits({2, 3}, dev());
  Tensor targets({2, 3}, dev());

  Tensor l_host({2, 3}, {DeviceType::VULKAN, 0});
  Tensor t_host({2, 3}, {DeviceType::VULKAN, 0});
  float l_vals[] = {2.0f, 1.0f, 0.1f, 0.1f, 1.0f, 2.0f};
  float t_vals[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  std::memcpy(l_host.data(), l_vals, sizeof(l_vals));
  std::memcpy(t_host.data(), t_vals, sizeof(t_vals));

  logits.impl_->backend().copy(l_host.data(), logits.data(), logits.bytes(),
                               l_host.device(), dev());
  targets.impl_->backend().copy(t_host.data(), targets.data(), targets.bytes(),
                                t_host.device(), dev());

  logits.set_requires_grad(true);

  Tensor loss = logits.cross_entropy(targets);
  Tensor loss_host = loss.to({DeviceType::VULKAN, 0});
  EXPECT_NEAR(static_cast<float *>(loss_host.data())[0], 0.41702f, 1e-4);

  loss.backward();

  Tensor grad_host = logits.grad().to({DeviceType::VULKAN, 0});
  const float *grad_l = static_cast<const float *>(grad_host.data());
  EXPECT_NEAR(grad_l[0], -0.17050f, 1e-4);
  EXPECT_NEAR(grad_l[5], -0.17050f, 1e-4);
}

TEST_P(LossTest, CrossEntropyLoss) {
  Tensor logits({1, 3}, dev(), DataType::Float32, true);
  Tensor target({1, 3}, dev());

  // Fill Host then copy to ensure exact values
  Tensor l_host({1, 3}, {DeviceType::VULKAN, 0});
  float *l_ptr = (float *)l_host.data();
  l_ptr[0] = 0.1f;
  l_ptr[1] = 0.2f;
  l_ptr[2] = 0.7f;

  Tensor t_host({1, 3}, {DeviceType::VULKAN, 0});
  float *t_ptr = (float *)t_host.data();
  t_ptr[0] = 0.0f;
  t_ptr[1] = 0.0f;
  t_ptr[2] = 1.0f;

  logits.impl_->backend().copy(l_host.data(), logits.data(), logits.bytes(),
                               l_host.device(), dev());
  target.impl_->backend().copy(t_host.data(), target.data(), target.bytes(),
                               t_host.device(), dev());

  Tensor loss = logits.cross_entropy(target);
  loss.backward();

  Tensor loss_host = loss.to({DeviceType::VULKAN, 0});
  // -log(exp(0.7) / (exp(0.1)+exp(0.2)+exp(0.7)))
  EXPECT_NEAR(((float *)loss_host.data())[0], 0.7679f, 1e-3);
  EXPECT_TRUE(logits.has_grad());
}

TEST(LossDTypeParityTest, Int8LossIsRejectedAsNonFloating) {
  Device host{DeviceType::VULKAN, 0};
  Tensor logits({2, 3}, host, DataType::Int8);
  Tensor targets({2, 3}, host, DataType::Int8);
  EXPECT_THROW((void)logits.cross_entropy(targets), std::runtime_error);
}
