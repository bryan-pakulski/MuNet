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

  Tensor p_cpu({4}, {DeviceType::CPU, 0});
  Tensor t_cpu({4}, {DeviceType::CPU, 0});
  float p_vals[] = {0.0f, 1.0f, 2.0f, 3.0f};
  float t_vals[] = {0.0f, 0.0f, 2.0f, 2.0f};
  std::memcpy(p_cpu.data(), p_vals, sizeof(p_vals));
  std::memcpy(t_cpu.data(), t_vals, sizeof(t_vals));

  pred.impl_->backend().copy(p_cpu.data(), pred.data(), pred.bytes(),
                             p_cpu.device(), dev());
  target.impl_->backend().copy(t_cpu.data(), target.data(), target.bytes(),
                               t_cpu.device(), dev());

  pred.set_requires_grad(true);

  Tensor loss = pred.mse_loss(target);
  Tensor loss_cpu = loss.to({DeviceType::CPU, 0});
  EXPECT_FLOAT_EQ(static_cast<float *>(loss_cpu.data())[0], 0.5f);

  loss.backward();

  Tensor grad_cpu = pred.grad().to({DeviceType::CPU, 0});
  const float *grad_p = static_cast<const float *>(grad_cpu.data());
  EXPECT_FLOAT_EQ(grad_p[0], 0.0f);
  EXPECT_FLOAT_EQ(grad_p[1], 0.5f);
  EXPECT_FLOAT_EQ(grad_p[2], 0.0f);
  EXPECT_FLOAT_EQ(grad_p[3], 0.5f);
}

TEST_P(LossTest, CrossEntropyForwardBackward) {
  Tensor logits({2, 3}, dev());
  Tensor targets({2, 3}, dev());

  Tensor l_cpu({2, 3}, {DeviceType::CPU, 0});
  Tensor t_cpu({2, 3}, {DeviceType::CPU, 0});
  float l_vals[] = {2.0f, 1.0f, 0.1f, 0.1f, 1.0f, 2.0f};
  float t_vals[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  std::memcpy(l_cpu.data(), l_vals, sizeof(l_vals));
  std::memcpy(t_cpu.data(), t_vals, sizeof(t_vals));

  logits.impl_->backend().copy(l_cpu.data(), logits.data(), logits.bytes(),
                               l_cpu.device(), dev());
  targets.impl_->backend().copy(t_cpu.data(), targets.data(), targets.bytes(),
                                t_cpu.device(), dev());

  logits.set_requires_grad(true);

  Tensor loss = logits.cross_entropy(targets);
  Tensor loss_cpu = loss.to({DeviceType::CPU, 0});
  EXPECT_NEAR(static_cast<float *>(loss_cpu.data())[0], 0.41702f, 1e-4);

  loss.backward();

  Tensor grad_cpu = logits.grad().to({DeviceType::CPU, 0});
  const float *grad_l = static_cast<const float *>(grad_cpu.data());
  EXPECT_NEAR(grad_l[0], -0.17050f, 1e-4);
  EXPECT_NEAR(grad_l[5], -0.17050f, 1e-4);
}
