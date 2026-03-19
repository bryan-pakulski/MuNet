#include "optim.hpp"
#include "tensor.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <gtest/gtest.h>

using namespace munet;

class OptimTest : public ::testing::TestWithParam<Device> {
protected:
  Device dev() { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(AllBackends, OptimTest,
                         ::testing::ValuesIn(test::get_available_devices()),
                         [](const ::testing::TestParamInfo<Device> &info) {
                           std::string name = info.param.to_string();
                           std::replace(name.begin(), name.end(), ':', '_');
                           return name;
                         });

TEST_P(OptimTest, AdamConvergence) {
  // Minimize f(x) = x^2. Optimal x = 0.
  Tensor x({1}, dev(), DataType::Float32, true);

  // Initialize to 2.0 (ensures convergence within 50 steps at LR=0.1)
  Tensor init({1}, {DeviceType::CPU, 0});
  ((float *)init.data())[0] = 2.0f;
  x.impl_->backend().copy(init.data(), x.data(), x.bytes(), init.device(),
                          dev());

  auto opt = std::make_shared<optim::Adam>(std::vector<Tensor>{x}, 1e-1f);

  float last_val = 2.1f;
  for (int i = 0; i < 50; ++i) {
    opt->zero_grad();

    // y = x * x (Grad is 2x)
    Tensor y = x * x;
    y.backward();

    opt->step();

    float current_val = std::abs(x.item());
    // Allow slight tolerance for momentum-based fluctuations
    ASSERT_TRUE(current_val <= last_val + 0.05f)
        << "Adam diverged at step " << i;
    last_val = current_val;
  }

  // Should be very close to 0 after 50 steps
  EXPECT_NEAR(x.item(), 0.0f, 1e-1);
}

TEST_P(OptimTest, AdamStepConsistency) {
  // Verify exactly one step of Adam
  // theta = 1.0, grad = 0.1, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, step=1
  // theta_new = 0.9

  Tensor w({1}, dev(), DataType::Float32, true);
  Tensor init({1}, {DeviceType::CPU, 0});
  ((float *)init.data())[0] = 1.0f;
  w.impl_->backend().copy(init.data(), w.data(), w.bytes(), init.device(),
                          dev());

  auto opt = std::make_shared<optim::Adam>(std::vector<Tensor>{w}, 0.1f);

  // Create a gradient of exactly 0.1 by calculating it through a graph
  Tensor scale({1}, dev());
  ((float *)init.data())[0] = 0.1f;
  scale.impl_->backend().copy(init.data(), scale.data(), scale.bytes(),
                              init.device(), dev());

  Tensor loss = w * scale;
  loss.backward(); // w.grad becomes 0.1

  opt->step();

  EXPECT_NEAR(w.item(), 0.9f, 1e-5);
}

TEST_P(OptimTest, SGDConvergence) {
  // Minimize f(x) = x^2. Optimal x = 0.
  Tensor x({1}, dev(), DataType::Float32, true);

  Tensor init({1}, {DeviceType::CPU, 0});
  ((float *)init.data())[0] = 1.0f;
  x.impl_->backend().copy(init.data(), x.data(), x.bytes(), init.device(),
                          dev());

  auto opt = std::make_shared<optim::SGD>(std::vector<Tensor>{x}, 0.1f);

  for (int i = 0; i < 50; ++i) {
    opt->zero_grad();
    Tensor y = x * x; // dy/dx = 2x
    y.backward();
    opt->step();
  }

  // w_new = w_old * (1 - 2*lr)^n
  // (1 - 0.2)^50 is effectively 0
  EXPECT_NEAR(x.item(), 0.0f, 1e-3);
}

TEST_P(OptimTest, SGDStepConsistency) {
  // theta = 1.0, grad = 0.5, lr=0.1
  // theta_new = 1.0 - (0.1 * 0.5) = 0.95

  Tensor w({1}, dev(), DataType::Float32, true);
  Tensor init({1}, {DeviceType::CPU, 0});
  ((float *)init.data())[0] = 1.0f;
  w.impl_->backend().copy(init.data(), w.data(), w.bytes(), init.device(),
                          dev());

  auto opt = std::make_shared<optim::SGD>(std::vector<Tensor>{w}, 0.1f);

  Tensor scale({1}, dev());
  ((float *)init.data())[0] = 0.5f;
  scale.impl_->backend().copy(init.data(), scale.data(), scale.bytes(),
                              init.device(), dev());

  Tensor loss = w * scale;
  loss.backward();

  opt->step();

  EXPECT_NEAR(w.item(), 0.95f, 1e-6);
}

TEST(OptimPolicyTest, AdamSupportsFloat16ParametersUsingTypedStateFallback) {
  Device cpu{DeviceType::CPU, 0};
  Tensor x({1}, cpu, DataType::Float16, true);
  x.fill_(1.0f);

  auto opt = std::make_shared<optim::Adam>(std::vector<Tensor>{x}, 0.1f);
  Tensor grad({1}, cpu, DataType::Float16, false);
  grad.fill_(0.5f);

  for (int i = 0; i < 8; ++i) {
    x.impl_->grad = grad.impl_;
    opt->step();
  }

  ScalarValue updated = x.item_value();
  EXPECT_EQ(updated.dtype, DataType::Float16);
  EXPECT_LT(updated.as_float(), 1.0f);
}
