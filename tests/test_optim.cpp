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


TEST(OptimPolicyTest, AdamParameterGroupsExposeStatePolicies) {
  Device cpu{DeviceType::CPU, 0};
  Tensor low_precision({1}, cpu, DataType::Float16, true);
  low_precision.fill_(1.0f);
  Tensor full_precision({1}, cpu, DataType::Float32, true);
  full_precision.fill_(1.0f);

  optim::OptimizerStatePolicy low_precision_policy;
  low_precision_policy.model_dtype = DataType::Float16;
  low_precision_policy.master_weight_dtype =
      optim::MasterWeightDTypePolicy::Float32;
  low_precision_policy.state_tensor_dtype =
      optim::OptimizerStateTensorDTypePolicy::Float32;

  optim::ParameterGroup low_precision_group({low_precision}, 0.05f,
                                            low_precision_policy,
                                            "low_precision");
  optim::ParameterGroup baseline_group({full_precision}, std::nullopt, {},
                                       "baseline");

  optim::Adam opt({low_precision_group, baseline_group}, 0.01f);

  ASSERT_EQ(opt.parameter_groups().size(), 2u);
  EXPECT_FLOAT_EQ(opt.parameter_groups()[0].lr.value(), 0.05f);
  EXPECT_FALSE(opt.parameter_groups()[1].lr.has_value());
  EXPECT_EQ(opt.state_dtype_for_parameter(0), DataType::Float32);
  EXPECT_TRUE(opt.has_master_weight_for_parameter(0));
  EXPECT_EQ(opt.master_weight_dtype_for_parameter(0), DataType::Float32);
  EXPECT_EQ(opt.state_dtype_for_parameter(1), DataType::Float32);
  EXPECT_FALSE(opt.has_master_weight_for_parameter(1));
}

TEST(OptimPolicyTest, AdamParameterGroupsUsePerGroupLearningRates) {
  Device cpu{DeviceType::CPU, 0};
  Tensor slow({1}, cpu, DataType::Float32, true);
  slow.fill_(1.0f);
  Tensor fast({1}, cpu, DataType::Float32, true);
  fast.fill_(1.0f);

  Tensor grad({1}, cpu, DataType::Float32, false);
  grad.fill_(1.0f);
  slow.impl_->grad = grad.impl_;
  fast.impl_->grad = grad.impl_;

  optim::ParameterGroup slow_group({slow}, 0.01f, {}, "slow");
  optim::ParameterGroup fast_group({fast}, 0.1f, {}, "fast");
  optim::Adam opt({slow_group, fast_group}, 1e-3f);
  opt.step();

  EXPECT_GT(slow.item(), fast.item());
}

TEST(OptimPolicyTest, GradScalerScalesAndUnscalesOptimizerGradients) {
  Device cpu{DeviceType::CPU, 0};
  Tensor param({1}, cpu, DataType::Float32, true);
  param.fill_(1.0f);
  Tensor grad({1}, cpu, DataType::Float32, false);
  grad.fill_(8.0f);
  param.impl_->grad = grad.impl_;

  optim::SGD opt(std::vector<Tensor>{param}, 0.1f);
  amp::GradScaler scaler(true, 8.0f, 2.0f, 0.5f, 2);
  scaler.unscale_(opt);

  EXPECT_NEAR(param.grad().item(), 1.0f, 1e-6f);

  scaler.update();
  EXPECT_FLOAT_EQ(scaler.scale_value(), 8.0f);
  scaler.update();
  EXPECT_FLOAT_EQ(scaler.scale_value(), 16.0f);
  scaler.update({true});
  EXPECT_FLOAT_EQ(scaler.scale_value(), 8.0f);
}

TEST(OptimPolicyTest, GradScalerScaleReturnsScaledLossTensor) {
  Device cpu{DeviceType::CPU, 0};
  Tensor loss({1}, cpu, DataType::Float32, false);
  loss.fill_(2.0f);

  amp::GradScaler scaler(true, 4.0f);
  Tensor scaled = scaler.scale(loss);
  EXPECT_NEAR(scaled.item(), 8.0f, 1e-6f);
}

TEST(OptimPolicyTest, AutocastGuardRestoresPreviousStateAndPolicies) {
  EXPECT_FALSE(amp::autocast_enabled());

  amp::AutocastOptions options;
  options.enabled = true;
  options.device_type = DeviceType::CUDA;
  options.compute_dtype = DataType::Float16;
  options.conversion_policy = amp::AutocastConversionPolicy::PromoteInputs;

  {
    amp::AutocastGuard guard(options);
    EXPECT_TRUE(amp::autocast_enabled());
    EXPECT_TRUE(amp::allows_implicit_conversion(DataType::Float32,
                                                DataType::Float16));
    EXPECT_FALSE(amp::allows_output_conversion(DataType::Float32,
                                               DataType::Float16));
  }

  EXPECT_FALSE(amp::autocast_enabled());
}
