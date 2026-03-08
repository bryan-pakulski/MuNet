#include "nn.hpp"
#include "test_utils.hpp"
#include <gtest/gtest.h>
#include <type_traits>

using namespace munet;

TEST(NNTest, ModuleInheritsCoreModule) {
  EXPECT_TRUE((std::is_base_of_v<core::Module, nn::Module>));
}

TEST(NNTest, ModuleParameters) {
  auto model = std::make_shared<nn::Sequential>();
  model->add(std::make_shared<nn::Linear>(10, 5));
  model->add(std::make_shared<nn::ReLU>());
  model->add(std::make_shared<nn::Linear>(5, 2));

  auto params = model->parameters();
  // 2 linear layers: (W, b) + (W, b) = 4 tensors
  EXPECT_EQ(params.size(), 4);

  auto named = model->named_parameters();
  EXPECT_TRUE(named.count("0.weight"));
  EXPECT_TRUE(named.count("0.bias"));
  EXPECT_TRUE(named.count("2.weight"));
  EXPECT_TRUE(named.count("2.bias"));

  auto named_modules = model->named_modules_typed();
  EXPECT_TRUE(named_modules.count("0"));
  EXPECT_TRUE(named_modules.count("1"));
  EXPECT_TRUE(named_modules.count("2"));
}

TEST(NNTest, BatchNormTrainEval) {
  Device cpu{DeviceType::CPU, 0};
  auto bn = std::make_shared<nn::BatchNorm2d>(1);
  bn->to(cpu);

  Tensor x({2, 1, 2, 2}, cpu);
  x.uniform_(10.0f, 10.0f); // All 10s

  // Training mode: running mean should update from 0
  bn->train(true);
  auto out1 = bn->forward(x);

  float rm = ((float *)bn->running_mean.data())[0];
  EXPECT_GT(rm, 0.0f);

  // Eval mode: should use running mean, output shouldn't be zero-centered if
  // stats differ
  bn->eval();
  float prev_rm = rm;
  auto out2 = bn->forward(x);
  EXPECT_FLOAT_EQ(((float *)bn->running_mean.data())[0], prev_rm);
}


TEST(NNTest, TanhForwardRange) {
  Device cpu{DeviceType::CPU, 0};
  nn::Tanh tanh;

  Tensor x({4}, cpu);
  x.uniform_(-2.0f, 2.0f);

  Tensor y = tanh.forward(x);
  Tensor y_cpu = y.to(cpu);
  const float *data = static_cast<const float *>(y_cpu.data());

  for (int i = 0; i < 4; ++i) {
    EXPECT_LE(data[i], 1.0f);
    EXPECT_GE(data[i], -1.0f);
  }
}


TEST(NNTest, LeakyReLUForwardBehavior) {
  Device cpu{DeviceType::CPU, 0};
  nn::LeakyReLU lrelu(0.1f);

  Tensor x({2}, cpu);
  float *x_data = static_cast<float *>(x.data());
  x_data[0] = -2.0f;
  x_data[1] = 3.0f;

  Tensor y = lrelu.forward(x).to(cpu);
  const float *d = static_cast<const float *>(y.data());
  EXPECT_NEAR(d[0], -0.2f, 1e-5f);
  EXPECT_NEAR(d[1], 3.0f, 1e-5f);
}


TEST(NNTest, GlobalAvgPool2dForward) {
  Device cpu{DeviceType::CPU, 0};
  nn::GlobalAvgPool2d gap;

  Tensor x({1, 1, 2, 2}, cpu);
  float *d = static_cast<float *>(x.data());
  d[0] = 1.0f; d[1] = 2.0f; d[2] = 3.0f; d[3] = 4.0f;

  Tensor y = gap.forward(x).to(cpu);
  EXPECT_EQ(y.shape()[0], 1);
  EXPECT_EQ(y.shape()[1], 1);
  EXPECT_EQ(y.shape()[2], 1);
  EXPECT_EQ(y.shape()[3], 1);

  const float *yo = static_cast<const float *>(y.data());
  EXPECT_NEAR(yo[0], 2.5f, 1e-4f);
}
