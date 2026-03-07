#include "nn.hpp"
#include "test_utils.hpp"
#include <gtest/gtest.h>

using namespace munet;

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
