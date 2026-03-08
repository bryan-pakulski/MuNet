#include "nn.hpp"
#include "test_utils.hpp"
#include <cmath>
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

TEST(NNTest, DropoutTrainEvalBehavior) {
  Device cpu{DeviceType::CPU, 0};
  nn::Dropout dropout(0.5f);

  Tensor x({2000}, cpu);
  x.uniform_(1.0f, 1.0f);

  dropout.train(true);
  Tensor y_train = dropout.forward(x).to(cpu);
  const float *train_data = static_cast<const float *>(y_train.data());

  int zeros = 0;
  int twos = 0;
  float sum = 0.0f;
  for (int i = 0; i < 2000; ++i) {
    sum += train_data[i];
    if (std::abs(train_data[i]) < 1e-6f)
      ++zeros;
    if (std::abs(train_data[i] - 2.0f) < 1e-6f)
      ++twos;
  }

  EXPECT_GT(zeros, 0);
  EXPECT_GT(twos, 0);
  EXPECT_NEAR(sum / 2000.0f, 1.0f, 0.15f);

  dropout.eval();
  Tensor y_eval = dropout.forward(x).to(cpu);
  const float *eval_data = static_cast<const float *>(y_eval.data());
  for (int i = 0; i < 2000; ++i) {
    EXPECT_NEAR(eval_data[i], 1.0f, 1e-6f);
  }
}


TEST(NNTest, EmbeddingForwardOneHot) {
  Device cpu{DeviceType::CPU, 0};
  nn::Embedding emb(4, 3);

  // Make weights deterministic: rows are basis-like vectors.
  float *w = static_cast<float *>(emb.weight.data());
  // row 0
  w[0] = 1.0f; w[1] = 0.0f; w[2] = 0.0f;
  // row 1
  w[3] = 0.0f; w[4] = 1.0f; w[5] = 0.0f;
  // row 2
  w[6] = 0.0f; w[7] = 0.0f; w[8] = 1.0f;
  // row 3
  w[9] = 1.0f; w[10] = 1.0f; w[11] = 1.0f;

  // x: [B=1, T=2, V=4], tokens [2, 3]
  Tensor x({1, 2, 4}, cpu);
  float *xd = static_cast<float *>(x.data());
  for (int i = 0; i < 8; ++i)
    xd[i] = 0.0f;
  xd[2] = 1.0f; // token 2 at t=0
  xd[7] = 1.0f; // token 3 at t=1

  Tensor y = emb.forward(x).to(cpu);
  EXPECT_EQ(y.shape()[0], 1);
  EXPECT_EQ(y.shape()[1], 2);
  EXPECT_EQ(y.shape()[2], 3);

  const float *yd = static_cast<const float *>(y.data());
  // token 2 -> [0,0,1]
  EXPECT_NEAR(yd[0], 0.0f, 1e-6f);
  EXPECT_NEAR(yd[1], 0.0f, 1e-6f);
  EXPECT_NEAR(yd[2], 1.0f, 1e-6f);
  // token 3 -> [1,1,1]
  EXPECT_NEAR(yd[3], 1.0f, 1e-6f);
  EXPECT_NEAR(yd[4], 1.0f, 1e-6f);
  EXPECT_NEAR(yd[5], 1.0f, 1e-6f);
}
