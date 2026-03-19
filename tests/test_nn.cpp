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

TEST(NNTest, ModuleToMovesParametersAndBuffers) {
  auto bn = std::make_shared<nn::BatchNorm2d>(3);
  Device target = test::get_available_devices().front();

  bn->to(target);

  EXPECT_EQ(bn->weight.device(), target);
  EXPECT_EQ(bn->bias.device(), target);
  EXPECT_EQ(bn->running_mean.device(), target);
  EXPECT_EQ(bn->running_var.device(), target);
  EXPECT_TRUE(bn->weight.requires_grad());
  EXPECT_TRUE(bn->bias.requires_grad());
  EXPECT_FALSE(bn->running_mean.requires_grad());
  EXPECT_FALSE(bn->running_var.requires_grad());
}

TEST(NNTest, ModuleOptionsControlParameterAndBufferDTypes) {
  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float16;

  nn::Linear linear(4, 2, true, options);
  EXPECT_EQ(linear.weight.dtype(), DataType::Float16);
  EXPECT_EQ(linear.bias.dtype(), DataType::Float16);
  EXPECT_TRUE(linear.weight.requires_grad());
  EXPECT_TRUE(linear.bias.requires_grad());

  nn::BatchNorm2d bn(3, 1e-5f, 0.1f, options);
  EXPECT_EQ(bn.weight.dtype(), DataType::Float16);
  EXPECT_EQ(bn.bias.dtype(), DataType::Float16);
  EXPECT_EQ(bn.running_mean.dtype(), DataType::Float32);
  EXPECT_EQ(bn.running_var.dtype(), DataType::Float32);
  EXPECT_FALSE(bn.running_mean.requires_grad());
  EXPECT_FALSE(bn.running_var.requires_grad());
}

TEST(NNTest, ModuleDefaultOptionsFollowConstructionAndMigration) {
  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float16;

  nn::BatchNorm2d bn(3, 1e-5f, 0.1f, options);
  EXPECT_EQ(bn.default_options().device, options.device);
  EXPECT_EQ(bn.default_options().dtype, DataType::Float16);
  EXPECT_EQ(bn.running_mean.dtype(), DataType::Float32);

  bn.to(DataType::Float32);
  EXPECT_EQ(bn.default_options().dtype, DataType::Float32);
  EXPECT_EQ(bn.weight.dtype(), DataType::Float32);
  EXPECT_EQ(bn.bias.dtype(), DataType::Float32);
  EXPECT_EQ(bn.running_mean.dtype(), DataType::Float32);
  EXPECT_EQ(bn.running_var.dtype(), DataType::Float32);

  bn.to(DataType::Float16);
  EXPECT_EQ(bn.default_options().dtype, DataType::Float16);
  EXPECT_EQ(bn.weight.dtype(), DataType::Float16);
  EXPECT_EQ(bn.bias.dtype(), DataType::Float16);
  EXPECT_EQ(bn.running_mean.dtype(), DataType::Float32);
  EXPECT_EQ(bn.running_var.dtype(), DataType::Float32);
}

TEST(NNTest, ParentModuleDefaultsPropagateToNestedModules) {
  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float16;

  auto seq = std::make_shared<nn::Sequential>(options);
  seq->add(std::make_shared<nn::Linear>(4, 3));
  seq->add(std::make_shared<nn::Linear>(3, 2));

  auto named = seq->named_parameters();
  EXPECT_EQ(named.at("0.weight").dtype(), DataType::Float16);
  EXPECT_EQ(named.at("0.bias").dtype(), DataType::Float16);
  EXPECT_EQ(named.at("1.weight").dtype(), DataType::Float16);
  EXPECT_EQ(named.at("1.bias").dtype(), DataType::Float16);
}

TEST(NNTest, ModuleToSupportsDTypeAndTensorOptionsConversions) {
  auto bn = std::make_shared<nn::BatchNorm2d>(3);

  bn->to(DataType::Float16);
  EXPECT_EQ(bn->weight.dtype(), DataType::Float16);
  EXPECT_EQ(bn->bias.dtype(), DataType::Float16);
  EXPECT_EQ(bn->running_mean.dtype(), DataType::Float32);
  EXPECT_EQ(bn->running_var.dtype(), DataType::Float32);
  EXPECT_TRUE(bn->weight.requires_grad());
  EXPECT_TRUE(bn->bias.requires_grad());
  EXPECT_FALSE(bn->running_mean.requires_grad());
  EXPECT_FALSE(bn->running_var.requires_grad());

  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float32;
  options.requires_grad = false;
  bn->to(options);

  EXPECT_EQ(bn->weight.dtype(), DataType::Float32);
  EXPECT_EQ(bn->bias.dtype(), DataType::Float32);
  EXPECT_EQ(bn->running_mean.dtype(), DataType::Float32);
  EXPECT_EQ(bn->running_var.dtype(), DataType::Float32);
  EXPECT_TRUE(bn->weight.requires_grad());
  EXPECT_TRUE(bn->bias.requires_grad());
  EXPECT_FALSE(bn->running_mean.requires_grad());
  EXPECT_FALSE(bn->running_var.requires_grad());
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

TEST(NNTest, DropoutSupportsFloat16ViaTypedMask) {
  Device cpu{DeviceType::CPU, 0};
  nn::Dropout dropout(0.25f);
  Tensor x32({16}, cpu, DataType::Float32);
  x32.fill_(1.0f);
  Tensor x = x32.to(DataType::Float16);

  dropout.train(true);
  Tensor y = dropout.forward(x);
  EXPECT_EQ(y.dtype(), DataType::Float16);
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


TEST(NNTest, GELUForwardBehavior) {
  Device cpu{DeviceType::CPU, 0};
  nn::GELU gelu;

  Tensor x({3}, cpu);
  float *d = static_cast<float *>(x.data());
  d[0] = -2.0f;
  d[1] = 0.0f;
  d[2] = 2.0f;

  Tensor y = gelu.forward(x).to(cpu);
  const float *yo = static_cast<const float *>(y.data());

  // Approximate expected values for x*sigmoid(1.702*x)
  EXPECT_NEAR(yo[0], -0.0643f, 5e-3f);
  EXPECT_NEAR(yo[1], 0.0f, 1e-6f);
  EXPECT_NEAR(yo[2], 1.9357f, 5e-3f);
}


TEST(NNTest, LayerNormFloat16BackwardUsesTypedFallback) {
  Device cpu{DeviceType::CPU, 0};
  TensorOptions options;
  options.device = cpu;
  options.dtype = DataType::Float16;
  nn::LayerNorm ln(4, 1e-5f, options);

  Tensor x32({2, 4}, cpu, DataType::Float32, true);
  float *xd = static_cast<float *>(x32.data());
  xd[0] = 1.0f; xd[1] = 2.0f; xd[2] = 3.0f; xd[3] = 4.0f;
  xd[4] = -1.0f; xd[5] = 0.0f; xd[6] = 1.0f; xd[7] = 2.0f;

  Tensor x = x32.to(DataType::Float16).detach();
  x.set_requires_grad(true);
  Tensor y = ln.forward(x);
  EXPECT_EQ(y.dtype(), DataType::Float16);

  Tensor loss = y.sum();
  EXPECT_NO_THROW(loss.backward());
  EXPECT_TRUE(x.has_grad());
  EXPECT_TRUE(ln.weight.has_grad());
  EXPECT_TRUE(ln.bias.has_grad());

  Tensor x_grad = x.grad().to(DataType::Float32);
  Tensor w_grad = ln.weight.grad().to(DataType::Float32);
  Tensor b_grad = ln.bias.grad().to(DataType::Float32);
  const float *xg = static_cast<const float *>(x_grad.data());
  const float *wg = static_cast<const float *>(w_grad.data());
  const float *bg = static_cast<const float *>(b_grad.data());
  for (int i = 0; i < 8; ++i)
    EXPECT_TRUE(std::isfinite(xg[i]));
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isfinite(wg[i]));
    EXPECT_TRUE(std::isfinite(bg[i]));
  }
}

TEST(NNTest, LayerNormForwardAndBackward) {
  Device cpu{DeviceType::CPU, 0};
  nn::LayerNorm ln(4);

  Tensor x({2, 4}, cpu, DataType::Float32, true);
  float *xd = static_cast<float *>(x.data());
  // row 0
  xd[0] = 1.0f; xd[1] = 2.0f; xd[2] = 3.0f; xd[3] = 4.0f;
  // row 1
  xd[4] = -1.0f; xd[5] = 0.0f; xd[6] = 1.0f; xd[7] = 2.0f;

  Tensor y = ln.forward(x).to(cpu);
  const float *yd = static_cast<const float *>(y.data());

  for (int r = 0; r < 2; ++r) {
    float m = 0.0f;
    for (int c = 0; c < 4; ++c)
      m += yd[r * 4 + c];
    m /= 4.0f;

    float v = 0.0f;
    for (int c = 0; c < 4; ++c) {
      float d = yd[r * 4 + c] - m;
      v += d * d;
    }
    v /= 4.0f;

    EXPECT_NEAR(m, 0.0f, 1e-4f);
    EXPECT_NEAR(v, 1.0f, 2e-3f);
  }

  Tensor loss = y.sum();
  loss.backward();
  EXPECT_TRUE(x.has_grad());
  EXPECT_TRUE(ln.weight.has_grad());
  EXPECT_TRUE(ln.bias.has_grad());
}


TEST(NNTest, EmbeddingForwardIndexPath) {
  Device cpu{DeviceType::CPU, 0};
  nn::Embedding emb(4, 2);
  emb.weight.set_requires_grad(false); // trigger fast gather path

  float *w = static_cast<float *>(emb.weight.data());
  w[0] = 1.0f; w[1] = 1.1f;
  w[2] = 2.0f; w[3] = 2.1f;
  w[4] = 3.0f; w[5] = 3.1f;
  w[6] = 4.0f; w[7] = 4.1f;

  Tensor idx({1, 3}, cpu);
  float *id = static_cast<float *>(idx.data());
  id[0] = 2.0f; id[1] = 0.0f; id[2] = 3.0f;

  Tensor y = emb.forward(idx).to(cpu);
  const float *o = static_cast<const float *>(y.data());
  EXPECT_NEAR(o[0], 3.0f, 1e-6f);
  EXPECT_NEAR(o[1], 3.1f, 1e-6f);
  EXPECT_NEAR(o[2], 1.0f, 1e-6f);
  EXPECT_NEAR(o[3], 1.1f, 1e-6f);
  EXPECT_NEAR(o[4], 4.0f, 1e-6f);
  EXPECT_NEAR(o[5], 4.1f, 1e-6f);
}

TEST(NNTest, EmbeddingForwardIndexPathSupportsInt32IndicesAndFloat16Weights) {
  Device cpu{DeviceType::CPU, 0};
  TensorOptions options;
  options.dtype = DataType::Float16;
  nn::Embedding emb(4, 2, options);
  emb.weight.set_requires_grad(false);

  Tensor weights32({4, 2}, cpu, DataType::Float32);
  float *w = static_cast<float *>(weights32.data());
  w[0] = 1.0f; w[1] = 1.5f;
  w[2] = 2.0f; w[3] = 2.5f;
  w[4] = 3.0f; w[5] = 3.5f;
  w[6] = 4.0f; w[7] = 4.5f;
  emb.weight = weights32.to(DataType::Float16);

  Tensor idx({1, 3}, cpu, DataType::Int32);
  int32_t *id = static_cast<int32_t *>(idx.data());
  id[0] = 2; id[1] = 0; id[2] = 3;

  Tensor y = emb.forward(idx).to(DataType::Float32);
  const float *o = static_cast<const float *>(y.data());
  EXPECT_NEAR(o[0], 3.0f, 1e-2f);
  EXPECT_NEAR(o[1], 3.5f, 1e-2f);
  EXPECT_NEAR(o[2], 1.0f, 1e-2f);
  EXPECT_NEAR(o[3], 1.5f, 1e-2f);
  EXPECT_NEAR(o[4], 4.0f, 1e-2f);
  EXPECT_NEAR(o[5], 4.5f, 1e-2f);
}

TEST(NNTest, MultiHeadAttentionForwardShape) {
  Device cpu{DeviceType::CPU, 0};
  nn::MultiHeadAttention mha(8, 2, true);

  Tensor x({2, 4, 8}, cpu);
  x.uniform_(-1.0f, 1.0f);

  Tensor y = mha.forward(x).to(cpu);
  EXPECT_EQ(y.shape()[0], 2);
  EXPECT_EQ(y.shape()[1], 4);
  EXPECT_EQ(y.shape()[2], 8);
}


TEST(NNTest, MultiHeadAttentionCausalMaskBehavior) {
  Device cpu{DeviceType::CPU, 0};
  nn::MultiHeadAttention mha(4, 2, true);

  Tensor x1({1, 2, 4}, cpu);
  float *d1 = static_cast<float *>(x1.data());
  // token 0
  d1[0] = 0.1f; d1[1] = 0.2f; d1[2] = 0.3f; d1[3] = 0.4f;
  // token 1
  d1[4] = 0.5f; d1[5] = 0.6f; d1[6] = 0.7f; d1[7] = 0.8f;

  Tensor x2 = x1.clone();
  float *d2 = static_cast<float *>(x2.data());
  // Change only future token heavily
  d2[4] = 10.0f; d2[5] = -10.0f; d2[6] = 20.0f; d2[7] = -20.0f;

  Tensor y1 = mha.forward(x1).to(cpu);
  Tensor y2 = mha.forward(x2).to(cpu);

  const float *o1 = static_cast<const float *>(y1.data());
  const float *o2 = static_cast<const float *>(y2.data());

  // First token output should be unaffected by future token under causal mask.
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(o1[i], o2[i], 1e-4f);
  }
}
