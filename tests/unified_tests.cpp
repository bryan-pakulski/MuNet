#include "optim.hpp"
#include "tensor.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <gtest/gtest.h>

using namespace munet;

class BackendTest : public ::testing::TestWithParam<Device> {
protected:
  Device dev() { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(AllBackends, BackendTest,
                         ::testing::ValuesIn(test::get_available_devices()),
                         [](const ::testing::TestParamInfo<Device> &info) {
                           std::string name = info.param.to_string();
                           // Replace ':' with '_' to satisfy GoogleTest naming
                           // rules
                           std::replace(name.begin(), name.end(), ':', '_');
                           return name;
                         });

TEST_P(BackendTest, ElementwiseAdd) {
  Tensor a({2, 2}, dev());
  Tensor b({2, 2}, dev());
  a.uniform_(1.0f, 1.0f); // All 1s
  b.uniform_(2.0f, 2.0f); // All 2s

  Tensor c = a + b;
  Tensor res = c.to({DeviceType::CPU, 0});
  const float *data = static_cast<const float *>(res.data());
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(data[i], 3.0f);
}

TEST_P(BackendTest, MatMul) {
  Tensor a({2, 3}, dev());
  Tensor b({3, 2}, dev());
  a.uniform_(1.0f, 1.0f);
  b.uniform_(1.0f, 1.0f);

  Tensor c = a.matmul(b); // [2,3] @ [3,2] -> [2,2]
  EXPECT_EQ(c.shape()[0], 2);
  EXPECT_EQ(c.shape()[1], 2);

  Tensor res = c.to({DeviceType::CPU, 0});
  const float *data = static_cast<const float *>(res.data());
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(data[i], 3.0f);
}

TEST_P(BackendTest, SimpleAutograd) {
  Tensor w({2, 2}, dev(), DataType::Float32, true);
  Tensor x({2, 2}, dev(), DataType::Float32, false);

  w.uniform_(1.0f, 1.0f);
  x.uniform_(1.0f, 1.0f);

  Tensor z = w * x;
  Tensor loss = z.sum();
  loss.backward();

  ASSERT_TRUE(w.has_grad());
  Tensor grad = w.grad().to({DeviceType::CPU, 0});
  const float *g_ptr = static_cast<const float *>(grad.data());
  // d(sum(w*x))/dw = x. Since x is all 1s, grad is all 1s.
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(g_ptr[i], 1.0f);
}

TEST_P(BackendTest, Conv2DForward) {
  // B=1, C=1, H=3, W=3
  Tensor in({1, 1, 3, 3}, dev());
  // Out=1, In=1, k=2, k=2
  Tensor weight({1, 1, 2, 2}, dev());
  in.uniform_(1.0f, 1.0f);
  weight.uniform_(1.0f, 1.0f);

  Tensor out = in.conv2d(weight, Tensor(), 1, 0); // Valid padding
  // Output should be (3-2+1) = 2x2. Each cell is sum of 2x2 kernel of 1s.
  EXPECT_EQ(out.shape()[2], 2);

  Tensor res = out.to({DeviceType::CPU, 0});
  const float *data = static_cast<const float *>(res.data());
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(data[i], 4.0f);
}

TEST_P(BackendTest, Concatenation) {
  Tensor a({1, 2, 2}, dev());
  Tensor b({1, 3, 2}, dev());
  a.uniform_(1.0f, 1.0f);
  b.uniform_(2.0f, 2.0f);

  std::vector<Tensor> inputs = {a, b};
  Tensor c = Tensor::cat(inputs, 1); // Concat on dim 1

  EXPECT_EQ(c.shape()[0], 1);
  EXPECT_EQ(c.shape()[1], 5);
  EXPECT_EQ(c.shape()[2], 2);

  Tensor res = c.to({DeviceType::CPU, 0});
  const float *data = static_cast<const float *>(res.data());
  // First 4 elements (1*2*2) should be 1.0, next 6 (1*3*2) should be 2.0
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(data[i], 1.0f);
  for (int i = 4; i < 10; ++i)
    EXPECT_FLOAT_EQ(data[i], 2.0f);
}

TEST_P(BackendTest, CrossEntropyLoss) {
  Tensor logits({1, 3}, dev(), DataType::Float32, true);
  Tensor target({1, 3}, dev());

  // Fill CPU then copy to ensure exact values
  Tensor l_cpu({1, 3}, {DeviceType::CPU, 0});
  float *l_ptr = (float *)l_cpu.data();
  l_ptr[0] = 0.1f;
  l_ptr[1] = 0.2f;
  l_ptr[2] = 0.7f;

  Tensor t_cpu({1, 3}, {DeviceType::CPU, 0});
  float *t_ptr = (float *)t_cpu.data();
  t_ptr[0] = 0.0f;
  t_ptr[1] = 0.0f;
  t_ptr[2] = 1.0f;

  logits.impl_->backend().copy(l_cpu.data(), logits.data(), logits.bytes(),
                               l_cpu.device(), dev());
  target.impl_->backend().copy(t_cpu.data(), target.data(), target.bytes(),
                               t_cpu.device(), dev());

  Tensor loss = logits.cross_entropy(target);
  loss.backward();

  Tensor loss_cpu = loss.to({DeviceType::CPU, 0});
  // -log(exp(0.7) / (exp(0.1)+exp(0.2)+exp(0.7)))
  EXPECT_NEAR(((float *)loss_cpu.data())[0], 0.7679f, 1e-3);
  EXPECT_TRUE(logits.has_grad());
}

TEST(TensorTest, BroadcastingAdd) {
  Device dev{DeviceType::CPU, 0};
  Tensor a({2, 3}, dev); // Matrix
  Tensor b({3}, dev);    // Vector

  // This should ideally broadcast b to [2, 3]
  // Currently MuNet ops.hpp will throw std::runtime_error("Add: shape
  // mismatch")
  auto c = a + b;

  EXPECT_EQ(c.shape()[0], 2);
  EXPECT_EQ(c.shape()[1], 3);
}

TEST(OptimTest, SGDConsistency) {
  Device dev{DeviceType::CPU, 0};
  Tensor w({1}, dev, DataType::Float32, true);
  ((float *)w.data())[0] = 1.0f;

  // Simulate a gradient
  Tensor loss = w * w; // y = x^2, dy/dx = 2x
  loss.backward();

  munet::optim::SGD opt({w}, 0.1f);
  opt.step();

  // w_new = 1.0 - (0.1 * 2.0) = 0.8
  float val = ((float *)w.to({DeviceType::CPU, 0}).data())[0];
  EXPECT_NEAR(val, 0.8f, 1e-6);
}
