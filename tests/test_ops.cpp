#include "tensor.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <gtest/gtest.h>

using namespace munet;

class OpsTest : public ::testing::TestWithParam<Device> {
protected:
  Device dev() { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(AllBackends, OpsTest,
                         ::testing::ValuesIn(test::get_available_devices()),
                         [](const ::testing::TestParamInfo<Device> &info) {
                           std::string name = info.param.to_string();
                           std::replace(name.begin(), name.end(), ':', '_');
                           return name;
                         });

TEST_P(OpsTest, ReshapeGradient) {
  Tensor x({2, 2}, dev(), DataType::Float32, true);
  x.uniform_(1.0f, 1.0f);

  Tensor y = x.reshape({4, 1});
  Tensor z = y.sum();
  z.backward();

  ASSERT_TRUE(x.has_grad());
  EXPECT_EQ(x.grad().shape()[0], 2);
  EXPECT_EQ(x.grad().shape()[1], 2);

  Tensor g = x.grad().to({DeviceType::CPU, 0});
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ((*(float *)g.data()), 1.0f);
}

TEST_P(OpsTest, CatDim0) {
  Tensor a({2, 2}, dev());
  Tensor b({1, 2}, dev());
  a.uniform_(1.0f, 1.0f);
  b.uniform_(2.0f, 2.0f);

  Tensor c = Tensor::cat({a, b}, 0);
  EXPECT_EQ(c.shape()[0], 3);
  EXPECT_EQ(c.shape()[1], 2);

  Tensor res = c.to({DeviceType::CPU, 0});
  float *data = (float *)res.data();
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[4], 2.0f);
}

TEST_P(OpsTest, BroadCastSubMul) {
  Tensor a({2, 2}, dev());
  Tensor b({2}, dev()); // To be broadcasted as row

  Tensor val_a({2, 2}, {DeviceType::CPU, 0});
  ((float *)val_a.data())[0] = 10;
  ((float *)val_a.data())[1] = 20;
  ((float *)val_a.data())[2] = 30;
  ((float *)val_a.data())[3] = 40;
  a.impl_->backend().copy(val_a.data(), a.data(), a.bytes(), val_a.device(),
                          dev());

  Tensor val_b({2}, {DeviceType::CPU, 0});
  ((float *)val_b.data())[0] = 1;
  ((float *)val_b.data())[1] = 2;
  b.impl_->backend().copy(val_b.data(), b.data(), b.bytes(), val_b.device(),
                          dev());

  Tensor c = a - b; // [10-1, 20-2, 30-1, 40-2]
  Tensor res = c.to({DeviceType::CPU, 0});
  EXPECT_FLOAT_EQ(((float *)res.data())[0], 9.0f);
  EXPECT_FLOAT_EQ(((float *)res.data())[1], 18.0f);
  EXPECT_FLOAT_EQ(((float *)res.data())[2], 29.0f);
  EXPECT_FLOAT_EQ(((float *)res.data())[3], 38.0f);

  Tensor d = a * b; // [10*1, 20*2, 30*1, 40*2]
  res = d.to({DeviceType::CPU, 0});
  EXPECT_FLOAT_EQ(((float *)res.data())[0], 10.0f);
  EXPECT_FLOAT_EQ(((float *)res.data())[1], 40.0f);
}

TEST_P(OpsTest, VectorToMatrixAdd) {
  // [2, 3] + [3]
  Tensor a({2, 3}, dev());
  a.uniform_(10.0f, 10.0f); // All 10s

  Tensor b({3}, dev());
  Tensor val_b({3}, {DeviceType::CPU, 0});
  ((float *)val_b.data())[0] = 1.0f;
  ((float *)val_b.data())[1] = 2.0f;
  ((float *)val_b.data())[2] = 3.0f;
  b.impl_->backend().copy(val_b.data(), b.data(), b.bytes(), val_b.device(),
                          dev());

  Tensor c = a + b;
  EXPECT_EQ(c.shape(), Shape({2, 3}));

  Tensor c_cpu = c.to({DeviceType::CPU, 0});
  float *data = (float *)c_cpu.data();
  // Row 1: 10+1, 10+2, 10+3
  EXPECT_FLOAT_EQ(data[0], 11.0f);
  EXPECT_FLOAT_EQ(data[1], 12.0f);
  EXPECT_FLOAT_EQ(data[2], 13.0f);
  // Row 2: (Identical due to broadcast)
  EXPECT_FLOAT_EQ(data[3], 11.0f);
  EXPECT_FLOAT_EQ(data[4], 12.0f);
  EXPECT_FLOAT_EQ(data[5], 13.0f);
}

TEST_P(OpsTest, ScalarToTensorMul) {
  // [2, 2] * [1]
  Tensor a({2, 2}, dev());
  a.uniform_(5.0f, 5.0f);

  Tensor b({1}, dev());
  Tensor val_b({1}, {DeviceType::CPU, 0});
  ((float *)val_b.data())[0] = 2.0f;
  b.impl_->backend().copy(val_b.data(), b.data(), b.bytes(), val_b.device(),
                          dev());

  Tensor c = a * b;
  Tensor res = c.to({DeviceType::CPU, 0});
  float *data = (float *)res.data();
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(*data, 10.0f);
}

TEST_P(OpsTest, MultiDimExpansion) {
  // [1, 3, 1] + [2, 1, 2] -> [2, 3, 2]
  Tensor a({1, 3, 1}, dev()); // All 1s
  a.uniform_(1.0f, 1.0f);
  Tensor b({2, 1, 2}, dev()); // All 2s
  b.uniform_(2.0f, 2.0f);

  Tensor c = a + b;
  EXPECT_EQ(c.shape(), Shape({2, 3, 2}));

  Tensor res = c.to({DeviceType::CPU, 0});
  float *data = (float *)res.data();
  for (int i = 0; i < 12; ++i)
    EXPECT_FLOAT_EQ(*data, 3.0f);
}

TEST_P(OpsTest, ScalarAutograd) {
  // Testing the path where target_shape.numel() == 1
  Tensor a({2, 2}, dev(), DataType::Float32, true);
  a.uniform_(1.0f, 1.0f);

  Tensor b({1}, dev(), DataType::Float32, true);
  Tensor val_b({1}, {DeviceType::CPU, 0});
  ((float *)val_b.data())[0] = 10.0f;
  b.impl_->backend().copy(val_b.data(), b.data(), b.bytes(), val_b.device(),
                          dev());

  Tensor c = a + b; // Result is [2, 2]
  Tensor loss = c.sum();
  loss.backward();

  // grad for 'a' should be [1, 1, 1, 1]
  // grad for 'b' should be 4.0 (sum of all gradients in c)
  Tensor gb = b.grad().to({DeviceType::CPU, 0});
  EXPECT_FLOAT_EQ(((float *)gb.data())[0], 4.0f);
}

TEST_P(OpsTest, MeanLastDimForwardAndBackward) {
  Tensor x({2, 3}, dev(), DataType::Float32, true);
  Tensor x_cpu({2, 3}, {DeviceType::CPU, 0});
  float *xp = static_cast<float *>(x_cpu.data());
  xp[0] = 1.0f;
  xp[1] = 2.0f;
  xp[2] = 3.0f;
  xp[3] = 4.0f;
  xp[4] = 5.0f;
  xp[5] = 6.0f;
  x.impl_->backend().copy(x_cpu.data(), x.data(), x.bytes(), x_cpu.device(),
                          dev());

  Tensor y = x.mean(-1, false);
  Tensor y_cpu = y.to({DeviceType::CPU, 0});
  const float *yp = static_cast<const float *>(y_cpu.data());
  EXPECT_NEAR(yp[0], 2.0f, 1e-5f);
  EXPECT_NEAR(yp[1], 5.0f, 1e-5f);

  Tensor loss = y.sum();
  loss.backward();
  Tensor g_cpu = x.grad().to({DeviceType::CPU, 0});
  const float *gp = static_cast<const float *>(g_cpu.data());
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(gp[i], 1.0f / 3.0f, 1e-5f);
  }
}

TEST_P(OpsTest, NarrowViewSharesStorageOffset) {
  Tensor x({2, 4}, dev());
  Tensor x_cpu({2, 4}, {DeviceType::CPU, 0});
  float *xp = static_cast<float *>(x_cpu.data());
  for (int i = 0; i < 8; ++i)
    xp[i] = static_cast<float>(i);
  x.impl_->backend().copy(x_cpu.data(), x.data(), x.bytes(), x_cpu.device(),
                          dev());

  Tensor slice = x.narrow(1, 1, 2);
  EXPECT_EQ(slice.shape(), Shape({2, 2}));
  EXPECT_EQ(slice.storage_offset(), static_cast<size_t>(1));

  Tensor slice_cpu = slice.contiguous().to({DeviceType::CPU, 0});
  const float *sp = static_cast<const float *>(slice_cpu.data());
  EXPECT_FLOAT_EQ(sp[0], 1.0f);
  EXPECT_FLOAT_EQ(sp[1], 2.0f);
  EXPECT_FLOAT_EQ(sp[2], 5.0f);
  EXPECT_FLOAT_EQ(sp[3], 6.0f);
}
