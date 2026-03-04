#include "tensor.hpp"
#include <gtest/gtest.h>

using namespace munet;

TEST(TensorTest, CreationAndMetadata) {
  Tensor t({2, 3}, {DeviceType::CPU, 0});

  EXPECT_EQ(t.size(), 6);
  EXPECT_EQ(t.shape()[0], 2);
  EXPECT_EQ(t.shape()[1], 3);
  EXPECT_EQ(t.device().type, DeviceType::CPU);
  EXPECT_FALSE(t.requires_grad());
  EXPECT_NE(t.data(), nullptr);
}

TEST(TensorTest, Addition) {
  Tensor a({2}, {DeviceType::CPU, 0});
  Tensor b({2}, {DeviceType::CPU, 0});

  float *a_data = static_cast<float *>(a.data());
  float *b_data = static_cast<float *>(b.data());

  a_data[0] = 1.0f;
  a_data[1] = 2.0f;
  b_data[0] = 3.0f;
  b_data[1] = 4.0f;

  Tensor c = a + b;

  float *c_data = static_cast<float *>(c.data());
  EXPECT_FLOAT_EQ(c_data[0], 4.0f);
  EXPECT_FLOAT_EQ(c_data[1], 6.0f);
}
