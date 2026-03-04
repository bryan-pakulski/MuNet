#include "tensor.hpp"
#include <gtest/gtest.h>

using namespace munet;

TEST(AutogradTest, AddBackward) {
  // 1. Create leaf tensors requiring gradients
  Tensor a({1}, {DeviceType::CPU, 0}, DataType::Float32, true);
  Tensor b({1}, {DeviceType::CPU, 0}, DataType::Float32, true);

  float *a_data = static_cast<float *>(a.data());
  float *b_data = static_cast<float *>(b.data());
  a_data[0] = 2.0f;
  b_data[0] = 3.0f;

  // 2. Forward pass
  // c = a + b
  Tensor c = a + b;

  // 3. Backward pass
  c.backward(); // implicitly uses grad=1.0

  // 4. Verify Gradients
  EXPECT_TRUE(a.has_grad());
  EXPECT_TRUE(b.has_grad());

  float *a_grad = static_cast<float *>(a.grad().data());
  float *b_grad = static_cast<float *>(b.grad().data());

  // dz/da = 1, dz/db = 1
  EXPECT_FLOAT_EQ(a_grad[0], 1.0f);
  EXPECT_FLOAT_EQ(b_grad[0], 1.0f);
}
