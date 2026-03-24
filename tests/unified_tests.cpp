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

class BroadcastTest : public ::testing::TestWithParam<Device> {
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

INSTANTIATE_TEST_SUITE_P(AllBackends, BroadcastTest,
                         ::testing::ValuesIn(test::get_available_devices()),
                         [](const ::testing::TestParamInfo<Device> &info) {
                           std::string name = info.param.to_string();
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

TEST_P(BroadcastTest, BroadcastingAdd) {
  Tensor a({2, 3}, dev()); // Matrix
  Tensor b({3}, dev());    // Vector

  // This should ideally broadcast b to [2, 3]
  // Currently MuNet ops.hpp will throw std::runtime_error("Add: shape
  // mismatch")
  auto c = a + b;

  EXPECT_EQ(c.shape()[0], 2);
  EXPECT_EQ(c.shape()[1], 3);
}

TEST(BroadcastTest, ShapeAndStrideLogic) {
  // Case 1: Identical Shapes [2, 3] + [2, 3]
  {
    Shape s1 = {2, 3};
    Strides st1 = default_strides(s1); // [3, 1]
    auto info = compute_broadcast(s1, st1, s1, st1);

    EXPECT_TRUE(info.can_broadcast);
    EXPECT_EQ(info.out_shape, Shape({2, 3}));
    EXPECT_EQ(info.strides_a, Strides({3, 1}));
    EXPECT_EQ(info.strides_b, Strides({3, 1}));
  }

  // Case 2: Prepending Dimensions [3] + [2, 3]
  {
    Shape s1 = {3};
    Strides st1 = default_strides(s1); // [1]
    Shape s2 = {2, 3};
    Strides st2 = default_strides(s2); // [3, 1]

    auto info = compute_broadcast(s1, st1, s2, st2);
    EXPECT_TRUE(info.can_broadcast);
    EXPECT_EQ(info.out_shape, Shape({2, 3}));
    // Tensor A gets a 0-stride for the prepended dimension
    EXPECT_EQ(info.strides_a, Strides({0, 1}));
    EXPECT_EQ(info.strides_b, Strides({3, 1}));
  }

  // Case 3: Middle Dimension Expansion [2, 1, 4] + [1, 3, 4]
  {
    Shape s1 = {2, 1, 4};
    Strides st1 = default_strides(s1); // [4, 4, 1]
    Shape s2 = {1, 3, 4};
    Strides st2 = default_strides(s2); // [12, 4, 1]

    auto info = compute_broadcast(s1, st1, s2, st2);
    EXPECT_TRUE(info.can_broadcast);
    EXPECT_EQ(info.out_shape, Shape({2, 3, 4}));
    // A is [2, 1, 4] -> strides [4, 0, 1]
    EXPECT_EQ(info.strides_a, Strides({4, 0, 1}));
    // B is [1, 3, 4] -> strides [0, 4, 1]
    EXPECT_EQ(info.strides_b, Strides({0, 4, 1}));
  }

  // Case 4: Incompatible [2, 2] + [2, 3]
  {
    auto info = compute_broadcast({2, 2}, {2, 1}, {2, 3}, {3, 1});
    EXPECT_FALSE(info.can_broadcast);
  }
}

TEST(BroadcastTest, ScalarBroadcasting) {
  // [2, 2] + [1]
  Shape s1 = {2, 2};
  Strides st1 = default_strides(s1);
  Shape s2 = {1};
  Strides st2 = {1};

  auto info = compute_broadcast(s1, st1, s2, st2);
  EXPECT_TRUE(info.can_broadcast);
  EXPECT_EQ(info.out_shape, Shape({2, 2}));
  EXPECT_EQ(info.strides_a, Strides({2, 1}));
  EXPECT_EQ(info.strides_b, Strides({0, 0})); // All 0s for scalar
}

TEST(BroadcastTest, CPUExecution) {
  Device cpu{DeviceType::CPU, 0};

  // Test [2, 3] + [3]
  Tensor a({2, 3}, cpu);
  a.uniform_(10.0f, 10.0f); // All 10s

  Tensor b({3}, cpu);
  Tensor val_b({3}, cpu);
  ((float *)val_b.data())[0] = 1.0f;
  ((float *)val_b.data())[1] = 2.0f;
  ((float *)val_b.data())[2] = 3.0f;
  b.impl_->backend().copy(val_b.data(), b.data(), b.bytes(), cpu, cpu);

  Tensor c = a + b;
  EXPECT_EQ(c.shape(), Shape({2, 3}));

  Tensor c_cpu = c.to(cpu);
  float *data = (float *)c_cpu.data();
  // Row 1
  EXPECT_FLOAT_EQ(data[0], 11.0f);
  EXPECT_FLOAT_EQ(data[1], 12.0f);
  EXPECT_FLOAT_EQ(data[2], 13.0f);
  // Row 2 (should be identical because b was broadcasted)
  EXPECT_FLOAT_EQ(data[3], 11.0f);
  EXPECT_FLOAT_EQ(data[4], 12.0f);
  EXPECT_FLOAT_EQ(data[5], 13.0f);
}

// ============================================================================
// Batched MatMul Tests
// ============================================================================

TEST_P(BackendTest, BatchedMatMul_2D_BackwardCompatible) {
  // Test that 2D matmul still works (backward compatibility)
  Tensor a({2, 3}, dev());
  Tensor b({3, 2}, dev());
  a.uniform_(1.0f, 1.0f);
  b.uniform_(1.0f, 1.0f);

  Tensor c = a.matmul(b);
  EXPECT_EQ(c.shape().size(), 2);
  EXPECT_EQ(c.shape()[0], 2);
  EXPECT_EQ(c.shape()[1], 2);

  Tensor res = c.to({DeviceType::CPU, 0});
  const float *data = static_cast<const float *>(res.data());
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(data[i], 3.0f);
}

TEST_P(BackendTest, BatchedMatMul_3D_BroadcastedWeights) {
  // Test [B, M, K] x [K, N] -> [B, M, N]
  // B=2, M=3, K=4, N=5
  Tensor a({2, 3, 4}, dev());
  Tensor b({4, 5}, dev());
  a.uniform_(1.0f, 1.0f);
  b.uniform_(1.0f, 1.0f);

  Tensor c = a.matmul(b);
  EXPECT_EQ(c.shape().size(), 3);
  EXPECT_EQ(c.shape()[0], 2);
  EXPECT_EQ(c.shape()[1], 3);
  EXPECT_EQ(c.shape()[2], 5);

  Tensor res = c.to({DeviceType::CPU, 0});
  const float *data = static_cast<const float *>(res.data());
  // Each element should be K=4 (sum of 4 ones)
  for (size_t i = 0; i < 2 * 3 * 5; ++i)
    EXPECT_FLOAT_EQ(data[i], 4.0f);
}

TEST_P(BackendTest, BatchedMatMul_3D_BatchedWeights) {
  // Test [B, M, K] x [B, K, N] -> [B, M, N]
  // B=2, M=3, K=4, N=5
  Tensor a({2, 3, 4}, dev());
  Tensor b({2, 4, 5}, dev());
  a.uniform_(1.0f, 1.0f);
  b.uniform_(2.0f, 2.0f);

  Tensor c = a.matmul(b);
  EXPECT_EQ(c.shape().size(), 3);
  EXPECT_EQ(c.shape()[0], 2);
  EXPECT_EQ(c.shape()[1], 3);
  EXPECT_EQ(c.shape()[2], 5);

  Tensor res = c.to({DeviceType::CPU, 0});
  const float *data = static_cast<const float *>(res.data());
  // Each element should be K=4 * 2.0 = 8.0
  for (size_t i = 0; i < 2 * 3 * 5; ++i)
    EXPECT_FLOAT_EQ(data[i], 8.0f);
}

TEST_P(BackendTest, BatchedMatMul_SingleBatch) {
  // Test with B=1: [1, M, K] x [K, N] -> [1, M, N]
  Tensor a({1, 3, 4}, dev());
  Tensor b({4, 5}, dev());
  a.uniform_(1.0f, 1.0f);
  b.uniform_(1.0f, 1.0f);

  Tensor c = a.matmul(b);
  EXPECT_EQ(c.shape().size(), 3);
  EXPECT_EQ(c.shape()[0], 1);
  EXPECT_EQ(c.shape()[1], 3);
  EXPECT_EQ(c.shape()[2], 5);

  Tensor res = c.to({DeviceType::CPU, 0});
  const float *data = static_cast<const float *>(res.data());
  for (size_t i = 0; i < 1 * 3 * 5; ++i)
    EXPECT_FLOAT_EQ(data[i], 4.0f);
}

TEST_P(BackendTest, BatchedMatMul_LargeBatch) {
  // Test with larger batch: [8, 4, 16] x [16, 32] -> [8, 4, 32]
  Tensor a({8, 4, 16}, dev());
  Tensor b({16, 32}, dev());
  a.uniform_(0.5f, 0.5f);
  b.uniform_(1.0f, 1.0f);

  Tensor c = a.matmul(b);
  EXPECT_EQ(c.shape().size(), 3);
  EXPECT_EQ(c.shape()[0], 8);
  EXPECT_EQ(c.shape()[1], 4);
  EXPECT_EQ(c.shape()[2], 32);

  Tensor res = c.to({DeviceType::CPU, 0});
  const float *data = static_cast<const float *>(res.data());
  // Each element should be 16 * 0.5 * 1.0 = 8.0
  for (size_t i = 0; i < 8 * 4 * 32; ++i)
    EXPECT_FLOAT_EQ(data[i], 8.0f);
}
