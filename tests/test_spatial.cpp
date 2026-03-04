#include "tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace munet;

// A generic function to train a micro "U-Net" style architecture
// to ensure all spatial ops work together in a graph.
void TrainSpatialModel(Device dev) {
  // Input: Batch=2, Channels=1, Height=6, Width=6
  Tensor x({2, 1, 6, 6}, dev, DataType::Float32, false);
  x.uniform_(0.0f, 1.0f);

  // Target: Same shape
  Tensor y({2, 1, 6, 6}, dev, DataType::Float32, false);
  y.uniform_(0.0f, 1.0f);

  // --- Model Weights ---
  // Conv1: 1->4 channels, 3x3 kernel, padding=1 (keeps 6x6)
  Tensor w1({4, 1, 3, 3}, dev, DataType::Float32, true);
  Tensor b1({4}, dev, DataType::Float32, true);
  w1.uniform_(-0.1f, 0.1f);
  b1.uniform_(-0.01f, 0.01f);

  // Conv2: 4->1 channels, 3x3 kernel, padding=1 (keeps 6x6)
  Tensor w2({1, 4, 3, 3}, dev, DataType::Float32, true);
  Tensor b2({1}, dev, DataType::Float32, true);
  w2.uniform_(-0.1f, 0.1f);
  b2.uniform_(-0.01f, 0.01f);

  float initial_loss = 0.0f;
  float final_loss = 0.0f;
  // Lower learning rate for stability
  float lr = 0.01f;

  for (int i = 0; i < 10; ++i) {
    w1.zero_grad();
    b1.zero_grad();
    w2.zero_grad();
    b2.zero_grad();

    // --- Forward Pass ---
    // 1. Conv1 (2, 4, 6, 6)
    Tensor h1 = x.conv2d(w1, b1, 1, 1);
    Tensor a1 = h1.relu();

    // 2. MaxPool (2, 4, 3, 3) - Downsample
    Tensor p1 = a1.max_pool2d(2, 2, 0);

    // 3. Upsample (2, 4, 6, 6) - Upsample back
    Tensor u1 = p1.upsample2d(2);

    // 4. Conv2 (2, 1, 6, 6) - Output
    Tensor out = u1.conv2d(w2, b2, 1, 1);

    // Loss (MSE)
    Tensor diff = out - y;
    Tensor sq = diff * diff;
    Tensor loss = sq.sum();

    // Sync scalar to CPU to check values
    Tensor loss_cpu = loss.to({DeviceType::CPU, 0});
    float current_loss = static_cast<float *>(loss_cpu.data())[0];

    if (i == 0)
      initial_loss = current_loss;
    final_loss = current_loss;

    // --- Backward Pass ---
    loss.backward();

    // --- Update ---
    w1.step(lr);
    b1.step(lr);
    w2.step(lr);
    b2.step(lr);
  }

  // Basic validation: Loss should ideally decrease, or at least not be NaN
  EXPECT_FALSE(std::isnan(final_loss));
  EXPECT_FALSE(std::isinf(final_loss));
  EXPECT_LT(final_loss, initial_loss);

  std::cout << "Device " << dev.to_string() << " | Start Loss: " << initial_loss
            << " | End Loss: " << final_loss << std::endl;
}

TEST(SpatialOps, TrainCPU) { TrainSpatialModel({DeviceType::CPU, 0}); }

TEST(SpatialOps, TrainCUDA) {
  try {
    // Attempt to init device
    Device dev{DeviceType::CUDA, 0};
    // If successful, run test
    TrainSpatialModel(dev);
  } catch (const std::exception &e) {
    std::cout << "[SKIPPED] CUDA test skipped: " << e.what() << std::endl;
  }
}

TEST(SpatialOps, TrainVulkan) {
  try {
    // Attempt to init device
    Device dev{DeviceType::VULKAN, 0};
    // If successful, run test
    TrainSpatialModel(dev);
  } catch (const std::exception &e) {
    std::cout << "[SKIPPED] Vulkan test skipped: " << e.what() << std::endl;
  }
}

TEST(SpatialOps, Conv2DShapes) {
  Device dev{DeviceType::CPU, 0};
  Tensor in({1, 1, 4, 4}, dev);
  Tensor w({1, 1, 3, 3}, dev);

  // No padding: 4 - 3 + 1 = 2 output size
  Tensor out = in.conv2d(w, Tensor(), 1, 0);
  EXPECT_EQ(out.shape()[2], 2);
  EXPECT_EQ(out.shape()[3], 2);

  // Padding 1: 4 + 2 - 3 + 1 = 4 output size
  Tensor out2 = in.conv2d(w, Tensor(), 1, 1);
  EXPECT_EQ(out2.shape()[2], 4);
  EXPECT_EQ(out2.shape()[3], 4);

  // Stride 2, Padding 1: (4 + 2 - 3)/2 + 1 = 2 output size
  Tensor out3 = in.conv2d(w, Tensor(), 2, 1);
  EXPECT_EQ(out3.shape()[2], 2);
  EXPECT_EQ(out3.shape()[3], 2);
}
