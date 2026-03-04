#include "tensor.hpp"
#include "types.hpp"
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <iostream>

using namespace munet;

TEST(CudaTest, MNISTTrain) {
  // 1. Setup Device
  Device dev{DeviceType::CUDA, 0};
  std::cout << "Device initialized: CUDA:0" << std::endl;

  // 2. Alloc Weights (Linear Layer Simulation: 784 -> 128 -> 10)
  int batch_size = 64;
  int in_features = 784;
  int hidden_dim = 128;
  int out_features = 10;

  std::cout << "Allocating weights..." << std::endl;

  // Layer 1
  Tensor w1({in_features, hidden_dim}, dev, DataType::Float32, true);
  Tensor b1({1, hidden_dim}, dev, DataType::Float32, true);
  w1.uniform_(-0.1f, 0.1f);
  b1.uniform_(-0.1f, 0.1f);

  // Layer 2
  Tensor w2({hidden_dim, out_features}, dev, DataType::Float32, true);
  Tensor b2({1, out_features}, dev, DataType::Float32, true);
  w2.uniform_(-0.1f, 0.1f);
  b2.uniform_(-0.1f, 0.1f);

  // 3. Training Loop Simulation
  // Running enough iterations to force ring-buffer wraparound (3 frames in
  // flight)
  int iterations = 500;
  std::cout << "Starting training loop (" << iterations << " iters)..."
            << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    std::cout.flush();

    // Input Data
    Tensor x({batch_size, in_features}, dev);
    x.uniform_(0.0f, 1.0f);
    // --- Forward Pass ---

    // Linear 1: x @ w1 + b1 (broadcasted via matmul trick)
    Tensor z1 = x.matmul(w1);

    // Bias broadcast: ones(B,1) @ b1(1,H) -> (B,H)
    Tensor ones1({batch_size, 1}, dev);
    ones1.uniform_(1.0f, 1.0f);
    Tensor bias_term1 = ones1.matmul(b1);

    Tensor out1 = z1 + bias_term1;
    Tensor a1 = out1.relu();

    // Linear 2: a1 @ w2 + b2
    Tensor z2 = a1.matmul(w2);

    Tensor ones2({batch_size, 1}, dev);
    ones2.uniform_(1.0f, 1.0f);
    Tensor bias_term2 = ones2.matmul(b2);

    Tensor out2 = z2 + bias_term2;

    // Loss: MSE with target
    Tensor target({batch_size, out_features}, dev);
    target.uniform_(0.0f, 1.0f);

    Tensor diff = out2 - target;
    Tensor sq =
        diff * diff; // Element-wise mul (missing overload? using explicit mul
                     // if needed, but assuming * operator works)

    // Since * operator calls ops::mul which calls backend->mul
    Tensor loss = sq.sum();

    // --- Backward Pass ---

    // Zero Grads
    w1.zero_grad();
    b1.zero_grad();
    w2.zero_grad();
    b2.zero_grad();

    // Backprop
    loss.backward();

    // --- Optimizer Step ---
    float lr = 0.001f;
    w1.step(lr);
    b1.step(lr);
    w2.step(lr);
    b2.step(lr);

    // --- Sync & Check ---
    // Moving result to CPU forces a wait on the queue
    Tensor cpu_loss = loss.to(Device{DeviceType::CPU, 0});
    float val = ((float *)cpu_loss.data())[0];

    // Validate Result
    if (std::isnan(val) || std::isinf(val)) {
      throw std::runtime_error("Loss exploded to NaN/Inf");
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Training took " << duration << " ms" << std::endl;
}
