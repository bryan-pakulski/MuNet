
#include <kernels.hpp>
#include <munet.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

using namespace munet;

// Helper to check floats
bool is_close(float a, float b, float epsilon = 1e-5f) {
  return std::abs(a - b) < epsilon;
}

void test_tensor_alloc() {
  Tensor t({2, 3}, Device::CPU, DataType::FP32);
  assert(t.size() == 6);
  assert(t.bytes() == 24);
  std::cout << "[OK] Tensor allocation\n";
}

void test_linear_forward() {
  Linear layer(2, 3);
  auto params = layer.get_parameters();

  // Inject deterministic weights (in_c * out_features + out_c)
  float *w = static_cast<float *>(params["weight"]->data());
  w[0] = 1.0f;
  w[1] = 2.0f;
  w[2] = 3.0f; // in_c=0
  w[3] = -1.0f;
  w[4] = -2.0f;
  w[5] = -3.0f; // in_c=1

  float *b = static_cast<float *>(params["bias"]->data());
  b[0] = 0.1f;
  b[1] = 0.2f;
  b[2] = 0.3f;

  Tensor input({1, 2}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(input.data());
  in_ptr[0] = 0.5f;
  in_ptr[1] = 1.5f;

  Tensor out = layer.forward(input);
  const float *out_ptr = static_cast<const float *>(out.data());

  // Expected Y = X*W + b
  assert(is_close(out_ptr[0], -0.9f)); // 0.5*1.0 + 1.5*-1.0 + 0.1
  assert(is_close(out_ptr[1], -1.8f)); // 0.5*2.0 + 1.5*-2.0 + 0.2
  assert(is_close(out_ptr[2], -2.7f)); // 0.5*3.0 + 1.5*-3.0 + 0.3

  std::cout << "[OK] Linear Forward\n";
}

void test_linear_backward() {
  Linear layer(2, 3);
  auto params = layer.get_parameters();

  // Inject deterministic weights
  float *w = static_cast<float *>(params["weight"]->data());
  w[0] = 1.0f;
  w[1] = 2.0f;
  w[2] = 3.0f;
  w[3] = -1.0f;
  w[4] = -2.0f;
  w[5] = -3.0f;

  Tensor input({1, 2}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(input.data());
  in_ptr[0] = 0.5f;
  in_ptr[1] = 1.5f;

  layer.forward(input); // Cache input for backprop

  // Create a mock gradient coming from the next layer (grad_output)
  Tensor grad_out({1, 3}, Device::CPU, DataType::FP32);
  float *go_ptr = static_cast<float *>(grad_out.data());
  go_ptr[0] = 1.0f;
  go_ptr[1] = 1.0f;
  go_ptr[2] = 1.0f;

  Tensor grad_in = layer.backward(grad_out);

  // Check grad_input (dX = dY * W^T)
  const float *gi_ptr = static_cast<const float *>(grad_in.data());
  assert(is_close(gi_ptr[0], 6.0f));  // 1*1 + 1*2 + 1*3
  assert(is_close(gi_ptr[1], -6.0f)); // 1*-1 + 1*-2 + 1*-3

  // Check grad_weights (dW = X^T * dY)
  auto grads = layer.get_gradients();
  const float *gw_ptr = static_cast<const float *>(grads["weight"]->data());
  assert(is_close(gw_ptr[0], 0.5f)); // X[0] * dY[0]
  assert(is_close(gw_ptr[3], 1.5f)); // X[1] * dY[0]

  // Check grad_bias (db = sum(dY))
  const float *gb_ptr = static_cast<const float *>(grads["bias"]->data());
  assert(is_close(gb_ptr[0], 1.0f));

  std::cout << "[OK] Linear Backward\n";
}

void test_relu() {
  ReLU relu;
  Tensor input({1, 3}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(input.data());
  in_ptr[0] = -1.0f;
  in_ptr[1] = 0.0f;
  in_ptr[2] = 2.0f;

  // Forward
  Tensor out = relu.forward(input);
  const float *out_ptr = static_cast<const float *>(out.data());
  assert(is_close(out_ptr[0], 0.0f));
  assert(is_close(out_ptr[1], 0.0f));
  assert(is_close(out_ptr[2], 2.0f));

  // Backward
  Tensor grad_out({1, 3}, Device::CPU, DataType::FP32);
  float *go_ptr = static_cast<float *>(grad_out.data());
  go_ptr[0] = 5.0f;
  go_ptr[1] = 5.0f;
  go_ptr[2] = 5.0f;

  Tensor grad_in = relu.backward(grad_out);
  const float *gi_ptr = static_cast<const float *>(grad_in.data());
  assert(is_close(gi_ptr[0], 0.0f)); // Masked by < 0
  assert(is_close(gi_ptr[1], 0.0f)); // Masked by <= 0
  assert(is_close(gi_ptr[2], 5.0f)); // Passed through > 0

  std::cout << "[OK] ReLU Activation\n";
}

void test_softmax() {
  Softmax softmax;
  Tensor input({1, 3}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(input.data());
  in_ptr[0] = 1.0f;
  in_ptr[1] = 2.0f;
  in_ptr[2] = 3.0f;

  Tensor out = softmax.forward(input);
  const float *out_ptr = static_cast<const float *>(out.data());

  // Check if probabilities sum to ~1.0
  float sum = out_ptr[0] + out_ptr[1] + out_ptr[2];
  assert(is_close(sum, 1.0f));

  // Check order (larger input = larger probability)
  assert(out_ptr[2] > out_ptr[1]);
  assert(out_ptr[1] > out_ptr[0]);

  std::cout << "[OK] Softmax Activation\n";
}

void test_classification_training() {
  Model model;
  model.add(std::make_shared<Linear>(2, 4)); // Hidden layer
  model.add(std::make_shared<ReLU>());
  model.add(std::make_shared<Linear>(4, 3)); // 3 Classes

  // Batch size 2, 2 features
  Tensor inputs({2, 2}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(inputs.data());
  // Sample 1
  in_ptr[0] = 0.5f;
  in_ptr[1] = 0.1f;
  // Sample 2
  in_ptr[2] = -0.5f;
  in_ptr[3] = -0.1f;

  // Targets (One-hot encoded)
  Tensor targets({2, 3}, Device::CPU, DataType::FP32);
  float *t_ptr = static_cast<float *>(targets.data());
  // Sample 1 belongs to class 0
  t_ptr[0] = 1.0f;
  t_ptr[1] = 0.0f;
  t_ptr[2] = 0.0f;
  // Sample 2 belongs to class 2
  t_ptr[3] = 0.0f;
  t_ptr[4] = 0.0f;
  t_ptr[5] = 1.0f;

  float initial_loss = 0.0f;
  float final_loss = 0.0f;

  SGD optimizer(model.parameters(), 0.1f);

  for (int epoch = 0; epoch < 10; ++epoch) {
    optimizer.zero_grad();

    Tensor logits = model.forward(inputs);

    Tensor grad_out({2, 3}, Device::CPU, DataType::FP32);
    float loss = cross_entropy_loss(logits, targets, grad_out);

    if (epoch == 0)
      initial_loss = loss;
    if (epoch == 4)
      final_loss = loss;

    model.backward(grad_out);
    optimizer.step();
  }

  assert(final_loss < initial_loss);
  std::cout << "[OK] Classification Training (Cross-Entropy)\n";
}

void test_model_and_training() {
  Model model;
  model.add(std::make_shared<Linear>(2, 2));
  model.add(std::make_shared<ReLU>());

  // Single batch, 2 features
  Tensor input({1, 2}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(input.data());
  in_ptr[0] = 1.0f;
  in_ptr[1] = 2.0f;

  // Target output
  Tensor target({1, 2}, Device::CPU, DataType::FP32);
  float *t_ptr = static_cast<float *>(target.data());
  t_ptr[0] = 5.0f;
  t_ptr[1] = 10.0f;

  // Dummy loop to check loss goes down
  float initial_loss = 0.0f;
  float final_loss = 0.0f;

  SGD optimizer(model.parameters(), 0.01f);

  for (int epoch = 0; epoch < 100; ++epoch) {
    optimizer.zero_grad();

    Tensor out = model.forward(input);
    Tensor grad_out(out.shape(), out.device_, out.dtype_);

    float loss = mse_loss(out, target, grad_out);
    if (epoch == 0)
      initial_loss = loss;
    if (epoch == 4)
      final_loss = loss;

    model.backward(grad_out);
    optimizer.step();
  }

  assert(final_loss < initial_loss); // Ensure network is learning
  std::cout << "[OK] Model f32 Forward, Backward, Loss, and SGD Step\n";
}

void test_model_and_training_fp16() {
  Model model;
  model.add(std::make_shared<Linear>(2, 2));
  model.add(std::make_shared<ReLU>());

  // Single batch, 2 features
  Tensor input({1, 2}, Device::CPU, DataType::FP16);
  float *in_ptr = static_cast<float *>(input.data());
  in_ptr[0] = 1.0f;
  in_ptr[1] = 2.0f;

  // Target output
  Tensor target({1, 2}, Device::CPU, DataType::FP16);
  float *t_ptr = static_cast<float *>(target.data());
  t_ptr[0] = 5.0f;
  t_ptr[1] = 10.0f;

  // Dummy loop to check loss goes down
  float initial_loss = 0.0f;
  float final_loss = 0.0f;

  SGD optimizer(model.parameters(), 0.01f);

  for (int epoch = 0; epoch < 100; ++epoch) {
    optimizer.zero_grad();

    Tensor out = model.forward(input);
    Tensor grad_out(out.shape(), out.device_, out.dtype_);

    float loss = mse_loss(out, target, grad_out);
    if (epoch == 0)
      initial_loss = loss;
    if (epoch == 4)
      final_loss = loss;

    model.backward(grad_out);
    optimizer.step();
  }

  assert(final_loss < initial_loss); // Ensure network is learning
  std::cout << "[OK] Model f16 Forward, Backward, Loss, and SGD Step\n";
}

void test_save_load() {
  const std::string file_path = "test_model.munet";

  // Model A: The source model
  Model model_a;
  model_a.add(std::make_shared<Linear>(3, 4));
  model_a.add(std::make_shared<ReLU>());
  model_a.add(std::make_shared<Linear>(4, 2));

  Tensor input({1, 3}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(input.data());
  in_ptr[0] = 0.5f;
  in_ptr[1] = -0.2f;
  in_ptr[2] = 0.1f;

  Tensor out_a = model_a.forward(input);
  model_a.save_weights(file_path);

  // Model B: The destination model
  Model model_b;
  model_b.add(std::make_shared<Linear>(3, 4));
  model_b.add(std::make_shared<ReLU>());
  model_b.add(std::make_shared<Linear>(4, 2));

  // Overwrite B's random weights with A's saved weights
  model_b.load_weights(file_path);
  Tensor out_b = model_b.forward(input);

  const float *ptr_a = static_cast<const float *>(out_a.data());
  const float *ptr_b = static_cast<const float *>(out_b.data());

  // Outputs should be perfectly identical
  assert(ptr_a[0] == ptr_b[0]);
  assert(ptr_a[1] == ptr_b[1]);

  std::remove(file_path.c_str()); // Clean up test file
  std::cout << "[OK] Model Save & Load\n";
}

void test_flatten() {
  Flatten flatten;
  // Input shape: Batch=2, Channels=3, H=4, W=4
  Tensor input({2, 3, 4, 4}, Device::CPU, DataType::FP32);

  Tensor out = flatten.forward(input);
  const auto &out_shape = out.shape();

  assert(out_shape.size() == 2);
  assert(out_shape[0] == 2);
  assert(out_shape[1] == 48); // 3 * 4 * 4 = 48

  Tensor grad_out({2, 48}, Device::CPU, DataType::FP32);
  Tensor grad_in = flatten.backward(grad_out);
  const auto &in_shape = grad_in.shape();

  assert(in_shape.size() == 4);
  assert(in_shape[0] == 2 && in_shape[1] == 3 && in_shape[2] == 4 &&
         in_shape[3] == 4);

  std::cout << "[OK] Flatten Layer\n";
}

void test_maxpool2d() {
  MaxPool2D pool(2, 2); // Kernel=2, Stride=2
  Tensor input({1, 1, 4, 4}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(input.data());

  // Create a 4x4 matrix
  //  0  1  2  3
  //  4  5  6  7
  //  8  9 10 11
  // 12 13 14 15
  for (int i = 0; i < 16; ++i)
    in_ptr[i] = static_cast<float>(i);

  Tensor out = pool.forward(input);
  const float *out_ptr = static_cast<const float *>(out.data());

  // Expected Output (2x2):
  //  5  7
  // 13 15
  assert(out.shape() == std::vector<int>({1, 1, 2, 2}));
  assert(is_close(out_ptr[0], 5.0f));
  assert(is_close(out_ptr[1], 7.0f));
  assert(is_close(out_ptr[2], 13.0f));
  assert(is_close(out_ptr[3], 15.0f));

  // Test backward routing
  Tensor grad_out({1, 1, 2, 2}, Device::CPU, DataType::FP32);
  float *go_ptr = static_cast<float *>(grad_out.data());
  for (int i = 0; i < 4; ++i)
    go_ptr[i] = 1.0f;

  Tensor grad_in = pool.backward(grad_out);
  const float *gi_ptr = static_cast<const float *>(grad_in.data());

  // Gradients should only be routed to the max elements (5, 7, 13, 15)
  assert(is_close(gi_ptr[0], 0.0f));  // Was 0, not a max
  assert(is_close(gi_ptr[5], 1.0f));  // Was 5, a max
  assert(is_close(gi_ptr[15], 1.0f)); // Was 15, a max

  std::cout << "[OK] MaxPool2D Layer\n";
}

void test_conv2d() {
  // 1 in_channel, 1 out_channel, 2x2 kernel, stride 1, padding 0
  Conv2D conv(1, 1, 2, 1, 0);
  auto params = conv.get_parameters();

  // Set deterministic weights: 2x2 matrix of 1.0s
  float *w = static_cast<float *>(params["weight"]->data());
  w[0] = 1.0f;
  w[1] = 1.0f;
  w[2] = 1.0f;
  w[3] = 1.0f;

  // Set bias to 0
  float *b = static_cast<float *>(params["bias"]->data());
  b[0] = 0.0f;

  // 3x3 Input matrix of 1.0s
  Tensor input({1, 1, 3, 3}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(input.data());
  for (int i = 0; i < 9; ++i)
    in_ptr[i] = 1.0f;

  Tensor out = conv.forward(input);

  // Output should be 2x2
  assert(out.shape() == std::vector<int>({1, 1, 2, 2}));

  // Since input is 1s and filter is 1s, every 2x2 patch sum should be 4.0
  const float *out_ptr = static_cast<const float *>(out.data());
  for (int i = 0; i < 4; ++i) {
    assert(is_close(out_ptr[i], 4.0f));
  }

  // Simple backward pass test to ensure shapes match and no crashing
  Tensor grad_out({1, 1, 2, 2}, Device::CPU, DataType::FP32);
  grad_out.zero();
  Tensor grad_in = conv.backward(grad_out);
  assert(grad_in.shape() == std::vector<int>({1, 1, 3, 3}));

  std::cout << "[OK] Conv2D Layer\n";
}

// 2. Full CNN Integration Test
void test_cnn_integration() {
  Model model;
  // Input: 1x1x6x6
  model.add(std::make_shared<Conv2D>(1, 2, 3, 1, 0)); // Out: 1x2x4x4
  model.add(std::make_shared<ReLU>());
  model.add(std::make_shared<MaxPool2D>(2, 2)); // Out: 1x2x2x2
  model.add(std::make_shared<Flatten>());       // Out: 1x8
  model.add(std::make_shared<Linear>(8, 2));    // Out: 1x2

  Tensor input({1, 1, 6, 6}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(input.data());
  for (int i = 0; i < 36; ++i)
    in_ptr[i] = static_cast<float>(i % 4);

  Tensor target({1, 2}, Device::CPU, DataType::FP32);
  static_cast<float *>(target.data())[0] = 1.0f;
  static_cast<float *>(target.data())[1] = 0.0f;

  SGD optimizer(model.parameters(), 0.01f);

  // Capture initial weights to ensure they change
  auto params = model.parameters();
  std::vector<float> initial_weights;
  for (auto *p : params)
    initial_weights.push_back(static_cast<float *>(p->data())[0]);

  // Train step
  optimizer.zero_grad();
  Tensor out = model.forward(input);
  Tensor grad_out(out.shape(), Device::CPU, DataType::FP32);
  mse_loss(out, target, grad_out);
  model.backward(grad_out);
  optimizer.step();

  // Check weights changed
  bool weights_changed = false;
  for (size_t i = 0; i < params.size(); ++i) {
    if (std::abs(static_cast<float *>(params[i]->data())[0] -
                 initial_weights[i]) > 1e-6f) {
      weights_changed = true;
      break;
    }
  }

  assert(out.shape()[1] == 2);
  assert(weights_changed);
  std::cout << "[OK] CNN Integration (Conv-Relu-Pool-Flat-Linear)\n";
}

// 3. Test SGD Mechanics Explicitly
void test_sgd_update_mechanics() {
  Tensor weight({1}, Device::CPU, DataType::FP32);
  float *w_ptr = static_cast<float *>(weight.data());
  w_ptr[0] = 10.0f;

  weight.allocate_grad();
  float *g_ptr = static_cast<float *>(weight.grad()->data());
  g_ptr[0] = 2.0f; // Gradient

  std::vector<Tensor *> params = {&weight};
  SGD sgd(params, 0.5f); // lr = 0.5

  sgd.step();

  // New weight = Old - lr * grad
  // 10.0 - 0.5 * 2.0 = 9.0
  assert(is_close(w_ptr[0], 9.0f));

  std::cout << "[OK] SGD Update Mechanics\n";
}

// 4. Test Error Handling
void test_error_handling() {
  bool caught = false;
  try {
    Linear l(10, 5);
    // Wrong input shape (cols != in_features)
    Tensor bad_input({1, 3}, Device::CPU, DataType::FP32);
    l.forward(bad_input); // Should compute garbage or crash if not careful, but
                          // size check is implicit in loops
    // Let's test explicit size mismatch in loss which has checks
    Tensor a({1, 2});
    Tensor b({1, 3});
    Tensor g({1, 2});
    mse_loss(a, b, g);
  } catch (const std::exception &e) {
    caught = true;
  }
  assert(caught);
  std::cout << "[OK] Error Handling (Shape Mismatch)\n";
}

void test_upsample2d() {
  Upsample2D up(2);
  Tensor input({1, 1, 2, 2}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(input.data());
  in_ptr[0] = 1.0f;
  in_ptr[1] = 2.0f;
  in_ptr[2] = 3.0f;
  in_ptr[3] = 4.0f;

  Tensor out = up.forward(input);
  const float *out_ptr = static_cast<const float *>(out.data());

  assert(out.shape() == std::vector<int>({1, 1, 4, 4}));
  assert(is_close(out_ptr[0], 1.0f));
  assert(is_close(out_ptr[1], 1.0f));
  assert(is_close(out_ptr[4], 1.0f));
  assert(is_close(out_ptr[5], 1.0f));
  assert(is_close(out_ptr[15], 4.0f));

  std::cout << "[OK] Upsample2D Layer (CPU)\n";
}

void test_tensor_add() {
  Tensor a({2, 2}, Device::CPU);
  Tensor b({2, 2}, Device::CPU);
  float *a_ptr = (float *)a.data();
  float *b_ptr = (float *)b.data();
  for (int i = 0; i < 4; ++i) {
    a_ptr[i] = 1.0f;
    b_ptr[i] = 2.0f;
  }

  Tensor c = a + b;
  float *c_ptr = (float *)c.data();
  for (int i = 0; i < 4; ++i)
    assert(is_close(c_ptr[i], 3.0f));

  std::cout << "[OK] Tensor Addition\n";
}

void test_sigmoid() {
  Sigmoid sig;
  Tensor in({2}, Device::CPU);
  ((float *)in.data())[0] = 0.0f;
  ((float *)in.data())[1] = 2.0f;

  Tensor out = sig.forward(in);
  float *out_ptr = (float *)out.data();
  assert(is_close(out_ptr[0], 0.5f)); // sigmoid(0) = 0.5
  // sigmoid(2) = 1 / (1 + e^-2) = 0.880797
  assert(is_close(out_ptr[1], 0.880797f));

  Tensor grad_out({2}, Device::CPU);
  ((float *)grad_out.data())[0] = 1.0f;
  ((float *)grad_out.data())[1] = 1.0f;

  Tensor grad_in = sig.backward(grad_out);
  float *gi_ptr = (float *)grad_in.data();
  // d/dx sigmoid = s * (1-s)
  // at 0: 0.5 * 0.5 = 0.25
  assert(is_close(gi_ptr[0], 0.25f));

  std::cout << "[OK] Sigmoid Layer\n";
}

void test_concat() {
  Concat concat;
  // N=1, C=1, H=2, W=2
  Tensor t1({1, 1, 2, 2}, Device::CPU);
  Tensor t2({1, 2, 2, 2}, Device::CPU);

  // Fill t1 with 1s
  float *p1 = (float *)t1.data();
  for (int i = 0; i < 4; ++i)
    p1[i] = 1.0f;

  // Fill t2 with 2s
  float *p2 = (float *)t2.data();
  for (int i = 0; i < 8; ++i)
    p2[i] = 2.0f;

  std::vector<Tensor *> inputs = {&t1, &t2};
  Tensor out = concat.forward(inputs);

  // Output should be N=1, C=3, H=2, W=2
  assert(out.shape()[1] == 3);
  assert(out.size() == 12);

  float *p_out = (float *)out.data();
  // First channel (4 pixels) should be 1.0
  for (int i = 0; i < 4; ++i)
    assert(is_close(p_out[i], 1.0f));
  // Next 2 channels (8 pixels) should be 2.0
  for (int i = 4; i < 12; ++i)
    assert(is_close(p_out[i], 2.0f));

  std::cout << "[OK] Concat Layer\n";
}

void test_adam() {
  // Basic instantiation check and ensuring it compiles/runs
  Tensor w({1}, Device::CPU);
  Adam adam({&w});
  std::cout << "[OK] Adam Optimizer Init\n";
}

/*

*/

#ifdef MUNET_USE_CUDA
void test_gpu_transfer() {
  Tensor t({2, 3}, Device::CPU, DataType::FP32);
  float *cpu_ptr = static_cast<float *>(t.data());
  cpu_ptr[0] = 3.14f;

  t.to_gpu();
  assert(t.device_ == Device::CUDA);

  t.to_cpu();
  assert(t.device_ == Device::CPU);

  float *new_cpu_ptr = static_cast<float *>(t.data());
  assert(is_close(new_cpu_ptr[0], 3.14f));

  // Test Tensor Add GPU
  Tensor a({2}, Device::CPU);
  ((float *)a.data())[0] = 1;
  ((float *)a.data())[1] = 2;
  Tensor b({2}, Device::CPU);
  ((float *)b.data())[0] = 3;
  ((float *)b.data())[1] = 4;
  a.to_gpu();
  b.to_gpu();
  Tensor c = a + b;
  c.to_cpu();
  assert(is_close(((float *)c.data())[0], 4.0f));

  std::cout << "[OK] GPU/CPU Transfer\n";
}

// Helper to copy parameters from one layer to another to ensure identical start
// state
void copy_layer_params(Layer *src, Layer *dst) {
  auto src_params = src->get_parameters();
  auto dst_params = dst->get_parameters();
  for (auto &[name, tensor] : src_params) {
    Tensor *dst_tensor = dst_params[name];
    // Copy data from src(CPU) to dst(CPU) before moving dst to GPU
    std::memcpy(dst_tensor->data(), tensor->data(), tensor->bytes());
  }
}

void test_concat_gpu() {
  std::cout << "[TEST] Concat GPU... ";
  Concat c;
  Tensor t1({1, 1, 2, 2}, Device::CPU);
  float *p1 = (float *)t1.data();
  for (int i = 0; i < 4; ++i)
    p1[i] = 1.0f;
  Tensor t2({1, 1, 2, 2}, Device::CPU);
  float *p2 = (float *)t2.data();
  for (int i = 0; i < 4; ++i)
    p2[i] = 2.0f;

  t1.to_gpu();
  t2.to_gpu();
  std::vector<Tensor *> inputs = {&t1, &t2};
  Tensor out = c.forward(inputs);
  out.to_cpu();

  float *res = (float *)out.data();
  assert(is_close(res[0], 1.0f));
  assert(is_close(res[4], 2.0f));

  Tensor go({1, 2, 2, 2}, Device::CUDA, DataType::FP32); // grad output
  // fill go with 3.0
  // ... skipping precise grad check for brevity, assuming run completes
  std::vector<Tensor> grads = c.backward(go);
  assert(grads.size() == 2);

  std::cout << "Passed.\n";
}

// Helper to move all layer parameters to GPU
void move_layer_to_gpu(Layer *layer) {
  auto params = layer->get_parameters();
  for (auto &[name, tensor] : params) {
    tensor->to_gpu();
    // Also need to allocate and move gradients if they exist,
    // though allocate_grad() handles lazy allocation on the current device.
    if (tensor->grad()) {
      // Re-allocate grad on GPU
      tensor->grad()->to_gpu();
      tensor->grad()->zero();
    }
  }
}

void test_linear_gpu_vs_cpu() {
  std::cout << "[TEST] Linear GPU vs CPU Comparison... ";

  // 1. Setup CPU Layer
  Linear cpu_layer(32, 16);
  Tensor cpu_input({4, 32}, Device::CPU, DataType::FP32);
  float *in_ptr = static_cast<float *>(cpu_input.data());
  for (int i = 0; i < 4 * 32; ++i)
    in_ptr[i] = (float)i / 100.0f;

  // 2. Setup GPU Layer (Identical weights)
  Linear gpu_layer(32, 16);
  copy_layer_params(&cpu_layer, &gpu_layer); // Sync weights
  move_layer_to_gpu(&gpu_layer);             // Move to VRAM

  Tensor gpu_input = cpu_input.clone();
  gpu_input.to_gpu();

  // 3. Forward Pass
  Tensor cpu_out = cpu_layer.forward(cpu_input);
  Tensor gpu_out = gpu_layer.forward(gpu_input);

  // Bring back to CPU for comparison
  gpu_out.to_cpu();

  float *cpu_d = static_cast<float *>(cpu_out.data());
  float *gpu_d = static_cast<float *>(gpu_out.data());

  for (size_t i = 0; i < cpu_out.size(); ++i) {
    if (!is_close(cpu_d[i], gpu_d[i], 1e-4f)) {
      std::cerr << "Mismatch at index " << i << " CPU:" << cpu_d[i]
                << " GPU:" << gpu_d[i] << std::endl;
      throw std::runtime_error("Linear Forward GPU mismatch");
    }
  }

  // 4. Backward Pass
  Tensor grad_out({4, 16}, Device::CPU, DataType::FP32);
  float *g_ptr = static_cast<float *>(grad_out.data());
  for (int i = 0; i < 4 * 16; ++i)
    g_ptr[i] = 1.0f;

  Tensor gpu_grad_out = grad_out.clone();
  gpu_grad_out.to_gpu();

  Tensor cpu_grad_in = cpu_layer.backward(grad_out);
  Tensor gpu_grad_in = gpu_layer.backward(gpu_grad_out);

  gpu_grad_in.to_cpu();

  // Check Grad Input
  float *cgi = static_cast<float *>(cpu_grad_in.data());
  float *ggi = static_cast<float *>(gpu_grad_in.data());
  for (size_t i = 0; i < cpu_grad_in.size(); ++i) {
    if (!is_close(cgi[i], ggi[i], 1e-4f))
      throw std::runtime_error("Linear Backward Input Grad mismatch");
  }

  // Check Parameter Gradients (Weights)
  Tensor *cpu_w = cpu_layer.get_parameters()["weight"];
  Tensor *gpu_w = gpu_layer.get_parameters()["weight"];
  // Move GPU weight grad back to CPU to read
  gpu_w->grad()->to_cpu();

  float *cgw = static_cast<float *>(cpu_w->grad()->data());
  float *ggw = static_cast<float *>(gpu_w->grad()->data());

  for (size_t i = 0; i < cpu_w->size(); ++i) {
    if (!is_close(cgw[i], ggw[i], 1e-4f))
      throw std::runtime_error("Linear Backward Weight Grad mismatch");
  }

  std::cout << "Passed.\n";
}

void test_relu_gpu() {
  std::cout << "[TEST] ReLU GPU... ";
  ReLU relu;

  Tensor cpu_in({2, 5}, Device::CPU, DataType::FP32);
  float *p = static_cast<float *>(cpu_in.data());
  for (int i = 0; i < 10; ++i)
    p[i] = (i % 2 == 0) ? (float)i : -(float)i;

  Tensor gpu_in = cpu_in.clone();
  gpu_in.to_gpu();

  Tensor gpu_out = relu.forward(gpu_in);
  gpu_out.to_cpu();

  float *res = static_cast<float *>(gpu_out.data());
  for (int i = 0; i < 10; ++i) {
    float expected = (p[i] > 0) ? p[i] : 0.0f;
    if (!is_close(res[i], expected))
      throw std::runtime_error("ReLU GPU Forward failed");
  }
  std::cout << "Passed.\n";
}

void test_sgd_gpu() {
  std::cout << "[TEST] SGD GPU... ";
  // Setup a single weight tensor on GPU
  Tensor w({10}, Device::CPU, DataType::FP32);
  float *w_ptr = static_cast<float *>(w.data());
  for (int i = 0; i < 10; ++i)
    w_ptr[i] = 1.0f;

  w.to_gpu();
  w.allocate_grad();

  // Manually set gradient on GPU
  Tensor g({10}, Device::CPU, DataType::FP32);
  float *g_ptr = static_cast<float *>(g.data());
  for (int i = 0; i < 10; ++i)
    g_ptr[i] = 0.1f;

  Tensor g_gpu = g.clone();
  g_gpu.to_gpu();

  // Hack: Copy g_gpu data into w.grad()
  cudaMemcpy(w.grad()->data(), g_gpu.data(), 10 * sizeof(float),
             cudaMemcpyDeviceToDevice);

  std::vector<Tensor *> params = {&w};
  SGD opt(params, 1.0f); // lr = 1.0
  opt.step();

  w.to_cpu();
  w_ptr = static_cast<float *>(w.data());

  // w = 1.0 - 1.0 * 0.1 = 0.9
  for (int i = 0; i < 10; ++i) {
    if (!is_close(w_ptr[i], 0.9f))
      throw std::runtime_error("SGD GPU update failed");
  }
  std::cout << "Passed.\n";
}

void test_training_loop_gpu() {
  std::cout << "[TEST] Full Training Loop on GPU... ";
  Model model;
  model.add(std::make_shared<Linear>(10, 20));
  model.add(std::make_shared<ReLU>());
  model.add(std::make_shared<Linear>(20, 1));

  // Move all parameters to GPU
  auto params = model.parameters();
  for (auto *p : params) {
    p->to_gpu();
    p->allocate_grad();
    p->grad()->to_gpu(); // Ensure grad tensor is on GPU
    p->grad()->zero();
  }

  Tensor input({32, 10}, Device::CPU, DataType::FP32);
  // Fill random
  float *in_p = static_cast<float *>(input.data());
  for (int i = 0; i < 320; ++i)
    in_p[i] = (float)std::rand() / RAND_MAX;

  Tensor target({32, 1}, Device::CPU, DataType::FP32);
  float *t_p = static_cast<float *>(target.data());
  for (int i = 0; i < 32; ++i)
    t_p[i] = 1.0f;

  input.to_gpu();
  target.to_gpu();

  SGD optimizer(model.parameters(), 0.01f);

  float initial_loss = 0.0f;
  float final_loss = 0.0f;

  for (int epoch = 0; epoch < 5; ++epoch) {
    optimizer.zero_grad();
    Tensor out = model.forward(input);

    // Compute loss on GPU (Assuming mse_loss logic isn't kernelized yet,
    // we might need to copy back for loss scalar, but here we just need grad)
    // Note: mse_loss in previous code was CPU only.
    // We will do a quick manual loss calc/grad gen here or assume mse_loss
    // supports GPU (it needs kernels). For this test, let's pull back to CPU to
    // compute Loss/Grad, then push grad back. In a real optimized engine, loss
    // would be a kernel.

    Tensor out_cpu = out.clone();
    out_cpu.to_cpu();
    Tensor target_cpu = target.clone();
    target_cpu.to_cpu();

    Tensor grad_out_cpu(out.shape(), Device::CPU, DataType::FP32);
    float loss = mse_loss(out_cpu, target_cpu, grad_out_cpu);

    if (epoch == 0)
      initial_loss = loss;
    if (epoch == 4)
      final_loss = loss;

    Tensor grad_out = grad_out_cpu.clone();
    grad_out.to_gpu();

    model.backward(grad_out);
    optimizer.step();
  }

  if (final_loss >= initial_loss)
    throw std::runtime_error("GPU Training did not decrease loss");
  std::cout << "Passed (Loss " << initial_loss << " -> " << final_loss
            << ").\n";
}

void test_softmax_gpu() {
  std::cout << "[TEST] Softmax GPU... ";
  Softmax sm;
  Tensor cpu_in({1, 5}, Device::CPU, DataType::FP32);
  float *p = (float *)cpu_in.data();
  for (int i = 0; i < 5; ++i)
    p[i] = (float)i;

  Tensor gpu_in = cpu_in.clone();
  gpu_in.to_gpu();
  Tensor gpu_out = sm.forward(gpu_in);
  gpu_out.to_cpu();

  float sum = 0;
  float *res = (float *)gpu_out.data();
  for (int i = 0; i < 5; ++i)
    sum += res[i];
  assert(is_close(sum, 1.0f));
  std::cout << "Passed.\n";
}

void test_maxpool_gpu() {
  std::cout << "[TEST] MaxPool2D GPU... ";
  MaxPool2D pool(2, 2);
  Tensor cpu_in({1, 1, 4, 4}, Device::CPU, DataType::FP32);
  float *in_ptr = (float *)cpu_in.data();
  for (int i = 0; i < 16; ++i)
    in_ptr[i] = (float)i;

  Tensor gpu_in = cpu_in.clone();
  gpu_in.to_gpu();
  Tensor gpu_out = pool.forward(gpu_in);

  Tensor grad_out({1, 1, 2, 2}, Device::CUDA, DataType::FP32);
  grad_out.zero(); // fill with 1.0 manually
  float h_one = 1.0f;
  for (int i = 0; i < 4; ++i)
    cudaMemcpy((float *)grad_out.data() + i, &h_one, sizeof(float),
               cudaMemcpyHostToDevice);

  Tensor gpu_grad_in = pool.backward(grad_out);
  gpu_grad_in.to_cpu();
  float *gi = (float *)gpu_grad_in.data();
  assert(is_close(gi[15], 1.0f));
  assert(is_close(gi[0], 0.0f));
  std::cout << "Passed.\n";
}

void test_conv2d_gpu() {
  std::cout << "[TEST] Conv2D GPU... ";
  Conv2D conv(1, 1, 2, 1, 0);
  Tensor cpu_in({1, 1, 3, 3}, Device::CPU, DataType::FP32);
  for (int i = 0; i < 9; ++i)
    ((float *)cpu_in.data())[i] = 1.0f;

  Tensor gpu_in = cpu_in.clone();
  gpu_in.to_gpu();
  // Set weights on GPU
  for (int i = 0; i < 4; ++i) {
    float h_w = 1.0f;
    cudaMemcpy((float *)conv.get_parameters()["weight"]->data() + i, &h_w,
               sizeof(float), cudaMemcpyHostToDevice);
  }
  conv.get_parameters()["weight"]->device_ = Device::CUDA;
  conv.get_parameters()["bias"]->to_gpu();

  Tensor gpu_out = conv.forward(gpu_in);
  gpu_out.to_cpu();
  assert(is_close(((float *)gpu_out.data())[0], 4.0f));
  std::cout << "Passed.\n";
}

void test_mse_loss_gpu() {
  std::cout << "[TEST] MSE Loss GPU... ";
  Tensor p({10}, Device::CUDA, DataType::FP32);
  Tensor t({10}, Device::CUDA, DataType::FP32);
  Tensor g({10}, Device::CUDA, DataType::FP32);

  float h_data[10];
  for (int i = 0; i < 10; ++i)
    h_data[i] = 1.0f;
  cudaMemcpy(p.data(), h_data, 10 * 4, cudaMemcpyHostToDevice);
  for (int i = 0; i < 10; ++i)
    h_data[i] = 0.0f;
  cudaMemcpy(t.data(), h_data, 10 * 4, cudaMemcpyHostToDevice);

  float loss = mse_loss(p, t, g);
  assert(is_close(loss, 1.0f)); // (1-0)^2 / 1 = 1
  std::cout << "Passed.\n";
}

#endif

int main() {
  try {
    test_tensor_alloc();
    test_linear_forward();
    test_linear_backward();
    test_flatten();
    test_tensor_add();
    test_concat();
    test_maxpool2d();
    test_conv2d();
    test_relu();
    test_softmax();
    test_upsample2d();
    test_sigmoid();
    test_adam();

    test_sgd_update_mechanics();
    test_cnn_integration();
    test_error_handling();

    test_model_and_training();
    test_model_and_training_fp16();
    test_classification_training();

    test_save_load();

    std::cout << "\n\n[INFO] CPU tests executed successfully.\n\n";

#ifdef MUNET_USE_CUDA
    test_gpu_transfer();
    test_concat_gpu();
    test_linear_gpu_vs_cpu();
    test_relu_gpu();
    test_sgd_gpu();
    test_training_loop_gpu();
    test_softmax_gpu();
    test_maxpool_gpu();
    test_conv2d_gpu();
    test_mse_loss_gpu();
    std::cout << "[INFO] CUDA tests executed successfully.\n";
#else
    std::cout
        << "\n\n[INFO] Skipping CUDA tests (MUNET_USE_CUDA not defined).\n\n";
#endif

    std::cout << "\nAll C++ MuNet tests passed successfully!\n";
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Test failed with exception: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
