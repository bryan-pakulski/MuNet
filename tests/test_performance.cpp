#include "tensor.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

using namespace munet;

namespace {

bool perf_tests_enabled() {
  const char *env = std::getenv("MUNET_RUN_PERF_TESTS");
  return env != nullptr && std::string(env) == "1";
}

bool has_device(DeviceType type) {
  auto devices = test::get_available_devices();
  return std::any_of(devices.begin(), devices.end(),
                     [type](const Device &d) { return d.type == type; });
}

double get_env_ratio(const char *name, double default_value) {
  const char *value = std::getenv(name);
  if (!value)
    return default_value;
  try {
    return std::stod(value);
  } catch (...) {
    return default_value;
  }
}

double benchmark_ms(const std::function<void()> &fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i)
    fn();

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
    fn();
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double, std::milli>(end - start).count() /
         static_cast<double>(iters);
}

void require_gpu_backends() {
#ifdef MUNET_USE_CUDA
  const bool has_cuda = has_device(DeviceType::CUDA);
#else
  const bool has_cuda = false;
#endif

#ifdef MUNET_USE_VULKAN
  const bool has_vulkan = has_device(DeviceType::VULKAN);
#else
  const bool has_vulkan = false;
#endif

  if (!perf_tests_enabled()) {
    GTEST_SKIP() << "Set MUNET_RUN_PERF_TESTS=1 to run performance tests.";
  }
  if (!has_cuda || !has_vulkan) {
    GTEST_SKIP() << "Performance comparison requires both CUDA and Vulkan "
                    "devices in this environment.";
  }
}

void run_perf_ratio_test(const std::string &name,
                         const std::function<void(Device)> &runner,
                         int warmup, int iters, const char *ratio_env,
                         double default_ratio) {
  Device cuda{DeviceType::CUDA, 0};
  Device vk{DeviceType::VULKAN, 0};

  const double cuda_ms = benchmark_ms([&]() { runner(cuda); }, warmup, iters);
  const double vk_ms = benchmark_ms([&]() { runner(vk); }, warmup, iters);
  const double ratio = vk_ms / std::max(cuda_ms, 1e-9);

  std::cout << "[PERF] " << name << " cuda_ms=" << cuda_ms
            << " vk_ms=" << vk_ms << " ratio=" << ratio << std::endl;

  const double max_ratio = get_env_ratio(ratio_env, default_ratio);
  EXPECT_LE(ratio, max_ratio)
      << "Vulkan is too slow relative to CUDA for test " << name
      << ". ratio=" << ratio << " max=" << max_ratio;
}

} // namespace

TEST(PerformanceTest, ElementwiseAddCudaVsVulkan) {
  require_gpu_backends();

  constexpr int N = 1 << 20;
  Tensor a_cpu({N}, {DeviceType::CPU, 0});
  Tensor b_cpu({N}, {DeviceType::CPU, 0});
  a_cpu.uniform_(0.0f, 1.0f);
  b_cpu.uniform_(0.0f, 1.0f);

  Tensor a_cuda = a_cpu.to({DeviceType::CUDA, 0});
  Tensor b_cuda = b_cpu.to({DeviceType::CUDA, 0});
  Tensor a_vk = a_cpu.to({DeviceType::VULKAN, 0});
  Tensor b_vk = b_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "ElementwiseAdd",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA) ? (a_cuda + b_cuda)
                                                     : (a_vk + b_vk);
        out.impl_->backend().synchronize();
      },
      10, 80, "MUNET_PERF_MAX_RATIO_ADD", 3.0);
}

TEST(PerformanceTest, ElementwiseMulCudaVsVulkan) {
  require_gpu_backends();

  constexpr int N = 1 << 20;
  Tensor a_cpu({N}, {DeviceType::CPU, 0});
  Tensor b_cpu({N}, {DeviceType::CPU, 0});
  a_cpu.uniform_(0.0f, 1.0f);
  b_cpu.uniform_(0.0f, 1.0f);

  Tensor a_cuda = a_cpu.to({DeviceType::CUDA, 0});
  Tensor b_cuda = b_cpu.to({DeviceType::CUDA, 0});
  Tensor a_vk = a_cpu.to({DeviceType::VULKAN, 0});
  Tensor b_vk = b_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "ElementwiseMul",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA) ? (a_cuda * b_cuda)
                                                     : (a_vk * b_vk);
        out.impl_->backend().synchronize();
      },
      10, 80, "MUNET_PERF_MAX_RATIO_MUL", 3.0);
}

TEST(PerformanceTest, MatmulCudaVsVulkan) {
  require_gpu_backends();

  constexpr int M = 256;
  constexpr int K = 256;
  constexpr int N = 256;

  Tensor a_cpu({M, K}, {DeviceType::CPU, 0});
  Tensor b_cpu({K, N}, {DeviceType::CPU, 0});
  a_cpu.uniform_(-1.0f, 1.0f);
  b_cpu.uniform_(-1.0f, 1.0f);

  Tensor a_cuda = a_cpu.to({DeviceType::CUDA, 0});
  Tensor b_cuda = b_cpu.to({DeviceType::CUDA, 0});
  Tensor a_vk = a_cpu.to({DeviceType::VULKAN, 0});
  Tensor b_vk = b_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "Matmul",
      [&](Device dev) {
        Tensor out =
            (dev.type == DeviceType::CUDA) ? a_cuda.matmul(b_cuda)
                                           : a_vk.matmul(b_vk);
        out.impl_->backend().synchronize();
      },
      4, 20, "MUNET_PERF_MAX_RATIO_MATMUL", 4.0);
}

TEST(PerformanceTest, ReluCudaVsVulkan) {
  require_gpu_backends();

  constexpr int N = 1 << 22;
  Tensor x_cpu({N}, {DeviceType::CPU, 0});
  x_cpu.uniform_(-2.0f, 2.0f);

  Tensor x_cuda = x_cpu.to({DeviceType::CUDA, 0});
  Tensor x_vk = x_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "ReLU",
      [&](Device dev) {
        Tensor out =
            (dev.type == DeviceType::CUDA) ? x_cuda.relu() : x_vk.relu();
        out.impl_->backend().synchronize();
      },
      10, 80, "MUNET_PERF_MAX_RATIO_RELU", 3.0);
}

TEST(PerformanceTest, SoftmaxCudaVsVulkan) {
  require_gpu_backends();

  constexpr int B = 512;
  constexpr int C = 512;
  Tensor x_cpu({B, C}, {DeviceType::CPU, 0});
  x_cpu.uniform_(-2.0f, 2.0f);

  Tensor x_cuda = x_cpu.to({DeviceType::CUDA, 0});
  Tensor x_vk = x_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "Softmax",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA) ? x_cuda.softmax(-1)
                                                     : x_vk.softmax(-1);
        out.impl_->backend().synchronize();
      },
      6, 30, "MUNET_PERF_MAX_RATIO_SOFTMAX", 4.0);
}

TEST(PerformanceTest, BroadcastAddCudaVsVulkan) {
  require_gpu_backends();

  constexpr int R = 2048;
  constexpr int C = 1024;
  Tensor a_cpu({R, C}, {DeviceType::CPU, 0});
  Tensor b_cpu({C}, {DeviceType::CPU, 0});
  a_cpu.uniform_(0.0f, 1.0f);
  b_cpu.uniform_(0.0f, 1.0f);

  Tensor a_cuda = a_cpu.to({DeviceType::CUDA, 0});
  Tensor b_cuda = b_cpu.to({DeviceType::CUDA, 0});
  Tensor a_vk = a_cpu.to({DeviceType::VULKAN, 0});
  Tensor b_vk = b_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "BroadcastAdd",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA) ? (a_cuda + b_cuda)
                                                     : (a_vk + b_vk);
        out.impl_->backend().synchronize();
      },
      8, 40, "MUNET_PERF_MAX_RATIO_BROADCAST_ADD", 4.0);
}

TEST(PerformanceTest, SigmoidCudaVsVulkan) {
  require_gpu_backends();

  constexpr int N = 1 << 22;
  Tensor x_cpu({N}, {DeviceType::CPU, 0});
  x_cpu.uniform_(-6.0f, 6.0f);

  Tensor x_cuda = x_cpu.to({DeviceType::CUDA, 0});
  Tensor x_vk = x_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "Sigmoid",
      [&](Device dev) {
        Tensor out =
            (dev.type == DeviceType::CUDA) ? x_cuda.sigmoid() : x_vk.sigmoid();
        out.impl_->backend().synchronize();
      },
      10, 60, "MUNET_PERF_MAX_RATIO_SIGMOID", 3.5);
}

TEST(PerformanceTest, ReduceSumCudaVsVulkan) {
  require_gpu_backends();

  constexpr int N = 1 << 22;
  Tensor x_cpu({N}, {DeviceType::CPU, 0});
  x_cpu.uniform_(-1.0f, 1.0f);

  Tensor x_cuda = x_cpu.to({DeviceType::CUDA, 0});
  Tensor x_vk = x_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "ReduceSum",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA) ? x_cuda.sum() : x_vk.sum();
        out.impl_->backend().synchronize();
      },
      8, 50, "MUNET_PERF_MAX_RATIO_SUM", 4.0);
}

TEST(PerformanceTest, MSELossCudaVsVulkan) {
  require_gpu_backends();

  constexpr int N = 1 << 20;
  Tensor pred_cpu({N}, {DeviceType::CPU, 0});
  Tensor target_cpu({N}, {DeviceType::CPU, 0});
  pred_cpu.uniform_(-1.0f, 1.0f);
  target_cpu.uniform_(-1.0f, 1.0f);

  Tensor pred_cuda = pred_cpu.to({DeviceType::CUDA, 0});
  Tensor target_cuda = target_cpu.to({DeviceType::CUDA, 0});
  Tensor pred_vk = pred_cpu.to({DeviceType::VULKAN, 0});
  Tensor target_vk = target_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "MSELoss",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA)
                         ? pred_cuda.mse_loss(target_cuda)
                         : pred_vk.mse_loss(target_vk);
        out.impl_->backend().synchronize();
      },
      8, 40, "MUNET_PERF_MAX_RATIO_MSE", 4.0);
}

TEST(PerformanceTest, CrossEntropyCudaVsVulkan) {
  require_gpu_backends();

  constexpr int B = 1024;
  constexpr int C = 128;
  Tensor logits_cpu({B, C}, {DeviceType::CPU, 0});
  logits_cpu.uniform_(-2.0f, 2.0f);

  Tensor targets_cpu({B, C}, {DeviceType::CPU, 0});
  auto *targets_ptr = static_cast<float *>(targets_cpu.data());
  for (int i = 0; i < B * C; ++i)
    targets_ptr[i] = 0.0f;
  for (int b = 0; b < B; ++b)
    targets_ptr[b * C + (b % C)] = 1.0f;

  Tensor logits_cuda = logits_cpu.to({DeviceType::CUDA, 0});
  Tensor targets_cuda = targets_cpu.to({DeviceType::CUDA, 0});
  Tensor logits_vk = logits_cpu.to({DeviceType::VULKAN, 0});
  Tensor targets_vk = targets_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "CrossEntropy",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA)
                         ? logits_cuda.cross_entropy(targets_cuda)
                         : logits_vk.cross_entropy(targets_vk);
        out.impl_->backend().synchronize();
      },
      6, 30, "MUNET_PERF_MAX_RATIO_CROSS_ENTROPY", 4.0);
}
