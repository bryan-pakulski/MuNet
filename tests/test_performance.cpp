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



Tensor make_one_hot_targets(int batch, int classes) {
  Tensor targets_cpu({batch, classes}, {DeviceType::CPU, 0});
  auto *ptr = static_cast<float *>(targets_cpu.data());
  for (int i = 0; i < batch * classes; ++i)
    ptr[i] = 0.0f;
  for (int b = 0; b < batch; ++b)
    ptr[b * classes + (b % classes)] = 1.0f;
  return targets_cpu;
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

  Tensor targets_cpu = make_one_hot_targets(B, C);

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


TEST(PerformanceTest, CrossEntropySmallClassCountCudaVsVulkan) {
  require_gpu_backends();

  constexpr int B = 4096;
  constexpr int C = 16;
  Tensor logits_cpu({B, C}, {DeviceType::CPU, 0});
  logits_cpu.uniform_(-2.0f, 2.0f);
  Tensor targets_cpu = make_one_hot_targets(B, C);

  Tensor logits_cuda = logits_cpu.to({DeviceType::CUDA, 0});
  Tensor targets_cuda = targets_cpu.to({DeviceType::CUDA, 0});
  Tensor logits_vk = logits_cpu.to({DeviceType::VULKAN, 0});
  Tensor targets_vk = targets_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "CrossEntropySmallClassCount",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA)
                         ? logits_cuda.cross_entropy(targets_cuda)
                         : logits_vk.cross_entropy(targets_vk);
        out.impl_->backend().synchronize();
      },
      6, 30, "MUNET_PERF_MAX_RATIO_CROSS_ENTROPY_SMALL_C", 4.5);
}

TEST(PerformanceTest, CrossEntropyLargeClassCountCudaVsVulkan) {
  require_gpu_backends();

  constexpr int B = 256;
  constexpr int C = 1024;
  Tensor logits_cpu({B, C}, {DeviceType::CPU, 0});
  logits_cpu.uniform_(-2.0f, 2.0f);
  Tensor targets_cpu = make_one_hot_targets(B, C);

  Tensor logits_cuda = logits_cpu.to({DeviceType::CUDA, 0});
  Tensor targets_cuda = targets_cpu.to({DeviceType::CUDA, 0});
  Tensor logits_vk = logits_cpu.to({DeviceType::VULKAN, 0});
  Tensor targets_vk = targets_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "CrossEntropyLargeClassCount",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA)
                         ? logits_cuda.cross_entropy(targets_cuda)
                         : logits_vk.cross_entropy(targets_vk);
        out.impl_->backend().synchronize();
      },
      4, 20, "MUNET_PERF_MAX_RATIO_CROSS_ENTROPY_LARGE_C", 5.0);
}

TEST(PerformanceTest, CrossEntropyBackwardCudaVsVulkan) {
  require_gpu_backends();

  constexpr int B = 1024;
  constexpr int C = 128;

  Tensor logits_cpu({B, C}, {DeviceType::CPU, 0});
  logits_cpu.uniform_(-2.0f, 2.0f);
  Tensor targets_cpu = make_one_hot_targets(B, C);

  Tensor logits_cuda = logits_cpu.to({DeviceType::CUDA, 0});
  Tensor targets_cuda = targets_cpu.to({DeviceType::CUDA, 0});
  Tensor logits_vk = logits_cpu.to({DeviceType::VULKAN, 0});
  Tensor targets_vk = targets_cpu.to({DeviceType::VULKAN, 0});
  logits_cuda.set_requires_grad(true);
  logits_vk.set_requires_grad(true);

  run_perf_ratio_test(
      "CrossEntropyBackward",
      [&](Device dev) {
        Tensor loss = (dev.type == DeviceType::CUDA)
                          ? logits_cuda.cross_entropy(targets_cuda)
                          : logits_vk.cross_entropy(targets_vk);
        loss.backward();
        if (dev.type == DeviceType::CUDA)
          logits_cuda.zero_grad();
        else
          logits_vk.zero_grad();
        loss.impl_->backend().synchronize();
      },
      4, 16, "MUNET_PERF_MAX_RATIO_CROSS_ENTROPY_BACKWARD", 5.5);
}

TEST(PerformanceTest, EndToEndTransferAndCrossEntropyCudaVsVulkan) {
  require_gpu_backends();

  constexpr int B = 1024;
  constexpr int C = 128;
  Tensor logits_cpu({B, C}, {DeviceType::CPU, 0});
  logits_cpu.uniform_(-2.0f, 2.0f);
  Tensor targets_cpu = make_one_hot_targets(B, C);

  run_perf_ratio_test(
      "EndToEndTransferAndCrossEntropy",
      [&](Device dev) {
        Tensor logits_dev = logits_cpu.to(dev);
        Tensor targets_dev = targets_cpu.to(dev);
        Tensor loss = logits_dev.cross_entropy(targets_dev);
        Tensor loss_cpu = loss.to({DeviceType::CPU, 0});
        (void)loss_cpu;
        loss.impl_->backend().synchronize();
      },
      2, 10, "MUNET_PERF_MAX_RATIO_E2E_CE", 6.0);
}


TEST(PerformanceTest, ElementwiseSubCudaVsVulkan) {
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
      "ElementwiseSub",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA) ? (a_cuda - b_cuda)
                                                     : (a_vk - b_vk);
        out.impl_->backend().synchronize();
      },
      10, 80, "MUNET_PERF_MAX_RATIO_SUB", 3.0);
}

TEST(PerformanceTest, LogSoftmaxCudaVsVulkan) {
  require_gpu_backends();

  constexpr int B = 512;
  constexpr int C = 512;
  Tensor x_cpu({B, C}, {DeviceType::CPU, 0});
  x_cpu.uniform_(-2.0f, 2.0f);

  Tensor x_cuda = x_cpu.to({DeviceType::CUDA, 0});
  Tensor x_vk = x_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "LogSoftmax",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA) ? x_cuda.log_softmax(-1)
                                                     : x_vk.log_softmax(-1);
        out.impl_->backend().synchronize();
      },
      6, 30, "MUNET_PERF_MAX_RATIO_LOG_SOFTMAX", 4.5);
}

TEST(PerformanceTest, MatmulSmallCudaVsVulkan) {
  require_gpu_backends();

  constexpr int M = 64;
  constexpr int K = 64;
  constexpr int N = 64;

  Tensor a_cpu({M, K}, {DeviceType::CPU, 0});
  Tensor b_cpu({K, N}, {DeviceType::CPU, 0});
  a_cpu.uniform_(-1.0f, 1.0f);
  b_cpu.uniform_(-1.0f, 1.0f);

  Tensor a_cuda = a_cpu.to({DeviceType::CUDA, 0});
  Tensor b_cuda = b_cpu.to({DeviceType::CUDA, 0});
  Tensor a_vk = a_cpu.to({DeviceType::VULKAN, 0});
  Tensor b_vk = b_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "MatmulSmall",
      [&](Device dev) {
        Tensor out =
            (dev.type == DeviceType::CUDA) ? a_cuda.matmul(b_cuda)
                                           : a_vk.matmul(b_vk);
        out.impl_->backend().synchronize();
      },
      8, 40, "MUNET_PERF_MAX_RATIO_MATMUL_SMALL", 4.0);
}

TEST(PerformanceTest, MatmulLargeCudaVsVulkan) {
  require_gpu_backends();

  constexpr int M = 1024;
  constexpr int K = 1024;
  constexpr int N = 1024;

  Tensor a_cpu({M, K}, {DeviceType::CPU, 0});
  Tensor b_cpu({K, N}, {DeviceType::CPU, 0});
  a_cpu.uniform_(-1.0f, 1.0f);
  b_cpu.uniform_(-1.0f, 1.0f);

  Tensor a_cuda = a_cpu.to({DeviceType::CUDA, 0});
  Tensor b_cuda = b_cpu.to({DeviceType::CUDA, 0});
  Tensor a_vk = a_cpu.to({DeviceType::VULKAN, 0});
  Tensor b_vk = b_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "MatmulLarge",
      [&](Device dev) {
        Tensor out =
            (dev.type == DeviceType::CUDA) ? a_cuda.matmul(b_cuda)
                                           : a_vk.matmul(b_vk);
        out.impl_->backend().synchronize();
      },
      2, 8, "MUNET_PERF_MAX_RATIO_MATMUL_LARGE", 4.0);
}

TEST(PerformanceTest, SoftmaxLargeClassCountCudaVsVulkan) {
  require_gpu_backends();

  constexpr int B = 256;
  constexpr int C = 2048;
  Tensor x_cpu({B, C}, {DeviceType::CPU, 0});
  x_cpu.uniform_(-2.0f, 2.0f);

  Tensor x_cuda = x_cpu.to({DeviceType::CUDA, 0});
  Tensor x_vk = x_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "SoftmaxLargeClassCount",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA) ? x_cuda.softmax(-1)
                                                     : x_vk.softmax(-1);
        out.impl_->backend().synchronize();
      },
      4, 20, "MUNET_PERF_MAX_RATIO_SOFTMAX_LARGE_C", 5.0);
}


TEST(PerformanceTest, TinyAddDispatchOverheadCudaVsVulkan) {
  require_gpu_backends();

  constexpr int N = 256;
  Tensor a_cpu({N}, {DeviceType::CPU, 0});
  Tensor b_cpu({N}, {DeviceType::CPU, 0});
  a_cpu.uniform_(0.0f, 1.0f);
  b_cpu.uniform_(0.0f, 1.0f);

  Tensor a_cuda = a_cpu.to({DeviceType::CUDA, 0});
  Tensor b_cuda = b_cpu.to({DeviceType::CUDA, 0});
  Tensor a_vk = a_cpu.to({DeviceType::VULKAN, 0});
  Tensor b_vk = b_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "TinyAddDispatchOverhead",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA) ? (a_cuda + b_cuda)
                                                     : (a_vk + b_vk);
        out.impl_->backend().synchronize();
      },
      20, 300, "MUNET_PERF_MAX_RATIO_TINY_ADD", 6.0);
}

TEST(PerformanceTest, ForwardGraphBuildChainCudaVsVulkan) {
  require_gpu_backends();

  constexpr int N = 4096;
  Tensor a_cpu({N}, {DeviceType::CPU, 0});
  Tensor b_cpu({N}, {DeviceType::CPU, 0});
  a_cpu.uniform_(-1.0f, 1.0f);
  b_cpu.uniform_(-1.0f, 1.0f);

  Tensor a_cuda = a_cpu.to({DeviceType::CUDA, 0});
  Tensor b_cuda = b_cpu.to({DeviceType::CUDA, 0});
  Tensor a_vk = a_cpu.to({DeviceType::VULKAN, 0});
  Tensor b_vk = b_cpu.to({DeviceType::VULKAN, 0});
  a_cuda.set_requires_grad(true);
  b_cuda.set_requires_grad(true);
  a_vk.set_requires_grad(true);
  b_vk.set_requires_grad(true);

  run_perf_ratio_test(
      "ForwardGraphBuildChain",
      [&](Device dev) {
        Tensor x = (dev.type == DeviceType::CUDA) ? a_cuda : a_vk;
        Tensor y = (dev.type == DeviceType::CUDA) ? b_cuda : b_vk;
        Tensor out = ((x + y).relu() * y).sigmoid().sum();
        out.impl_->backend().synchronize();
      },
      8, 80, "MUNET_PERF_MAX_RATIO_FORWARD_GRAPH_CHAIN", 5.0);
}

TEST(PerformanceTest, BackwardStepOverheadCudaVsVulkan) {
  require_gpu_backends();

  constexpr int N = 4096;
  Tensor a_cpu({N}, {DeviceType::CPU, 0});
  Tensor b_cpu({N}, {DeviceType::CPU, 0});
  a_cpu.uniform_(-1.0f, 1.0f);
  b_cpu.uniform_(-1.0f, 1.0f);

  Tensor a_cuda = a_cpu.to({DeviceType::CUDA, 0});
  Tensor b_cuda = b_cpu.to({DeviceType::CUDA, 0});
  Tensor a_vk = a_cpu.to({DeviceType::VULKAN, 0});
  Tensor b_vk = b_cpu.to({DeviceType::VULKAN, 0});
  a_cuda.set_requires_grad(true);
  b_cuda.set_requires_grad(true);
  a_vk.set_requires_grad(true);
  b_vk.set_requires_grad(true);

  run_perf_ratio_test(
      "BackwardStepOverhead",
      [&](Device dev) {
        Tensor x = (dev.type == DeviceType::CUDA) ? a_cuda : a_vk;
        Tensor y = (dev.type == DeviceType::CUDA) ? b_cuda : b_vk;
        Tensor loss = ((x + y).relu() * y).sum();
        loss.backward();
        x.zero_grad();
        y.zero_grad();
        loss.impl_->backend().synchronize();
      },
      4, 40, "MUNET_PERF_MAX_RATIO_BACKWARD_STEP", 6.0);
}

TEST(PerformanceTest, CopyOnlyCpuToGpuCudaVsVulkan) {
  require_gpu_backends();

  constexpr int N = 1 << 22;
  Tensor x_cpu({N}, {DeviceType::CPU, 0});
  x_cpu.uniform_(-1.0f, 1.0f);

  run_perf_ratio_test(
      "CopyOnlyCpuToGpu",
      [&](Device dev) {
        Tensor x_dev = x_cpu.to(dev);
        x_dev.impl_->backend().synchronize();
      },
      4, 25, "MUNET_PERF_MAX_RATIO_COPY_H2D", 4.0);
}

TEST(PerformanceTest, CopyOnlyGpuToCpuCudaVsVulkan) {
  require_gpu_backends();

  constexpr int N = 1 << 22;
  Tensor x_cpu({N}, {DeviceType::CPU, 0});
  x_cpu.uniform_(-1.0f, 1.0f);
  Tensor x_cuda = x_cpu.to({DeviceType::CUDA, 0});
  Tensor x_vk = x_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "CopyOnlyGpuToCpu",
      [&](Device dev) {
        Tensor x_dev = (dev.type == DeviceType::CUDA) ? x_cuda : x_vk;
        Tensor x_back = x_dev.to({DeviceType::CPU, 0});
        (void)x_back;
      },
      4, 25, "MUNET_PERF_MAX_RATIO_COPY_D2H", 4.0);
}
