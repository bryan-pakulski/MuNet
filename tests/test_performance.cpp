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

} // namespace

TEST(PerformanceTest, ElementwiseAddCudaVsVulkan) {
  require_gpu_backends();

  Device cuda{DeviceType::CUDA, 0};
  Device vk{DeviceType::VULKAN, 0};

  constexpr int N = 1 << 20;
  Tensor a_cpu({N}, {DeviceType::CPU, 0});
  Tensor b_cpu({N}, {DeviceType::CPU, 0});
  a_cpu.uniform_(0.0f, 1.0f);
  b_cpu.uniform_(0.0f, 1.0f);

  Tensor a_cuda = a_cpu.to(cuda);
  Tensor b_cuda = b_cpu.to(cuda);
  Tensor a_vk = a_cpu.to(vk);
  Tensor b_vk = b_cpu.to(vk);

  auto run_cuda = [&]() {
    Tensor out = a_cuda + b_cuda;
    out.impl_->backend().synchronize();
  };

  auto run_vk = [&]() {
    Tensor out = a_vk + b_vk;
    out.impl_->backend().synchronize();
  };

  double cuda_ms = benchmark_ms(run_cuda, 10, 80);
  double vk_ms = benchmark_ms(run_vk, 10, 80);
  double ratio = vk_ms / std::max(cuda_ms, 1e-9);

  std::cout << "[PERF] ElementwiseAdd cuda_ms=" << cuda_ms
            << " vk_ms=" << vk_ms << " ratio=" << ratio << std::endl;

  const double max_ratio = get_env_ratio("MUNET_PERF_MAX_RATIO_ADD", 3.0);
  EXPECT_LE(ratio, max_ratio)
      << "Vulkan add is too slow relative to CUDA."
      << " ratio=" << ratio << " max=" << max_ratio;
}

TEST(PerformanceTest, MatmulCudaVsVulkan) {
  require_gpu_backends();

  Device cuda{DeviceType::CUDA, 0};
  Device vk{DeviceType::VULKAN, 0};

  constexpr int M = 256;
  constexpr int K = 256;
  constexpr int N = 256;

  Tensor a_cpu({M, K}, {DeviceType::CPU, 0});
  Tensor b_cpu({K, N}, {DeviceType::CPU, 0});
  a_cpu.uniform_(-1.0f, 1.0f);
  b_cpu.uniform_(-1.0f, 1.0f);

  Tensor a_cuda = a_cpu.to(cuda);
  Tensor b_cuda = b_cpu.to(cuda);
  Tensor a_vk = a_cpu.to(vk);
  Tensor b_vk = b_cpu.to(vk);

  auto run_cuda = [&]() {
    Tensor out = a_cuda.matmul(b_cuda);
    out.impl_->backend().synchronize();
  };

  auto run_vk = [&]() {
    Tensor out = a_vk.matmul(b_vk);
    out.impl_->backend().synchronize();
  };

  double cuda_ms = benchmark_ms(run_cuda, 4, 20);
  double vk_ms = benchmark_ms(run_vk, 4, 20);
  double ratio = vk_ms / std::max(cuda_ms, 1e-9);

  std::cout << "[PERF] Matmul cuda_ms=" << cuda_ms << " vk_ms=" << vk_ms
            << " ratio=" << ratio << std::endl;

  const double max_ratio = get_env_ratio("MUNET_PERF_MAX_RATIO_MATMUL", 4.0);
  EXPECT_LE(ratio, max_ratio)
      << "Vulkan matmul is too slow relative to CUDA."
      << " ratio=" << ratio << " max=" << max_ratio;
}

TEST(PerformanceTest, ReluCudaVsVulkan) {
  require_gpu_backends();

  Device cuda{DeviceType::CUDA, 0};
  Device vk{DeviceType::VULKAN, 0};

  constexpr int N = 1 << 22;
  Tensor x_cpu({N}, {DeviceType::CPU, 0});
  x_cpu.uniform_(-2.0f, 2.0f);

  Tensor x_cuda = x_cpu.to(cuda);
  Tensor x_vk = x_cpu.to(vk);

  auto run_cuda = [&]() {
    Tensor out = x_cuda.relu();
    out.impl_->backend().synchronize();
  };

  auto run_vk = [&]() {
    Tensor out = x_vk.relu();
    out.impl_->backend().synchronize();
  };

  double cuda_ms = benchmark_ms(run_cuda, 10, 80);
  double vk_ms = benchmark_ms(run_vk, 10, 80);
  double ratio = vk_ms / std::max(cuda_ms, 1e-9);

  std::cout << "[PERF] ReLU cuda_ms=" << cuda_ms << " vk_ms=" << vk_ms
            << " ratio=" << ratio << std::endl;

  const double max_ratio = get_env_ratio("MUNET_PERF_MAX_RATIO_RELU", 3.0);
  EXPECT_LE(ratio, max_ratio)
      << "Vulkan ReLU is too slow relative to CUDA."
      << " ratio=" << ratio << " max=" << max_ratio;
}
