#include "inference.hpp"
#include "core/backend.hpp"
#include "core/kernel_fusion_planner.hpp"
#include "nn.hpp"
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

bool require_gpu_backends(std::string *reason = nullptr) {
#ifdef MUNET_USE_CUDA
  const bool has_cuda = has_device(DeviceType::CUDA);
  constexpr bool cuda_compiled = true;
#else
  const bool has_cuda = false;
  constexpr bool cuda_compiled = false;
#endif

#ifdef MUNET_USE_VULKAN
  const bool has_vulkan = has_device(DeviceType::VULKAN);
  constexpr bool vulkan_compiled = true;
#else
  const bool has_vulkan = false;
  constexpr bool vulkan_compiled = false;
#endif

  if (!perf_tests_enabled()) {
    if (reason) {
      *reason = "Set MUNET_RUN_PERF_TESTS=1 to run performance tests.";
    }
    return false;
  }
  if (!has_cuda || !has_vulkan) {
    if (reason) {
      std::string backend_details;
      for (const auto &status : BackendManager::backend_status()) {
        if (status.source != "builtin") {
          continue;
        }
        if (status.name == "cuda" || status.name == "vulkan") {
          backend_details += status.name + "{reason=" + status.reason_code +
                             ", detail=" + status.detail + "} ";
        }
      }
      *reason =
          "Performance comparison requires both CUDA and Vulkan devices. "
          "compiled(cuda=" +
          std::string(cuda_compiled ? "yes" : "no") + ", vulkan=" +
          std::string(vulkan_compiled ? "yes" : "no") +
          ") runtime(cuda=" + (has_cuda ? std::string("yes") : "no") +
          ", vulkan=" + (has_vulkan ? std::string("yes") : "no") + "). " +
          backend_details;
    }
    return false;
  }
  return true;
}

double run_chain_baseline_ms(const Tensor &x, const Tensor &y, int warmup,
                             int iters) {
  return benchmark_ms(
      [&]() {
        Tensor out = ((x + y).relu() * y).sigmoid();
        out.impl_->backend().synchronize();
      },
      warmup, iters);
}

double run_chain_workspace_ms(const Tensor &x, const Tensor &y, int warmup,
                              int iters) {
  const Device dev = x.device();
  Tensor ping(x.shape(), dev, x.dtype());
  Tensor pong(x.shape(), dev, x.dtype());
  auto &backend = x.impl_->backend();
  auto *alloc = backend.allocation_transfer_capability();
  auto *elt = backend.elementwise_capability();
  if (!alloc || !elt) {
    return 0.0;
  }

  const size_t n = x.size();
  const BroadcastInfo info =
      compute_broadcast(x.shape(), x.strides(), y.shape(), y.strides());
  return benchmark_ms(
      [&]() {
        elt->add(*x.impl_->storage, *y.impl_->storage, *ping.impl_->storage,
                 info);
        elt->relu(*ping.impl_->storage, *pong.impl_->storage, n);
        elt->mul(*pong.impl_->storage, *y.impl_->storage, *ping.impl_->storage,
                 info);
        elt->sigmoid(*ping.impl_->storage, *pong.impl_->storage, n);
        alloc->synchronize();
      },
      warmup, iters);
}

void run_fusion_chain_speedup_test(Device dev, const Tensor &x, const Tensor &y,
                                   const char *min_speedup_env,
                                   double default_speedup) {
  std::vector<ForwardNode> nodes(4);
  nodes[0].op_name = "Add";
  nodes[1].op_name = "Relu";
  nodes[2].op_name = "Mul";
  nodes[3].op_name = "Sigmoid";
  const auto groups = fusion::plan_elementwise_fusion_groups(nodes);
  ASSERT_EQ(groups.size(), 1u);
  ASSERT_TRUE(groups[0].fusible);

  const double baseline_ms = run_chain_baseline_ms(x, y, 8, 50);
  const double workspace_ms = run_chain_workspace_ms(x, y, 8, 50);
  ASSERT_GT(workspace_ms, 0.0);
  const double speedup = baseline_ms / workspace_ms;
  const double min_speedup = get_env_ratio(min_speedup_env, default_speedup);

  std::cout << "[PERF] ElementwiseChainFusion backend="
            << (dev.type == DeviceType::CUDA ? "cuda" : "vulkan")
            << " baseline_ms=" << baseline_ms
            << " workspace_ms=" << workspace_ms << " speedup=" << speedup
            << std::endl;
  EXPECT_GE(speedup, min_speedup)
      << "Expected measurable speedup for chained elementwise plan on backend.";
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
                         const std::function<void(Device)> &runner, int warmup,
                         int iters, const char *ratio_env,
                         double default_ratio) {
  Device cuda{DeviceType::CUDA, 0};
  Device vk{DeviceType::VULKAN, 0};

  const double cuda_ms = benchmark_ms([&]() { runner(cuda); }, warmup, iters);
  const double vk_ms = benchmark_ms([&]() { runner(vk); }, warmup, iters);
  const double ratio = vk_ms / std::max(cuda_ms, 1e-9);

  std::cout << "[PERF] " << name << " cuda_ms=" << cuda_ms << " vk_ms=" << vk_ms
            << " ratio=" << ratio << std::endl;

  const double max_ratio = get_env_ratio(ratio_env, default_ratio);
  EXPECT_LE(ratio, max_ratio)
      << "Vulkan is too slow relative to CUDA for test " << name
      << ". ratio=" << ratio << " max=" << max_ratio;
}

} // namespace

TEST(PerformanceTest, ElementwiseAddCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
        Tensor out =
            (dev.type == DeviceType::CUDA) ? (a_cuda + b_cuda) : (a_vk + b_vk);
        out.impl_->backend().synchronize();
      },
      10, 80, "MUNET_PERF_MAX_RATIO_ADD", 3.0);
}

TEST(PerformanceTest, ElementwiseMulCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
        Tensor out =
            (dev.type == DeviceType::CUDA) ? (a_cuda * b_cuda) : (a_vk * b_vk);
        out.impl_->backend().synchronize();
      },
      10, 80, "MUNET_PERF_MAX_RATIO_MUL", 3.0);
}

TEST(PerformanceTest, ElementwiseChainFusionSpeedupCudaAndVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

  constexpr int N = 1 << 20;
  Tensor x_cpu({N}, {DeviceType::CPU, 0});
  Tensor y_cpu({N}, {DeviceType::CPU, 0});
  x_cpu.uniform_(-1.0f, 1.0f);
  y_cpu.uniform_(-1.0f, 1.0f);

  Tensor x_cuda = x_cpu.to({DeviceType::CUDA, 0});
  Tensor y_cuda = y_cpu.to({DeviceType::CUDA, 0});
  Tensor x_vk = x_cpu.to({DeviceType::VULKAN, 0});
  Tensor y_vk = y_cpu.to({DeviceType::VULKAN, 0});

  run_fusion_chain_speedup_test(Device{DeviceType::CUDA, 0}, x_cuda, y_cuda,
                                "MUNET_PERF_MIN_FUSION_SPEEDUP_CUDA", 1.03);
  run_fusion_chain_speedup_test(Device{DeviceType::VULKAN, 0}, x_vk, y_vk,
                                "MUNET_PERF_MIN_FUSION_SPEEDUP_VK", 1.03);
}

TEST(PerformanceTest, MatmulCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
        Tensor out = (dev.type == DeviceType::CUDA) ? a_cuda.matmul(b_cuda)
                                                    : a_vk.matmul(b_vk);
        out.impl_->backend().synchronize();
      },
      4, 20, "MUNET_PERF_MAX_RATIO_MATMUL", 4.0);
}

TEST(PerformanceTest, ReluCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
        Tensor out =
            (dev.type == DeviceType::CUDA) ? (a_cuda + b_cuda) : (a_vk + b_vk);
        out.impl_->backend().synchronize();
      },
      8, 40, "MUNET_PERF_MAX_RATIO_BROADCAST_ADD", 4.0);
}

TEST(PerformanceTest, SigmoidCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
        Tensor out =
            (dev.type == DeviceType::CUDA) ? (a_cuda - b_cuda) : (a_vk - b_vk);
        out.impl_->backend().synchronize();
      },
      10, 80, "MUNET_PERF_MAX_RATIO_SUB", 3.0);
}

TEST(PerformanceTest, LogSoftmaxCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
        Tensor out = (dev.type == DeviceType::CUDA) ? a_cuda.matmul(b_cuda)
                                                    : a_vk.matmul(b_vk);
        out.impl_->backend().synchronize();
      },
      8, 40, "MUNET_PERF_MAX_RATIO_MATMUL_SMALL", 4.0);
}

TEST(PerformanceTest, MatmulLargeCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
        Tensor out = (dev.type == DeviceType::CUDA) ? a_cuda.matmul(b_cuda)
                                                    : a_vk.matmul(b_vk);
        out.impl_->backend().synchronize();
      },
      2, 8, "MUNET_PERF_MAX_RATIO_MATMUL_LARGE", 4.0);
}

TEST(PerformanceTest, SoftmaxLargeClassCountCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
        Tensor out =
            (dev.type == DeviceType::CUDA) ? (a_cuda + b_cuda) : (a_vk + b_vk);
        out.impl_->backend().synchronize();
      },
      20, 300, "MUNET_PERF_MAX_RATIO_TINY_ADD", 6.0);
}

TEST(PerformanceTest, ForwardGraphBuildChainCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

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

TEST(PerformanceTest, Conv2DForwardCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

  constexpr int B = 16, IC = 32, OC = 64, H = 32, W = 32, K = 3;
  Tensor in_cpu({B, IC, H, W}, {DeviceType::CPU, 0});
  Tensor w_cpu({OC, IC, K, K}, {DeviceType::CPU, 0});
  in_cpu.uniform_(-1.0f, 1.0f);
  w_cpu.uniform_(-1.0f, 1.0f);

  Tensor in_cuda = in_cpu.to({DeviceType::CUDA, 0});
  Tensor w_cuda = w_cpu.to({DeviceType::CUDA, 0});
  Tensor in_vk = in_cpu.to({DeviceType::VULKAN, 0});
  Tensor w_vk = w_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "Conv2DForward",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA)
                         ? in_cuda.conv2d(w_cuda, Tensor(), 1, 1)
                         : in_vk.conv2d(w_vk, Tensor(), 1, 1);
        out.impl_->backend().synchronize();
      },
      3, 12, "MUNET_PERF_MAX_RATIO_CONV2D", 5.0);
}

TEST(PerformanceTest, MaxPool2DCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

  constexpr int B = 32, C = 64, H = 56, W = 56;
  Tensor in_cpu({B, C, H, W}, {DeviceType::CPU, 0});
  in_cpu.uniform_(-1.0f, 1.0f);

  Tensor in_cuda = in_cpu.to({DeviceType::CUDA, 0});
  Tensor in_vk = in_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "MaxPool2D",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA) ? in_cuda.max_pool2d(2, 2)
                                                    : in_vk.max_pool2d(2, 2);
        out.impl_->backend().synchronize();
      },
      4, 16, "MUNET_PERF_MAX_RATIO_MAXPOOL2D", 5.0);
}

TEST(PerformanceTest, Upsample2DCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

  constexpr int B = 16, C = 64, H = 64, W = 64;
  Tensor in_cpu({B, C, H, W}, {DeviceType::CPU, 0});
  in_cpu.uniform_(-1.0f, 1.0f);

  Tensor in_cuda = in_cpu.to({DeviceType::CUDA, 0});
  Tensor in_vk = in_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "Upsample2D",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA) ? in_cuda.upsample2d(2)
                                                    : in_vk.upsample2d(2);
        out.impl_->backend().synchronize();
      },
      3, 12, "MUNET_PERF_MAX_RATIO_UPSAMPLE2D", 5.0);
}

TEST(PerformanceTest, ConcatCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

  constexpr int B = 64, C1 = 32, C2 = 32, H = 28, W = 28;
  Tensor a_cpu({B, C1, H, W}, {DeviceType::CPU, 0});
  Tensor b_cpu({B, C2, H, W}, {DeviceType::CPU, 0});
  a_cpu.uniform_(-1.0f, 1.0f);
  b_cpu.uniform_(-1.0f, 1.0f);

  Tensor a_cuda = a_cpu.to({DeviceType::CUDA, 0});
  Tensor b_cuda = b_cpu.to({DeviceType::CUDA, 0});
  Tensor a_vk = a_cpu.to({DeviceType::VULKAN, 0});
  Tensor b_vk = b_cpu.to({DeviceType::VULKAN, 0});

  run_perf_ratio_test(
      "Concat",
      [&](Device dev) {
        Tensor out = (dev.type == DeviceType::CUDA)
                         ? Tensor::cat({a_cuda, b_cuda}, 1)
                         : Tensor::cat({a_vk, b_vk}, 1);
        out.impl_->backend().synchronize();
      },
      4, 16, "MUNET_PERF_MAX_RATIO_CONCAT", 5.0);
}

TEST(PerformanceTest, OptimizerStepCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

  constexpr int N = 1 << 20;
  Tensor p_cpu({N}, {DeviceType::CPU, 0});
  Tensor x_cpu({N}, {DeviceType::CPU, 0});
  p_cpu.uniform_(-1.0f, 1.0f);
  x_cpu.uniform_(-1.0f, 1.0f);

  Tensor p_cuda = p_cpu.to({DeviceType::CUDA, 0});
  Tensor x_cuda = x_cpu.to({DeviceType::CUDA, 0});
  Tensor p_vk = p_cpu.to({DeviceType::VULKAN, 0});
  Tensor x_vk = x_cpu.to({DeviceType::VULKAN, 0});
  p_cuda.set_requires_grad(true);
  p_vk.set_requires_grad(true);

  run_perf_ratio_test(
      "OptimizerStep",
      [&](Device dev) {
        Tensor &p = (dev.type == DeviceType::CUDA) ? p_cuda : p_vk;
        Tensor &x = (dev.type == DeviceType::CUDA) ? x_cuda : x_vk;
        Tensor loss = (p * x).sum();
        loss.backward();
        p.step(1e-3f);
        p.zero_grad();
        loss.impl_->backend().synchronize();
      },
      3, 14, "MUNET_PERF_MAX_RATIO_OPTIMIZER_STEP", 5.0);
}

TEST(PerformanceTest, RepresentativeInferenceEngineMemoryProfile) {
  if (!perf_tests_enabled()) {
    GTEST_SKIP() << "Set MUNET_RUN_PERF_TESTS=1 to run performance tests.";
  }

  Profiler::get().reset();

  TensorOptions options;
  options.device = Device{DeviceType::CPU, 0};
  options.dtype = DataType::Float32;

  auto model = std::make_shared<nn::Sequential>();
  model->add(std::make_shared<nn::Linear>(128, 256, true, options));
  model->add(std::make_shared<nn::ReLU>());
  model->add(std::make_shared<nn::Linear>(256, 64, true, options));
  model->add(std::make_shared<nn::ReLU>());
  model->add(std::make_shared<nn::Linear>(64, 16, true, options));

  inference::Engine engine;
  engine.load(model);

  Tensor input({32, 128}, options.device, options.dtype, false);
  input.uniform_(-1.0f, 1.0f);

  engine.compile(input, {-1, 128}, {-1, 16});
  const size_t baseline_current = Profiler::get().current_memory_bytes();

  double avg_ms = benchmark_ms(
      [&]() {
        Tensor output = engine.run(input);
        output = output.to(Device{DeviceType::CPU, 0});
      },
      2, 12);

  const size_t final_current = Profiler::get().current_memory_bytes();
  const size_t peak_memory = Profiler::get().peak_memory_bytes();

  EXPECT_GT(avg_ms, 0.0);
  EXPECT_GE(peak_memory, baseline_current);
  EXPECT_LE(final_current, peak_memory);
  EXPECT_LE(final_current,
            baseline_current + input.size() * dtype_size(input.dtype()) * 8);
  EXPECT_EQ(engine.stats().compiled_output_shape, (std::vector<int>{32, 16}));
  EXPECT_GE(engine.stats().peak_memory_bytes, peak_memory);
}
