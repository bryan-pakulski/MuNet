#include "inference.hpp"
#include "nn.hpp"
#include "tensor.hpp"
#include "test_utils.hpp"
#include "core/op_dispatch.hpp"
#include "core/util/logging.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <optional>
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
#else
  const bool has_cuda = false;
#endif

#ifdef MUNET_USE_VULKAN
  const bool has_vulkan = has_device(DeviceType::VULKAN);
#else
  const bool has_vulkan = false;
#endif

  if (!perf_tests_enabled()) {
    if (reason) {
      *reason = "Set MUNET_RUN_PERF_TESTS=1 to run performance tests.";
    }
    return false;
  }
  if (!has_cuda || !has_vulkan) {
    if (reason) {
      *reason = "Performance comparison requires both CUDA and Vulkan devices "
                "in this environment.";
    }
    return false;
  }
  return true;
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

struct BackendPerfBreakdown {
  double e2e_ms = 0.0;
  double total_cpu_us = 0.0;
  double total_gpu_us = 0.0;
  double dispatch_us = 0.0;
  double kernel_us = 0.0;
  uint64_t fallback_count = 0;
  size_t profiled_entries = 0;
};

BackendPerfBreakdown run_instance_segmentation_workload(Device dev, int warmup,
                                                        int iters) {
  constexpr int B = 2;
  constexpr int H = 128;
  constexpr int W = 128;
  constexpr int NUM_CLASSES = 6;

  Tensor input_cpu({B, 3, H, W}, {DeviceType::CPU, 0});
  input_cpu.uniform_(-1.0f, 1.0f);

  Tensor class_target_cpu({B, NUM_CLASSES, H / 2, W / 2}, {DeviceType::CPU, 0});
  class_target_cpu.uniform_(0.0f, 1.0f);
  Tensor mask_target_cpu({B, 1, H, W}, {DeviceType::CPU, 0});
  mask_target_cpu.uniform_(0.0f, 1.0f);

  auto weight_like = [&](const Shape &shape) {
    Tensor t(shape, {DeviceType::CPU, 0}, DataType::Float32, true);
    t.uniform_(-0.05f, 0.05f);
    return t.to(dev);
  };
  auto bias_like = [&](int c) {
    Tensor t({c}, {DeviceType::CPU, 0}, DataType::Float32, true);
    t.uniform_(-0.02f, 0.02f);
    return t.to(dev);
  };

  Tensor input = input_cpu.to(dev);
  Tensor class_target = class_target_cpu.to(dev);
  Tensor mask_target = mask_target_cpu.to(dev);

  Tensor w_stem = weight_like({8, 3, 3, 3});
  Tensor b_stem = bias_like(8);
  Tensor w_body = weight_like({16, 8, 3, 3});
  Tensor b_body = bias_like(16);
  Tensor w_cls = weight_like({NUM_CLASSES, 16, 1, 1});
  Tensor b_cls = bias_like(NUM_CLASSES);
  Tensor w_seed = weight_like({1, 16, 3, 3});
  Tensor b_seed = bias_like(1);
  Tensor w_mask = weight_like({1, 24, 3, 3});
  Tensor b_mask = bias_like(1);

  auto run_once = [&]() {
    Tensor stem = input.conv2d(w_stem, b_stem, 1, 1).relu();
    Tensor pooled = stem.max_pool2d(2, 2, 0);
    Tensor body = pooled.conv2d(w_body, b_body, 1, 1).relu();

    Tensor class_logits = body.conv2d(w_cls, b_cls, 1, 0);
    Tensor class_probs = class_logits.softmax(1);

    Tensor instance_seed = body.conv2d(w_seed, b_seed, 1, 1).sigmoid();
    Tensor upsampled = body.upsample2d(2);
    Tensor fused = Tensor::cat({upsampled, stem}, 1);
    Tensor mask_logits = fused.conv2d(w_mask, b_mask, 1, 1);
    Tensor mask_probs = mask_logits.sigmoid();
    Tensor edge_hint = mask_probs.max_pool2d(3, 1, 1);

    Tensor class_loss = class_probs.mse_loss(class_target);
    Tensor mask_loss = mask_probs.mse_loss(mask_target);
    Tensor edge_loss = edge_hint.mse_loss(mask_target);
    Tensor seed_loss = instance_seed.mean();
    Tensor total = class_loss + mask_loss + edge_loss + seed_loss;

    total.backward();
    w_stem.step(1e-3f);
    b_stem.step(1e-3f);
    w_body.step(1e-3f);
    b_body.step(1e-3f);
    w_cls.step(1e-3f);
    b_cls.step(1e-3f);
    w_seed.step(1e-3f);
    b_seed.step(1e-3f);
    w_mask.step(1e-3f);
    b_mask.step(1e-3f);

    w_stem.zero_grad();
    b_stem.zero_grad();
    w_body.zero_grad();
    b_body.zero_grad();
    w_cls.zero_grad();
    b_cls.zero_grad();
    w_seed.zero_grad();
    b_seed.zero_grad();
    w_mask.zero_grad();
    b_mask.zero_grad();

    total.impl_->backend().synchronize();
    Tensor total_cpu = total.to({DeviceType::CPU, 0});
    (void)total_cpu;
  };

  for (int i = 0; i < warmup; ++i) {
    run_once();
  }

  Profiler::get().reset();
  ops::reset_fallback_telemetry();
  const double e2e_ms = benchmark_ms(run_once, 0, iters);

  const auto snapshot = Profiler::get().snapshot();
  const auto fallback = ops::fallback_telemetry_snapshot();

  BackendPerfBreakdown out;
  out.e2e_ms = e2e_ms;
  out.fallback_count = fallback.accelerator_cpu_fallback_total;
  out.profiled_entries = snapshot.stats.size();

  for (const auto &entry : snapshot.stats) {
    const auto &name = entry.first;
    const auto &stat = entry.second;
    out.total_cpu_us += stat.cpu_us;
    out.total_gpu_us += stat.gpu_us;
    if (name.rfind("dispatch.", 0) == 0) {
      out.dispatch_us += stat.cpu_us + stat.gpu_us;
    }
    if (name.find(".cuda") != std::string::npos ||
        name.find(".vulkan") != std::string::npos) {
      out.kernel_us += stat.cpu_us + stat.gpu_us;
    }
  }

  return out;
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

TEST(PerformanceTest, InstanceSegmentationE2EMetricsCudaVsVulkan) {
  std::string reason;
  if (!require_gpu_backends(&reason)) {
    GTEST_SKIP() << reason;
  }

  set_profile_enabled_override(true);
  Device cuda{DeviceType::CUDA, 0};
  Device vk{DeviceType::VULKAN, 0};

  const BackendPerfBreakdown cuda_metrics =
      run_instance_segmentation_workload(cuda, 2, 8);
  const BackendPerfBreakdown vk_metrics = run_instance_segmentation_workload(vk, 2, 8);

  const double e2e_ratio = vk_metrics.e2e_ms / std::max(cuda_metrics.e2e_ms, 1e-9);
  const double dispatch_ratio =
      vk_metrics.dispatch_us / std::max(cuda_metrics.dispatch_us, 1e-9);
  const double kernel_ratio =
      vk_metrics.kernel_us / std::max(cuda_metrics.kernel_us, 1e-9);
  const double cpu_total_ratio =
      vk_metrics.total_cpu_us / std::max(cuda_metrics.total_cpu_us, 1e-9);

  std::cout << "[PERF][InstanceSeg] cuda_e2e_ms=" << cuda_metrics.e2e_ms
            << " vk_e2e_ms=" << vk_metrics.e2e_ms << " e2e_ratio=" << e2e_ratio
            << " cuda_dispatch_us=" << cuda_metrics.dispatch_us
            << " vk_dispatch_us=" << vk_metrics.dispatch_us
            << " dispatch_ratio=" << dispatch_ratio
            << " cuda_kernel_us=" << cuda_metrics.kernel_us
            << " vk_kernel_us=" << vk_metrics.kernel_us
            << " kernel_ratio=" << kernel_ratio
            << " cuda_total_cpu_us=" << cuda_metrics.total_cpu_us
            << " vk_total_cpu_us=" << vk_metrics.total_cpu_us
            << " cpu_total_ratio=" << cpu_total_ratio
            << " cuda_fallbacks=" << cuda_metrics.fallback_count
            << " vk_fallbacks=" << vk_metrics.fallback_count
            << " cuda_profiled_entries=" << cuda_metrics.profiled_entries
            << " vk_profiled_entries=" << vk_metrics.profiled_entries
            << std::endl;

  constexpr double kMaxE2ERatio = 1.10;
  constexpr double kMaxDispatchRatio = 1.30;
  constexpr double kMaxKernelRatio = 1.20;
  constexpr double kMaxCpuTotalRatio = 1.30;

  EXPECT_GT(cuda_metrics.profiled_entries, 0u);
  EXPECT_GT(vk_metrics.profiled_entries, 0u);
  EXPECT_GT(cuda_metrics.dispatch_us, 0.0);
  EXPECT_GT(vk_metrics.dispatch_us, 0.0);
  EXPECT_GT(cuda_metrics.kernel_us, 0.0);
  EXPECT_GT(vk_metrics.kernel_us, 0.0);
  EXPECT_EQ(cuda_metrics.fallback_count, 0u);
  EXPECT_EQ(vk_metrics.fallback_count, 0u);
  EXPECT_LE(e2e_ratio, kMaxE2ERatio)
      << "Vulkan e2e latency regressed for instance-segmentation style "
         "workload. ratio="
      << e2e_ratio << " max=" << kMaxE2ERatio;
  EXPECT_LE(dispatch_ratio, kMaxDispatchRatio);
  EXPECT_LE(kernel_ratio, kMaxKernelRatio);
  EXPECT_LE(cpu_total_ratio, kMaxCpuTotalRatio);

  set_profile_enabled_override(std::nullopt);
}
