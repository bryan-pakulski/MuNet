#include "backend.hpp"
#include "backend/cpu_backend.hpp"
#include "core/util/profiler.hpp"
#include "tensor.hpp"
#include <gtest/gtest.h>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

using namespace munet;

namespace {

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    const char *existing = std::getenv(name_);
    if (existing) {
      had_previous_ = true;
      previous_ = existing;
    }

    if (value) {
      setenv(name_, value, 1);
    } else {
      unsetenv(name_);
    }
  }

  ~ScopedEnvVar() {
    if (had_previous_) {
      setenv(name_, previous_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

private:
  const char *name_;
  bool had_previous_ = false;
  std::string previous_;
};

class PartialMatmulBackend : public Backend,
                             public BackendAllocationTransferCapability,
                             public BackendBlasCapability {
public:
  explicit PartialMatmulBackend(int device_index)
      : device_index_(device_index) {}

  const char *name() const override { return "partial_matmul"; }

  BackendAllocationTransferCapability *allocation_transfer_capability() override {
    return this;
  }
  const BackendAllocationTransferCapability *allocation_transfer_capability() const override {
    return this;
  }

  BackendBlasCapability *blas_capability() override { return this; }
  const BackendBlasCapability *blas_capability() const override {
    return this;
  }

  DataType preferred_accumulation_dtype(BackendFeature feature,
                                        DataType dtype) const override {
    if (feature == BackendFeature::Matmul && dtype == DataType::Float16) {
      return DataType::Float32;
    }
    return Backend::preferred_accumulation_dtype(feature, dtype);
  }

  BackendFallbackPolicy preferred_fallback_policy(BackendFeature feature,
                                                  DataType dtype) const override {
    if (feature == BackendFeature::ElementwiseBinary) {
      return BackendFallbackPolicy::CPUFallback;
    }
    return Backend::preferred_fallback_policy(feature, dtype);
  }

  double get_last_kernel_time_us() override { return 0.0; }

  void *allocate(size_t bytes) override { return std::malloc(bytes); }
  void deallocate(void *ptr) override { std::free(ptr); }
  void memset(void *ptr, int value, size_t bytes) override {
    std::memset(ptr, value, bytes);
  }
  void copy(const void *src, void *dst, size_t bytes, Device, Device) override {
    std::memcpy(dst, src, bytes);
  }
  void synchronize() override {}
  void all_reduce(Storage &, size_t) override {}

  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) override {
    ++matmul_calls_;
    const float *ap = static_cast<const float *>(a.data());
    const float *bp = static_cast<const float *>(b.data());
    float *op = static_cast<float *>(out.data());

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          const int a_index = transA ? (k * M + m) : (m * K + k);
          const int b_index = transB ? (n * K + k) : (k * N + n);
          sum += ap[a_index] * bp[b_index];
        }
        op[m * N + n] = sum;
      }
    }
  }

  int device_index() const { return device_index_; }
  int matmul_calls() const { return matmul_calls_; }

private:
  int device_index_ = 0;
  int matmul_calls_ = 0;
};

class TimedAddBackend : public Backend,
                        public BackendAllocationTransferCapability,
                        public BackendElementwiseCapability {
public:
  TimedAddBackend(int device_index, bool gpu_timing, double kernel_time_us)
      : device_index_(device_index), gpu_timing_(gpu_timing),
        kernel_time_us_(kernel_time_us) {}

  const char *name() const override {
    return gpu_timing_ ? "timed_gpu_add" : "timed_cpu_add";
  }

  BackendAllocationTransferCapability *allocation_transfer_capability() override {
    return this;
  }
  const BackendAllocationTransferCapability *allocation_transfer_capability() const override {
    return this;
  }

  BackendElementwiseCapability *elementwise_capability() override { return this; }
  const BackendElementwiseCapability *elementwise_capability() const override {
    return this;
  }

  void *allocate(size_t bytes) override { return std::malloc(bytes); }
  void deallocate(void *ptr) override { std::free(ptr); }
  void memset(void *ptr, int value, size_t bytes) override {
    std::memset(ptr, value, bytes);
  }
  void copy(const void *src, void *dst, size_t bytes, Device, Device) override {
    std::memcpy(dst, src, bytes);
  }
  void synchronize() override {
    ++sync_calls_;
    last_kernel_time_us_ = gpu_timing_ ? kernel_time_us_ : 0.0;
  }
  void all_reduce(Storage &, size_t) override {}
  double get_last_kernel_time_us() override { return last_kernel_time_us_; }
  bool reports_gpu_kernel_time() const override { return gpu_timing_; }

  void add(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &) override {
    ++add_calls_;
    const float *ap = static_cast<const float *>(a.data());
    const float *bp = static_cast<const float *>(b.data());
    float *op = static_cast<float *>(out.data());
    op[0] = ap[0] + bp[0];
  }

  void sub(const Storage &, const Storage &, Storage &,
           const BroadcastInfo &) override {
    throw std::runtime_error("unused");
  }
  void mul(const Storage &, const Storage &, Storage &,
           const BroadcastInfo &) override {
    throw std::runtime_error("unused");
  }
  void div(const Storage &, const Storage &, Storage &,
           const BroadcastInfo &) override {
    throw std::runtime_error("unused");
  }
  void broadcast_row(const Storage &, Storage &, int, int) override {
    throw std::runtime_error("unused");
  }
  void relu(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void relu_backward(const Storage &, const Storage &, Storage &,
                     size_t) override {
    throw std::runtime_error("unused");
  }
  void sigmoid(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void sigmoid_backward(const Storage &, const Storage &, Storage &,
                        size_t) override {
    throw std::runtime_error("unused");
  }
  void softmax(const Storage &, Storage &, int, int) override {
    throw std::runtime_error("unused");
  }
  void softmax_backward(const Storage &, const Storage &, Storage &, int,
                        int) override {
    throw std::runtime_error("unused");
  }

  int device_index() const { return device_index_; }
  int add_calls() const { return add_calls_; }
  int sync_calls() const { return sync_calls_; }

private:
  int device_index_ = 0;
  bool gpu_timing_ = false;
  double kernel_time_us_ = 0.0;
  double last_kernel_time_us_ = 0.0;
  int add_calls_ = 0;
  int sync_calls_ = 0;
};

} // namespace

TEST(BackendManagerTest, CanOverrideBackendFactoryForDeviceType) {
  BackendManager::register_backend(
      DeviceType::CPU,
      [](Device) { return std::make_shared<CPUBackend>(); });

  auto backend = BackendManager::get(Device{DeviceType::CPU, 0});
  EXPECT_NE(backend, nullptr);
}

TEST(BackendManagerTest, ForwardsDeviceIndexToFactoryAndCachesPerIndex) {
  std::mutex mu;
  std::vector<int> requested_indices;

  BackendManager::register_backend(DeviceType::UNKNOWN, [&](Device d) {
    std::lock_guard<std::mutex> lock(mu);
    requested_indices.push_back(d.index);
    return std::make_shared<CPUBackend>();
  });

  auto b0 = BackendManager::get(Device{DeviceType::UNKNOWN, 0});
  auto b1 = BackendManager::get(Device{DeviceType::UNKNOWN, 1});
  auto b1_again = BackendManager::get(Device{DeviceType::UNKNOWN, 1});

  EXPECT_NE(b0, nullptr);
  EXPECT_NE(b1, nullptr);
  EXPECT_EQ(b1, b1_again);

  std::lock_guard<std::mutex> lock(mu);
  ASSERT_EQ(requested_indices.size(), 2);
  EXPECT_EQ(requested_indices[0], 0);
  EXPECT_EQ(requested_indices[1], 1);
}

TEST(BackendManagerTest, ReRegisteringBackendTypeClearsCachedInstances) {
  int generation = 0;

  BackendManager::register_backend(DeviceType::UNKNOWN, [&](Device) {
    ++generation;
    return std::make_shared<CPUBackend>();
  });

  auto first = BackendManager::get(Device{DeviceType::UNKNOWN, 7});
  EXPECT_NE(first, nullptr);
  EXPECT_EQ(generation, 1);

  BackendManager::register_backend(DeviceType::UNKNOWN, [&](Device) {
    ++generation;
    return std::make_shared<CPUBackend>();
  });

  auto second = BackendManager::get(Device{DeviceType::UNKNOWN, 7});
  EXPECT_NE(second, nullptr);
  EXPECT_NE(first, second);
  EXPECT_EQ(generation, 2);
}

TEST(BackendManagerTest, ExposesBackendCapabilitySurface) {
  auto backend = BackendManager::get(Device{DeviceType::CPU, 0});
  ASSERT_NE(backend, nullptr);
  EXPECT_STREQ(backend->name(), "cpu");
  EXPECT_TRUE(backend->supports(BackendFeature::Matmul, DataType::Float32));
  EXPECT_TRUE(backend->supports(BackendFeature::Matmul, DataType::Float32,
                                Shape{2, 2}));
  EXPECT_FALSE(backend->supports(BackendFeature::Matmul, DataType::Float16));
  EXPECT_TRUE(backend->supports(BackendFeature::RandomFill, DataType::Float16));
  EXPECT_FALSE(backend->supports(BackendFeature::RandomFill, DataType::Int32));
}

TEST(BackendManagerTest, CapabilityDTypePolicyLivesInOnePlace) {
  EXPECT_TRUE(supports_backend_feature_dtype(BackendFeature::ElementwiseBinary,
                                            DataType::Float32));
  EXPECT_FALSE(supports_backend_feature_dtype(BackendFeature::ElementwiseBinary,
                                             DataType::Float16));
  EXPECT_TRUE(supports_backend_feature_dtype(BackendFeature::RandomFill,
                                            DataType::Float16));
  EXPECT_FALSE(supports_backend_feature_dtype(BackendFeature::RandomFill,
                                             DataType::Int32));
  EXPECT_TRUE(supports_backend_feature_dtype(BackendFeature::Reduction,
                                            DataType::Float32));
}

TEST(BackendRegistryTest, LocalRegistryAllowsIsolatedOverridesAndCacheControl) {
  BackendRegistry registry;
  int generation = 0;

  registry.register_backend(DeviceType::UNKNOWN, [&](Device device) {
    ++generation;
    return std::make_shared<PartialMatmulBackend>(device.index);
  });

  auto first = registry.get(Device{DeviceType::UNKNOWN, 3});
  auto again = registry.get(Device{DeviceType::UNKNOWN, 3});
  EXPECT_EQ(first, again);
  EXPECT_EQ(generation, 1);

  registry.clear_cache(DeviceType::UNKNOWN);
  auto second = registry.get(Device{DeviceType::UNKNOWN, 3});
  EXPECT_NE(first, second);
  EXPECT_EQ(generation, 2);
}

TEST(BackendRegistryTest, PartialBackendReportsCapabilitiesAndFallbackMetadata) {
  BackendRegistry registry;
  registry.register_backend(DeviceType::UNKNOWN, [](Device device) {
    return std::make_shared<PartialMatmulBackend>(device.index);
  });

  auto backend = registry.get(Device{DeviceType::UNKNOWN, 9});
  ASSERT_NE(backend, nullptr);
  EXPECT_STREQ(backend->name(), "partial_matmul");
  EXPECT_TRUE(backend->supports(BackendFeature::Matmul, DataType::Float32));
  EXPECT_TRUE(backend->supports(BackendFeature::Matmul, DataType::Float32,
                                Shape{2, 2}));
  EXPECT_FALSE(
      backend->supports(BackendFeature::ElementwiseBinary, DataType::Float32));

  const auto matmul_support =
      backend->query_support(BackendFeature::Matmul, DataType::Float32);
  EXPECT_TRUE(matmul_support.available);
  EXPECT_EQ(matmul_support.preferred_accumulation_dtype, DataType::Float32);

  const auto missing_support =
      backend->query_support(BackendFeature::ElementwiseBinary,
                             DataType::Float32);
  EXPECT_FALSE(missing_support.available);
  EXPECT_EQ(missing_support.fallback_policy,
            BackendFallbackPolicy::CPUFallback);
}

TEST(BackendManagerTest, ProfileWrapperDoesNotReportGpuTimeForCpuBackends) {
  ScopedEnvVar profile("MUNET_PROFILE", "1");
  Profiler::get().reset();

  std::shared_ptr<TimedAddBackend> base_backend;
  BackendManager::register_backend(DeviceType::UNKNOWN, [&](Device device) {
    base_backend = std::make_shared<TimedAddBackend>(device.index, false, 321.0);
    return base_backend;
  });

  const Device test_device{DeviceType::UNKNOWN, 42};
  Storage a(sizeof(float), test_device, DataType::Float32, {1});
  Storage b(sizeof(float), test_device, DataType::Float32, {1});
  Storage out(sizeof(float), test_device, DataType::Float32, {1});
  *static_cast<float *>(a.data()) = 1.25f;
  *static_cast<float *>(b.data()) = 2.75f;

  auto backend = BackendManager::get(test_device);
  auto info =
      compute_broadcast({1}, default_strides({1}), {1}, default_strides({1}));
  backend->add(a, b, out, info);

  const auto snapshot = Profiler::get().snapshot();
  const auto it = snapshot.stats.find("add");
  ASSERT_NE(it, snapshot.stats.end());
  EXPECT_GT(it->second.cpu_us, 0.0);
  EXPECT_DOUBLE_EQ(it->second.gpu_us, 0.0);
  ASSERT_NE(base_backend, nullptr);
  EXPECT_EQ(base_backend->sync_calls(), 0);
  EXPECT_EQ(*static_cast<float *>(out.data()), 4.0f);
}

TEST(BackendManagerTest, ProfileWrapperSynchronizesGpuBackendsToCaptureGpuTime) {
  ScopedEnvVar profile("MUNET_PROFILE", "1");
  Profiler::get().reset();

  std::shared_ptr<TimedAddBackend> base_backend;
  BackendManager::register_backend(DeviceType::UNKNOWN, [&](Device device) {
    base_backend = std::make_shared<TimedAddBackend>(device.index, true, 456.0);
    return base_backend;
  });

  const Device test_device{DeviceType::UNKNOWN, 43};
  Storage a(sizeof(float), test_device, DataType::Float32, {1});
  Storage b(sizeof(float), test_device, DataType::Float32, {1});
  Storage out(sizeof(float), test_device, DataType::Float32, {1});
  *static_cast<float *>(a.data()) = 3.0f;
  *static_cast<float *>(b.data()) = 4.0f;

  auto backend = BackendManager::get(test_device);
  auto info =
      compute_broadcast({1}, default_strides({1}), {1}, default_strides({1}));
  backend->add(a, b, out, info);

  const auto snapshot = Profiler::get().snapshot();
  const auto it = snapshot.stats.find("add");
  ASSERT_NE(it, snapshot.stats.end());
  EXPECT_GT(it->second.cpu_us, 0.0);
  EXPECT_DOUBLE_EQ(it->second.gpu_us, 456.0);
  ASSERT_NE(base_backend, nullptr);
  EXPECT_EQ(base_backend->sync_calls(), 1);
  EXPECT_EQ(*static_cast<float *>(out.data()), 7.0f);
}

TEST(BackendManagerTest, PartialBackendSupportsFallbackAndSupportedOpsEndToEnd) {
  std::shared_ptr<PartialMatmulBackend> base_backend;
  BackendManager::register_backend(DeviceType::UNKNOWN, [&](Device device) {
    base_backend = std::make_shared<PartialMatmulBackend>(device.index);
    return base_backend;
  });

  const Device partial_device{DeviceType::UNKNOWN, 0};
  Tensor a({2, 2}, partial_device, DataType::Float32);
  Tensor b({2, 2}, partial_device, DataType::Float32);
  a.fill_(1.0f);
  b.fill_(2.0f);

  Tensor add_out = a + b;
  EXPECT_EQ(add_out.device(), partial_device);
  Tensor add_cpu = add_out.to(Device{DeviceType::CPU, 0});
  const float *add_ptr = static_cast<const float *>(add_cpu.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(add_ptr[i], 3.0f);
  }

  ASSERT_NE(base_backend, nullptr);
  EXPECT_EQ(base_backend->matmul_calls(), 0);

  Tensor mm_out = a.matmul(b);
  EXPECT_EQ(mm_out.device(), partial_device);
  EXPECT_EQ(base_backend->matmul_calls(), 1);

  Tensor mm_cpu = mm_out.to(Device{DeviceType::CPU, 0});
  const float *mm_ptr = static_cast<const float *>(mm_cpu.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(mm_ptr[i], 4.0f);
  }
}

TEST(BackendManagerTest, PartialBackendSurfacesUnsupportedOpsDuringCapabilityCheck) {
  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device device) {
    return std::make_shared<PartialMatmulBackend>(device.index);
  });

  const Device partial_device{DeviceType::UNKNOWN, 0};
  Tensor a({1, 1}, partial_device, DataType::Float32);
  a.fill_(1.0f);

  try {
    (void)Tensor::cat({a, a}, 0);
    FAIL() << "Expected concat to fail for a backend without shape capability";
  } catch (const std::runtime_error &err) {
    const std::string message = err.what();
    EXPECT_NE(message.find("partial_matmul"), std::string::npos);
    EXPECT_NE(message.find("concat"), std::string::npos);
    EXPECT_NE(message.find("fallback policy: explicit_unsupported"),
              std::string::npos);
  }
}

TEST(BackendManagerTest, ProfilingCapturesDispatchPathMarkers) {
  ScopedEnvVar profile("MUNET_PROFILE", "1");
  Profiler::get().reset();

  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device device) {
    return std::make_shared<PartialMatmulBackend>(device.index);
  });

  const Device partial_device{DeviceType::UNKNOWN, 0};
  Tensor a({2, 2}, partial_device, DataType::Float32);
  Tensor b({2, 2}, partial_device, DataType::Float32);
  a.fill_(1.0f);
  b.fill_(2.0f);

  (void)(a + b);
  (void)a.matmul(b);

  const auto snapshot = Profiler::get().snapshot();
  EXPECT_NE(snapshot.stats.find("dispatch.resolve.cpu_fallback.Add"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("dispatch.resolve.backend.Matmul"),
            snapshot.stats.end());
}
