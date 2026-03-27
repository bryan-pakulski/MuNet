#include "backend.hpp"
#include "backend/cpu_backend.hpp"
#include "core/all_reduce_runtime.hpp"
#include "core/util/profiler.hpp"
#include "tensor.hpp"
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace munet;

namespace {

class ScopedProfileOverride {
public:
  explicit ScopedProfileOverride(bool enabled) {
    set_profile_enabled_override(enabled);
  }

  ~ScopedProfileOverride() { set_profile_enabled_override(std::nullopt); }
};

class ScopedDebugOverride {
public:
  explicit ScopedDebugOverride(bool enabled) {
    set_debug_enabled_override(enabled);
  }

  ~ScopedDebugOverride() { set_debug_enabled_override(std::nullopt); }
};

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    const char *existing = std::getenv(name_);
    if (existing) {
      had_original_ = true;
      original_value_ = existing;
    }
    setenv(name_, value, 1);
  }

  ~ScopedEnvVar() {
    if (had_original_) {
      setenv(name_, original_value_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

private:
  const char *name_;
  bool had_original_{false};
  std::string original_value_;
};

class PartialMatmulBackend : public Backend,
                             public BackendAllocationTransferCapability,
                             public BackendBlasCapability {
public:
  explicit PartialMatmulBackend(int device_index)
      : device_index_(device_index) {}

  ~PartialMatmulBackend() {
    for (void *p : allocated_ptrs_)
      std::free(p);
  }

  const char *name() const override { return "partial_matmul"; }

  BackendAllocationTransferCapability *
  allocation_transfer_capability() override {
    return this;
  }
  const BackendAllocationTransferCapability *
  allocation_transfer_capability() const override {
    return this;
  }

  BackendBlasCapability *blas_capability() override { return this; }
  const BackendBlasCapability *blas_capability() const override { return this; }

  DataType preferred_accumulation_dtype(BackendFeature feature,
                                        DataType dtype) const override {
    if (feature == BackendFeature::Matmul && dtype == DataType::Float16) {
      return DataType::Float32;
    }
    return Backend::preferred_accumulation_dtype(feature, dtype);
  }

  BackendFallbackPolicy
  preferred_fallback_policy(BackendFeature feature,
                            DataType dtype) const override {
    if (feature == BackendFeature::ElementwiseBinary) {
      return BackendFallbackPolicy::CPUFallback;
    }
    return Backend::preferred_fallback_policy(feature, dtype);
  }

  double get_last_kernel_time_us() override { return 0.0; }

  void *allocate(size_t bytes) override {
    void *ptr = std::malloc(bytes);
    allocated_ptrs_.insert(ptr);
    return ptr;
  }
  void deallocate(void *ptr) override {
    if (ptr) {
      allocated_ptrs_.erase(ptr);
      std::free(ptr);
    }
  }
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

  void batched_matmul(const Storage &a, const Storage &b, Storage &out,
                      int batch, int M, int K, int N, bool transA, bool transB,
                      int64_t stride_a, int64_t stride_b, int64_t stride_out) override {
    ++matmul_calls_;  // Reuse the same counter
    const float *ap = static_cast<const float *>(a.data());
    const float *bp = static_cast<const float *>(b.data());
    float *op = static_cast<float *>(out.data());

    for (int b_idx = 0; b_idx < batch; ++b_idx) {
      const float *a_ptr = ap + b_idx * stride_a;
      const float *b_ptr = bp + b_idx * stride_b;
      float *o_ptr = op + b_idx * stride_out;
      
      // Call regular matmul for each batch
      matmul(a, b, out, M, K, N, transA, transB);
    }
  }

  void to_contiguous(const Storage &src, Storage &dst, const Shape &shape,
                     const Strides &strides, size_t offset) override {
    const char *src_ptr = static_cast<const char *>(src.data());
    char *dst_ptr = static_cast<char *>(dst.data());
    size_t elem_size = dtype_size(src.dtype());
    size_t total = numel(shape);
    for (size_t linear = 0; linear < total; ++linear) {
      size_t rem = linear;
      size_t src_off = offset;
      for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        src_off += (rem % shape[d]) * strides[d];
        rem /= shape[d];
      }
      std::memcpy(dst_ptr + linear * elem_size, src_ptr + src_off * elem_size,
                  elem_size);
    }
  }

  int device_index() const { return device_index_; }
  int matmul_calls() const { return matmul_calls_; }

private:
  int device_index_ = 0;
  int matmul_calls_ = 0;
  std::unordered_set<void *> allocated_ptrs_;
};

class TimedAddBackend : public Backend,
                        public BackendAllocationTransferCapability,
                        public BackendElementwiseCapability {
public:
  TimedAddBackend(int device_index, bool gpu_timing, double kernel_time_us)
      : device_index_(device_index), gpu_timing_(gpu_timing),
        kernel_time_us_(kernel_time_us) {}

  ~TimedAddBackend() {
    for (void *p : allocated_ptrs_)
      std::free(p);
  }

  const char *name() const override {
    return gpu_timing_ ? "timed_gpu_add" : "timed_cpu_add";
  }

  BackendAllocationTransferCapability *
  allocation_transfer_capability() override {
    return this;
  }
  const BackendAllocationTransferCapability *
  allocation_transfer_capability() const override {
    return this;
  }

  BackendElementwiseCapability *elementwise_capability() override {
    return this;
  }
  const BackendElementwiseCapability *elementwise_capability() const override {
    return this;
  }

  void *allocate(size_t bytes) override {
    void *ptr = std::malloc(bytes);
    allocated_ptrs_.insert(ptr);
    return ptr;
  }
  void deallocate(void *ptr) override {
    if (ptr) {
      allocated_ptrs_.erase(ptr);
      std::free(ptr);
    }
  }

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

  void to_contiguous(const Storage &src, Storage &dst, const Shape &shape,
                     const Strides &strides, size_t offset) override {
    const char *src_ptr = static_cast<const char *>(src.data());
    char *dst_ptr = static_cast<char *>(dst.data());
    size_t elem_size = dtype_size(src.dtype());
    size_t total = numel(shape);
    for (size_t linear = 0; linear < total; ++linear) {
      size_t rem = linear;
      size_t src_off = offset;
      for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        src_off += (rem % shape[d]) * strides[d];
        rem /= shape[d];
      }
      std::memcpy(dst_ptr + linear * elem_size, src_ptr + src_off * elem_size,
                  elem_size);
    }
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
  void exp(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void log(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void sqrt(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void rsqrt(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void sin(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void cos(const Storage &, Storage &, size_t) override {
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
  std::unordered_set<void *> allocated_ptrs_;
};

class ReusingTimedAddBackend : public Backend,
                               public BackendAllocationTransferCapability,
                               public BackendElementwiseCapability {
public:
  explicit ReusingTimedAddBackend(int device_index)
      : device_index_(device_index) {}

  ~ReusingTimedAddBackend() {
    for (void *p : all_mallocs_)
      std::free(p);
  }

  const char *name() const override { return "reusing_timed_add"; }

  BackendAllocationTransferCapability *
  allocation_transfer_capability() override {
    return this;
  }
  const BackendAllocationTransferCapability *
  allocation_transfer_capability() const override {
    return this;
  }

  BackendElementwiseCapability *elementwise_capability() override {
    return this;
  }
  const BackendElementwiseCapability *elementwise_capability() const override {
    return this;
  }

  void *allocate(size_t bytes) override {
    auto &bucket = free_blocks_[bytes];
    if (!bucket.empty()) {
      void *ptr = bucket.back();
      bucket.pop_back();
      allocated_sizes_[ptr] = bytes;
      return ptr;
    }

    void *ptr = std::malloc(bytes);
    allocated_sizes_[ptr] = bytes;
    all_mallocs_.push_back(ptr);
    return ptr;
  }

  void deallocate(void *ptr) override {
    if (!ptr)
      return;
    const size_t bytes = allocation_size(ptr);
    if (bytes > 0)
      free_blocks_[bytes].push_back(ptr);
  }

  void memset(void *ptr, int value, size_t bytes) override {
    std::memset(ptr, value, bytes);
  }

  void copy(const void *src, void *dst, size_t bytes, Device, Device) override {
    std::memcpy(dst, src, bytes);
  }

  void synchronize() override {
    ++sync_calls_;
    last_kernel_time_us_ = 77.0;
  }

  void to_contiguous(const Storage &src, Storage &dst, const Shape &shape,
                     const Strides &strides, size_t offset) override {
    const char *src_ptr = static_cast<const char *>(src.data());
    char *dst_ptr = static_cast<char *>(dst.data());
    size_t elem_size = dtype_size(src.dtype());
    size_t total = numel(shape);
    for (size_t linear = 0; linear < total; ++linear) {
      size_t rem = linear;
      size_t src_off = offset;
      for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        src_off += (rem % shape[d]) * strides[d];
        rem /= shape[d];
      }
      std::memcpy(dst_ptr + linear * elem_size, src_ptr + src_off * elem_size,
                  elem_size);
    }
  }

  void all_reduce(Storage &, size_t) override {}
  double get_last_kernel_time_us() override { return last_kernel_time_us_; }
  bool reports_gpu_kernel_time() const override { return true; }

  void add(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &) override {
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
  void exp(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void log(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void sqrt(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void rsqrt(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void sin(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void cos(const Storage &, Storage &, size_t) override {
    throw std::runtime_error("unused");
  }
  void softmax(const Storage &, Storage &, int, int) override {
    throw std::runtime_error("unused");
  }
  void softmax_backward(const Storage &, const Storage &, Storage &, int,
                        int) override {
    throw std::runtime_error("unused");
  }

  int sync_calls() const { return sync_calls_; }

private:
  size_t allocation_size(void *ptr) const {
    auto it = allocated_sizes_.find(ptr);
    return it == allocated_sizes_.end() ? 0 : it->second;
  }

  int device_index_ = 0;
  double last_kernel_time_us_ = 0.0;
  int sync_calls_ = 0;
  std::unordered_map<void *, size_t> allocated_sizes_;
  std::unordered_map<size_t, std::vector<void *>> free_blocks_;
  std::vector<void *> all_mallocs_;
};

} // namespace

TEST(BackendManagerTest, CanOverrideBackendFactoryForDeviceType) {
  BackendManager::register_backend(
      DeviceType::CPU, [](Device) { return std::make_shared<CPUBackend>(); });

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

TEST(BackendManagerTest, SeparatesDeployAndTrainingOnlyBackendFeatures) {
  EXPECT_FALSE(is_training_only_backend_feature(BackendFeature::Matmul));
  EXPECT_FALSE(is_training_only_backend_feature(BackendFeature::Convolution));
  EXPECT_TRUE(is_training_only_backend_feature(BackendFeature::OptimizerStep));
}

TEST(BackendManagerTest, ConstrainedFallbackPolicyIsDefinedPerFeature) {
  EXPECT_EQ(backend_feature_default_fallback_policy(BackendFeature::Matmul),
            BackendFallbackPolicy::CPUFallback);
  EXPECT_EQ(backend_feature_default_fallback_policy(BackendFeature::RandomFill),
            BackendFallbackPolicy::CPUFallback);
  EXPECT_EQ(
      backend_feature_default_fallback_policy(BackendFeature::Convolution),
      BackendFallbackPolicy::ExplicitUnsupported);
  EXPECT_EQ(
      backend_feature_default_fallback_policy(BackendFeature::OptimizerStep),
      BackendFallbackPolicy::ExplicitUnsupported);
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

TEST(BackendRegistryTest,
     PartialBackendReportsCapabilitiesAndFallbackMetadata) {
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

  const auto missing_support = backend->query_support(
      BackendFeature::ElementwiseBinary, DataType::Float32);
  EXPECT_FALSE(missing_support.available);
  EXPECT_EQ(missing_support.fallback_policy,
            BackendFallbackPolicy::CPUFallback);
}

TEST(BackendManagerTest, ProfileWrapperDoesNotReportGpuTimeForCpuBackends) {
  ScopedProfileOverride profile(true);
  Profiler::get().reset();

  std::shared_ptr<TimedAddBackend> base_backend;
  BackendManager::register_backend(DeviceType::UNKNOWN, [&](Device device) {
    base_backend =
        std::make_shared<TimedAddBackend>(device.index, false, 321.0);
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

TEST(BackendManagerTest,
     ProfileWrapperSynchronizesGpuBackendsToCaptureGpuTime) {
  ScopedProfileOverride profile(true);
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

TEST(BackendManagerTest,
     PartialBackendSupportsFallbackAndSupportedOpsEndToEnd) {
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

TEST(BackendManagerTest,
     PartialBackendSurfacesUnsupportedOpsDuringCapabilityCheck) {
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
  ScopedProfileOverride profile(true);
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
  const auto feature_reason =
      snapshot.stats.find("dispatch.fallback.reason.feature");
  ASSERT_NE(feature_reason, snapshot.stats.end());
  EXPECT_NE(feature_reason->second.last_shape.find("op=Add"),
            std::string::npos);
  EXPECT_NE(feature_reason->second.last_shape.find("reason=feature"),
            std::string::npos);
}

TEST(BackendManagerTest, ProfilingCapturesDTypeFallbackReasonMarkers) {
  ScopedProfileOverride profile(true);
  Profiler::get().reset();

  const Device cpu{DeviceType::CPU, 0};
  Tensor a32({2, 2}, cpu, DataType::Float32);
  Tensor b32({2, 2}, cpu, DataType::Float32);
  a32.fill_(1.0f);
  b32.fill_(1.0f);

  Tensor a = a32.to(DataType::Float16);
  Tensor b = b32.to(DataType::Float16);
  Profiler::get().reset();

  Tensor c = a.matmul(b);
  (void)c;

  const auto snapshot = Profiler::get().snapshot();
  const auto dtype_reason =
      snapshot.stats.find("dispatch.fallback.reason.dtype");
  ASSERT_NE(dtype_reason, snapshot.stats.end());
  EXPECT_NE(dtype_reason->second.last_shape.find("op=Matmul"),
            std::string::npos);
  EXPECT_NE(dtype_reason->second.last_shape.find("dtype=float16"),
            std::string::npos);
  EXPECT_NE(dtype_reason->second.last_shape.find("reason=dtype"),
            std::string::npos);
}

TEST(BackendManagerTest, ProfilingCapturesAllocatorAndSynchronizationMarkers) {
  ScopedProfileOverride profile(true);
  Profiler::get().reset();

  std::shared_ptr<ReusingTimedAddBackend> base_backend;
  BackendManager::register_backend(DeviceType::UNKNOWN, [&](Device device) {
    base_backend = std::make_shared<ReusingTimedAddBackend>(device.index);
    return base_backend;
  });

  const auto backend = BackendManager::get(Device{DeviceType::UNKNOWN, 7});
  ASSERT_NE(backend, nullptr);

  void *first = backend->allocate(128);
  backend->deallocate(first);
  void *second = backend->allocate(128);
  ASSERT_EQ(first, second);
  backend->deallocate(second);

  void *large = backend->allocate(20 * 1024 * 1024);
  backend->deallocate(large);
  backend->synchronize();

  const Device test_device{DeviceType::UNKNOWN, 7};
  Tensor a({1}, test_device, DataType::Float32);
  Tensor b({1}, test_device, DataType::Float32);
  a.fill_(2.0f);
  b.fill_(3.0f);
  (void)(a + b);

  const auto snapshot = Profiler::get().snapshot();
  EXPECT_NE(snapshot.stats.find("allocator.reuse_miss.reusing_timed_add"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("allocator.reuse_hit.reusing_timed_add"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("allocator.pool_growth.reusing_timed_add"),
            snapshot.stats.end());
  EXPECT_NE(
      snapshot.stats.find("allocator.large_alloc_slow_path.reusing_timed_add"),
      snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("allocator.deallocate.reusing_timed_add"),
            snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("sync.explicit.reusing_timed_add"),
            snapshot.stats.end());
  const auto implicit_sync =
      snapshot.stats.find("sync.implicit_timing.reusing_timed_add");
  ASSERT_NE(implicit_sync, snapshot.stats.end());
  EXPECT_NE(implicit_sync->second.last_shape.find("trigger=add"),
            std::string::npos);
  ASSERT_NE(base_backend, nullptr);
  EXPECT_GE(base_backend->sync_calls(), 2);
}

TEST(BackendManagerTest, ProfilingCapturesDirectionalTransferMarkers) {
  ScopedProfileOverride profile(true);
  Profiler::get().reset();
  BackendManager::registry().clear_cache(DeviceType::CPU);

  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device device) {
    return std::make_shared<PartialMatmulBackend>(device.index);
  });

  const Device cpu{DeviceType::CPU, 0};
  const Device unknown0{DeviceType::UNKNOWN, 0};
  const Device unknown1{DeviceType::UNKNOWN, 1};

  Tensor cpu_tensor({1}, cpu, DataType::Float32);
  cpu_tensor.fill_(1.0f);

  Tensor to_unknown = cpu_tensor.to(unknown0);
  Tensor to_unknown_other = to_unknown.to(unknown1);
  Tensor back_to_cpu = to_unknown.to(cpu);
  Tensor half = cpu_tensor.to(DataType::Float16);
  (void)to_unknown_other;
  (void)back_to_cpu;
  (void)half;

  Tensor cpu_copy_dst({1}, cpu, DataType::Float32);
  BackendManager::get(cpu)->copy(cpu_tensor.data(), cpu_copy_dst.data(),
                                 cpu_tensor.bytes(), cpu, cpu);

  const auto snapshot = Profiler::get().snapshot();
  EXPECT_NE(snapshot.stats.find("transfer.h2d"), snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("transfer.d2d"), snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("transfer.d2h"), snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("transfer.cpu_copy"), snapshot.stats.end());
  EXPECT_NE(snapshot.stats.find("transfer.dtype_convert"),
            snapshot.stats.end());
}

TEST(BackendManagerTest, TraceContextCorrelatesTransferDispatchAndLogs) {
  ScopedProfileOverride profile(true);
  ScopedDebugOverride debug(true);
  Profiler::get().reset();
  BackendManager::registry().clear_cache(DeviceType::CPU);

  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device device) {
    return std::make_shared<PartialMatmulBackend>(device.index);
  });

  const Device cpu{DeviceType::CPU, 0};
  const Device partial_device{DeviceType::UNKNOWN, 0};

  Tensor cpu_tensor({1}, cpu, DataType::Float32);
  cpu_tensor.fill_(1.0f);

  testing::internal::CaptureStderr();
  {
    ScopedTraceContext trace("run.forward", 4242);
    Tensor partial = cpu_tensor.to(partial_device);
    (void)(partial + partial);
  }
  const std::string stderr_output = testing::internal::GetCapturedStderr();

  const auto snapshot = Profiler::get().snapshot();
  const auto transfer = snapshot.stats.find("transfer.h2d");
  ASSERT_NE(transfer, snapshot.stats.end());
  EXPECT_NE(transfer->second.last_shape.find("trace_id=4242"),
            std::string::npos);
  EXPECT_NE(transfer->second.last_shape.find("span=run.forward"),
            std::string::npos);

  const auto dispatch =
      snapshot.stats.find("dispatch.resolve.cpu_fallback.Add");
  ASSERT_NE(dispatch, snapshot.stats.end());
  EXPECT_NE(dispatch->second.last_shape.find("trace_id=4242"),
            std::string::npos);
  EXPECT_NE(dispatch->second.last_shape.find("span=run.forward"),
            std::string::npos);

  const auto fallback = snapshot.stats.find("dispatch.fallback.reason.feature");
  ASSERT_NE(fallback, snapshot.stats.end());
  EXPECT_NE(fallback->second.last_shape.find("trace_id=4242"),
            std::string::npos);
  EXPECT_NE(fallback->second.last_shape.find("span=run.forward"),
            std::string::npos);

  EXPECT_NE(stderr_output.find("[trace_id=4242 span=run.forward]"),
            std::string::npos);
  EXPECT_NE(stderr_output.find("dispatch_fallback"), std::string::npos);
}

TEST(BackendManagerTest, CpuAllReduceAggregatesAcrossParticipants) {
  ScopedEnvVar world_size("MUNET_ALLREDUCE_WORLD_SIZE", "2");
  BackendManager::registry().clear_cache(DeviceType::CPU);

  const Device cpu{DeviceType::CPU, 0};
  auto backend = BackendManager::get(cpu);
  ASSERT_NE(backend, nullptr);

  Storage a(2 * sizeof(float), cpu, DataType::Float32, {2});
  Storage b(2 * sizeof(float), cpu, DataType::Float32, {2});
  float *ap = static_cast<float *>(a.data());
  float *bp = static_cast<float *>(b.data());
  ap[0] = 1.0f;
  ap[1] = 2.0f;
  bp[0] = 3.0f;
  bp[1] = 4.0f;

  std::thread t1([&] { backend->all_reduce(a, 2); });
  std::thread t2([&] { backend->all_reduce(b, 2); });
  t1.join();
  t2.join();

  EXPECT_FLOAT_EQ(ap[0], 4.0f);
  EXPECT_FLOAT_EQ(ap[1], 6.0f);
  EXPECT_FLOAT_EQ(bp[0], 4.0f);
  EXPECT_FLOAT_EQ(bp[1], 6.0f);
}

TEST(BackendManagerTest,
     AllReduceHostPathRequiresExplicitOverrideForAcceleratorDevices) {
  ScopedEnvVar world_size("MUNET_ALLREDUCE_WORLD_SIZE", "2");
  ScopedEnvVar mode("MUNET_ALLREDUCE_MODE", "device_native");
  BackendManager::registry().clear_cache(DeviceType::UNKNOWN);
  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device device) {
    return std::make_shared<PartialMatmulBackend>(device.index);
  });

  const Device unknown{DeviceType::UNKNOWN, 0};
  Storage a(2 * sizeof(float), unknown, DataType::Float32, {2});
  float *ap = static_cast<float *>(a.data());
  ap[0] = 1.0f;
  ap[1] = 2.0f;

  EXPECT_THROW(
      detail::all_reduce_via_host(a, 2, a.backend(), Device{DeviceType::CUDA, 0}),
      std::runtime_error);
}

TEST(BackendManagerTest, ForcedHostAllReduceSupportsAcceleratorDeviceKeys) {
  ScopedEnvVar world_size("MUNET_ALLREDUCE_WORLD_SIZE", "2");
  ScopedEnvVar mode("MUNET_ALLREDUCE_MODE", "device_native");
  BackendManager::registry().clear_cache(DeviceType::UNKNOWN);
  BackendManager::register_backend(DeviceType::UNKNOWN, [](Device device) {
    return std::make_shared<PartialMatmulBackend>(device.index);
  });

  const Device unknown{DeviceType::UNKNOWN, 0};
  Storage a(2 * sizeof(float), unknown, DataType::Float32, {2});
  Storage b(2 * sizeof(float), unknown, DataType::Float32, {2});
  float *ap = static_cast<float *>(a.data());
  float *bp = static_cast<float *>(b.data());
  ap[0] = 1.0f;
  ap[1] = 2.0f;
  bp[0] = 3.0f;
  bp[1] = 4.0f;

  std::thread t1([&] {
    detail::all_reduce_via_host(a, 2, a.backend(), Device{DeviceType::CUDA, 0},
                                true);
  });
  std::thread t2([&] {
    detail::all_reduce_via_host(b, 2, b.backend(), Device{DeviceType::CUDA, 0},
                                true);
  });
  t1.join();
  t2.join();

  EXPECT_FLOAT_EQ(ap[0], 4.0f);
  EXPECT_FLOAT_EQ(ap[1], 6.0f);
  EXPECT_FLOAT_EQ(bp[0], 4.0f);
  EXPECT_FLOAT_EQ(bp[1], 6.0f);
}

TEST(BackendManagerTest, CudaMultiDeviceAllReduceAggregatesAcrossGpuIndices) {
  const Device cuda0{DeviceType::CUDA, 0};
  const Device cuda1{DeviceType::CUDA, 1};
  try {
    (void)Tensor({1}, cuda0, DataType::Float32);
    (void)Tensor({1}, cuda1, DataType::Float32);
  } catch (const std::runtime_error &) {
    GTEST_SKIP() << "CUDA multi-device environment unavailable";
  }

  ScopedEnvVar world_size("MUNET_ALLREDUCE_WORLD_SIZE", "2");
  ScopedEnvVar mode("MUNET_ALLREDUCE_MODE", "device_native");

  Tensor a_cpu({2}, Device{DeviceType::CPU, 0}, DataType::Float32);
  Tensor b_cpu({2}, Device{DeviceType::CPU, 0}, DataType::Float32);
  float *ac = static_cast<float *>(a_cpu.data());
  float *bc = static_cast<float *>(b_cpu.data());
  ac[0] = 1.0f;
  ac[1] = 2.0f;
  bc[0] = 3.0f;
  bc[1] = 4.0f;

  Tensor a = a_cpu.to(cuda0);
  Tensor b = b_cpu.to(cuda1);
  auto backend0 = BackendManager::get(cuda0);
  auto backend1 = BackendManager::get(cuda1);

  std::exception_ptr eptr = nullptr;
  std::mutex err_mtx;
  auto run_reduce = [&](std::shared_ptr<Backend> backend, Tensor &tensor) {
    try {
      backend->all_reduce(*tensor.impl_->storage, 2);
    } catch (...) {
      std::lock_guard<std::mutex> lock(err_mtx);
      if (!eptr)
        eptr = std::current_exception();
    }
  };

  std::thread t0(run_reduce, backend0, std::ref(a));
  std::thread t1(run_reduce, backend1, std::ref(b));
  t0.join();
  t1.join();

  if (eptr) {
    try {
      std::rethrow_exception(eptr);
    } catch (const std::runtime_error &err) {
      GTEST_SKIP() << "CUDA multi-device all_reduce unavailable at runtime: "
                   << err.what();
    }
  }

  Tensor a_out = a.to(Device{DeviceType::CPU, 0});
  Tensor b_out = b.to(Device{DeviceType::CPU, 0});
  const float *ao = static_cast<const float *>(a_out.data());
  const float *bo = static_cast<const float *>(b_out.data());
  EXPECT_FLOAT_EQ(ao[0], 4.0f);
  EXPECT_FLOAT_EQ(ao[1], 6.0f);
  EXPECT_FLOAT_EQ(bo[0], 4.0f);
  EXPECT_FLOAT_EQ(bo[1], 6.0f);
}

TEST(BackendManagerTest,
     VulkanMultiDeviceTrainingGradientAllReduceAggregatesAcrossGpuIndices) {
  const Device vk0{DeviceType::VULKAN, 0};
  const Device vk1{DeviceType::VULKAN, 1};
  try {
    (void)Tensor({1}, vk0, DataType::Float32);
    (void)Tensor({1}, vk1, DataType::Float32);
  } catch (const std::runtime_error &) {
    GTEST_SKIP() << "Vulkan multi-device environment unavailable";
  }

  ScopedEnvVar world_size("MUNET_ALLREDUCE_WORLD_SIZE", "2");
  ScopedEnvVar mode("MUNET_ALLREDUCE_MODE", "device_native");

  // Simulate two training replicas with per-replica gradient buffers.
  Tensor g0_cpu({2}, Device{DeviceType::CPU, 0}, DataType::Float32);
  Tensor g1_cpu({2}, Device{DeviceType::CPU, 0}, DataType::Float32);
  float *g0p = static_cast<float *>(g0_cpu.data());
  float *g1p = static_cast<float *>(g1_cpu.data());
  g0p[0] = 0.5f;
  g0p[1] = 1.5f;
  g1p[0] = 1.0f;
  g1p[1] = 2.0f;

  Tensor g0 = g0_cpu.to(vk0);
  Tensor g1 = g1_cpu.to(vk1);
  auto backend0 = BackendManager::get(vk0);
  auto backend1 = BackendManager::get(vk1);

  std::exception_ptr eptr = nullptr;
  std::mutex err_mtx;
  auto run_reduce = [&](std::shared_ptr<Backend> backend, Tensor &tensor) {
    try {
      backend->all_reduce(*tensor.impl_->storage, 2);
    } catch (...) {
      std::lock_guard<std::mutex> lock(err_mtx);
      if (!eptr)
        eptr = std::current_exception();
    }
  };

  std::thread t0(run_reduce, backend0, std::ref(g0));
  std::thread t1(run_reduce, backend1, std::ref(g1));
  t0.join();
  t1.join();

  if (eptr) {
    try {
      std::rethrow_exception(eptr);
    } catch (const std::runtime_error &err) {
      GTEST_SKIP() << "Vulkan multi-device all_reduce unavailable at runtime: "
                   << err.what();
    }
  }

  Tensor g0_out = g0.to(Device{DeviceType::CPU, 0});
  Tensor g1_out = g1.to(Device{DeviceType::CPU, 0});
  const float *o0 = static_cast<const float *>(g0_out.data());
  const float *o1 = static_cast<const float *>(g1_out.data());
  EXPECT_FLOAT_EQ(o0[0], 1.5f);
  EXPECT_FLOAT_EQ(o0[1], 3.5f);
  EXPECT_FLOAT_EQ(o1[0], 1.5f);
  EXPECT_FLOAT_EQ(o1[1], 3.5f);
}

TEST(BackendManagerTest, MixedCpuCudaVulkanAllReduceAggregatesTogether) {
  const Device cpu{DeviceType::CPU, 0};
  const Device cuda{DeviceType::CUDA, 0};
  const Device vk{DeviceType::VULKAN, 0};
  try {
    (void)Tensor({1}, cpu, DataType::Float32);
    (void)Tensor({1}, cuda, DataType::Float32);
    (void)Tensor({1}, vk, DataType::Float32);
  } catch (const std::runtime_error &) {
    GTEST_SKIP() << "CPU/CUDA/Vulkan mixed backend environment unavailable";
  }

  ScopedEnvVar world_size("MUNET_ALLREDUCE_WORLD_SIZE", "3");
  ScopedEnvVar mode("MUNET_ALLREDUCE_MODE", "device_native");
  ScopedEnvVar group("MUNET_ALLREDUCE_GROUP", "mixed_cpu_cuda_vulkan");

  Tensor c0({2}, cpu, DataType::Float32);
  Tensor c1({2}, cpu, DataType::Float32);
  Tensor c2({2}, cpu, DataType::Float32);
  float *p0 = static_cast<float *>(c0.data());
  float *p1 = static_cast<float *>(c1.data());
  float *p2 = static_cast<float *>(c2.data());
  p0[0] = 1.0f;
  p0[1] = 2.0f;
  p1[0] = 2.0f;
  p1[1] = 3.0f;
  p2[0] = 3.0f;
  p2[1] = 4.0f;

  Tensor t_cpu = c0;
  Tensor t_cuda = c1.to(cuda);
  Tensor t_vk = c2.to(vk);

  auto b_cpu = BackendManager::get(cpu);
  auto b_cuda = BackendManager::get(cuda);
  auto b_vk = BackendManager::get(vk);
  std::exception_ptr eptr = nullptr;
  std::mutex err_mtx;

  auto run_reduce = [&](std::shared_ptr<Backend> backend, Tensor &tensor) {
    try {
      backend->all_reduce(*tensor.impl_->storage, 2);
    } catch (...) {
      std::lock_guard<std::mutex> lock(err_mtx);
      if (!eptr)
        eptr = std::current_exception();
    }
  };

  std::thread t0(run_reduce, b_cpu, std::ref(t_cpu));
  std::thread t1(run_reduce, b_cuda, std::ref(t_cuda));
  std::thread t2(run_reduce, b_vk, std::ref(t_vk));
  t0.join();
  t1.join();
  t2.join();

  if (eptr) {
    try {
      std::rethrow_exception(eptr);
    } catch (const std::runtime_error &err) {
      FAIL() << "Mixed CPU/CUDA/Vulkan all_reduce failed at runtime: "
             << err.what();
    } catch (const std::exception &err) {
      FAIL() << "Mixed CPU/CUDA/Vulkan all_reduce failed with exception: "
             << err.what();
    } catch (...) {
      FAIL() << "Mixed CPU/CUDA/Vulkan all_reduce failed with unknown "
                "exception";
    }
  }

  Tensor out_cpu = t_cpu.to(Device{DeviceType::CPU, 0});
  Tensor out_cuda = t_cuda.to(Device{DeviceType::CPU, 0});
  Tensor out_vk = t_vk.to(Device{DeviceType::CPU, 0});

  const float *oc = static_cast<const float *>(out_cpu.data());
  const float *ou = static_cast<const float *>(out_cuda.data());
  const float *ov = static_cast<const float *>(out_vk.data());

  EXPECT_FLOAT_EQ(oc[0], 6.0f);
  EXPECT_FLOAT_EQ(oc[1], 9.0f);
  EXPECT_FLOAT_EQ(ou[0], 6.0f);
  EXPECT_FLOAT_EQ(ou[1], 9.0f);
  EXPECT_FLOAT_EQ(ov[0], 6.0f);
  EXPECT_FLOAT_EQ(ov[1], 9.0f);
}
