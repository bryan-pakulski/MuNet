#include "backend/debug_backend.hpp"
#include "storage.hpp"
#include "core/util.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace munet {

namespace {

constexpr size_t kLargeAllocationSlowPathBytes = 16 * 1024 * 1024;

std::string profiler_name(const char *domain, const char *event,
                          const char *backend_name) {
  return std::string(domain) + "." + event + "." + backend_name;
}

void record_backend_profile_event(const char *domain, const char *event,
                                  const char *backend_name, double cpu_us,
                                  size_t bytes = 0,
                                  const std::string &detail = "") {
  if (!is_profile_enabled()) {
    return;
  }
  Profiler::get().record(profiler_name(domain, event, backend_name), cpu_us, 0.0,
                         bytes, detail);
}

// Intercepts all backend calls and enforces a synchronized sanity check
class DebugBackend : public Backend,
                     public BackendAllocationTransferCapability,
                     public BackendElementwiseCapability,
                     public BackendReductionCapability,
                     public BackendBlasCapability,
                     public BackendShapeCapability,
                     public BackendLossCapability,
                     public BackendSpatialCapability,
                     public BackendNormalizationCapability,
                     public BackendOptimizerCapability,
                     public BackendRandomCapability {
  std::shared_ptr<Backend> base_;
  std::mutex alloc_mtx_;
  std::unordered_map<void *, size_t> alloc_sizes_;
  std::unordered_map<size_t, size_t> pooled_blocks_by_size_;
  std::unordered_map<size_t, size_t> peak_blocks_by_size_;
  std::unordered_set<void *> reusable_ptrs_;

  void record_sync_event(const char *event, double cpu_us,
                         const std::string &detail = "") const {
    record_backend_profile_event("sync", event, base_->name(), cpu_us, 0,
                                 detail);
  }

  void record_allocator_event(const char *event, double cpu_us, size_t bytes,
                              const std::string &detail = "") const {
    record_backend_profile_event("allocator", event, base_->name(), cpu_us, bytes,
                                 detail);
  }

  bool should_collect_gpu_time() const { return base_->reports_gpu_kernel_time(); }

  void check(const std::string &name, double cpu_us,
             const Storage *out_storage = nullptr) {
    try {
      double gpu_us = 0.0;
      const bool collect_gpu_time = should_collect_gpu_time();

      if (collect_gpu_time && (is_debug_enabled() || is_profile_enabled())) {
        Timer sync_timer;
        base_->synchronize();
        record_sync_event("implicit_timing", sync_timer.elapsed_us(),
                          "trigger=" + name);
        gpu_us = base_->get_last_kernel_time_us();
      }

      // Full synchronization and NaN checks are expensive and should only run
      // in explicit debug mode, not in profile-only mode.
      if (is_debug_enabled()) {
        if (out_storage && out_storage->device().type == DeviceType::CPU) {
          float *data = (float *)out_storage->data();
          for (size_t i = 0; i < out_storage->size_bytes() / 4; ++i) {
            if (!std::isfinite(data[i])) {
              MUNET_ERROR << "Non-finite value detected in output of " << name
                          << " at index " << i << std::endl;
            }
          }
        }
      }

      if (is_profile_enabled()) {
        size_t bytes = out_storage ? out_storage->size_bytes() : 0;
        Profiler::get().record(name, cpu_us, gpu_us, bytes,
                               out_storage ? to_string(out_storage->shape())
                                           : "");
      }
    } catch (const std::exception &e) {
      MUNET_ERROR << "CRASH in/after " << name << ": " << e.what() << std::endl;
      throw;
    }
  }

public:
  DebugBackend(std::shared_ptr<Backend> base) : base_(base) {}

  const char *name() const override { return base_->name(); }

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
  BackendReductionCapability *reduction_capability() override { return this; }
  const BackendReductionCapability *reduction_capability() const override {
    return this;
  }
  BackendBlasCapability *blas_capability() override { return this; }
  const BackendBlasCapability *blas_capability() const override { return this; }
  BackendShapeCapability *shape_capability() override { return this; }
  const BackendShapeCapability *shape_capability() const override { return this; }
  BackendLossCapability *loss_capability() override { return this; }
  const BackendLossCapability *loss_capability() const override { return this; }
  BackendSpatialCapability *spatial_capability() override { return this; }
  const BackendSpatialCapability *spatial_capability() const override { return this; }
  BackendNormalizationCapability *normalization_capability() override {
    return this;
  }
  const BackendNormalizationCapability *normalization_capability() const override {
    return this;
  }
  BackendOptimizerCapability *optimizer_capability() override { return this; }
  const BackendOptimizerCapability *optimizer_capability() const override {
    return this;
  }
  BackendRandomCapability *random_capability() override { return this; }
  const BackendRandomCapability *random_capability() const override {
    return this;
  }

  BackendSupport query_support(BackendFeature feature, DataType dtype,
                               const Shape *shape) const override {
    return base_->query_support(feature, dtype, shape);
  }

  double get_last_kernel_time_us() override {
    return base_->get_last_kernel_time_us();
  }
  bool reports_gpu_kernel_time() const override {
    return base_->reports_gpu_kernel_time();
  }

  void *allocate(size_t bytes) override {
    Timer timer;
    void *ptr = base_->allocate(bytes);
    bool reuse_hit = false;
    bool pool_growth = false;
    {
      std::lock_guard<std::mutex> lock(alloc_mtx_);
      reuse_hit = reusable_ptrs_.erase(ptr) > 0;
      alloc_sizes_[ptr] = bytes;
      if (reuse_hit) {
        auto pooled_it = pooled_blocks_by_size_.find(bytes);
        if (pooled_it != pooled_blocks_by_size_.end() && pooled_it->second > 0) {
          --pooled_it->second;
        }
      } else {
        size_t &peak_blocks = peak_blocks_by_size_[bytes];
        size_t live_blocks = 0;
        for (const auto &[allocated_ptr, allocated_bytes] : alloc_sizes_) {
          if (allocated_bytes == bytes && reusable_ptrs_.count(allocated_ptr) == 0) {
            ++live_blocks;
          }
        }
        if (live_blocks > peak_blocks) {
          peak_blocks = live_blocks;
          pool_growth = true;
        }
      }
    }
    Profiler::get().record_alloc(bytes);
    const double cpu_us = timer.elapsed_us();
    record_allocator_event(reuse_hit ? "reuse_hit" : "reuse_miss", cpu_us, bytes);
    if (pool_growth) {
      record_allocator_event("pool_growth", cpu_us, bytes);
    }
    if (bytes >= kLargeAllocationSlowPathBytes) {
      record_allocator_event("large_alloc_slow_path", cpu_us, bytes);
    }
    return ptr;
  }
  void deallocate(void *ptr) override {
    size_t bytes = 0;
    Timer timer;
    {
      std::lock_guard<std::mutex> lock(alloc_mtx_);
      auto it = alloc_sizes_.find(ptr);
      if (it != alloc_sizes_.end()) {
        bytes = it->second;
        reusable_ptrs_.insert(ptr);
        pooled_blocks_by_size_[bytes]++;
      }
    }
    if (bytes > 0)
      Profiler::get().record_free(bytes);
    base_->deallocate(ptr);
    if (bytes > 0) {
      record_allocator_event("deallocate", timer.elapsed_us(), bytes);
    }
  }

  void memset(void *ptr, int value, size_t bytes) override {
    MUNET_DEBUG << "memset | " << bytes << " bytes" << std::endl;
    Timer t;
    base_->memset(ptr, value, bytes);
    check("memset", t.elapsed_us());
  }
  void copy(const void *src, void *dst, size_t bytes, Device src_dev,
            Device dst_dev) override {
    MUNET_DEBUG << "copy | " << bytes << " bytes" << std::endl;
    Timer t;
    base_->copy(src, dst, bytes, src_dev, dst_dev);
    check(transfer_profile_name(src_dev, dst_dev), t.elapsed_us());
  }
  void synchronize() override {
    Timer timer;
    base_->synchronize();
    record_sync_event("explicit", timer.elapsed_us());
  }
  void all_reduce(Storage &buffer, size_t num_elements) override {
    MUNET_DEBUG << "all_reduce | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->all_reduce(buffer, num_elements);
    check("all_reduce", t.elapsed_us(), &buffer);
  }

  void add(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    MUNET_DEBUG << "add (broadcast) | " << to_string(info.out_shape)
                << std::endl;
    Timer t;
    base_->add(a, b, out, info);
    check("add", t.elapsed_us(), &out);
  }

  void sub(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    MUNET_DEBUG << "sub (broadcast)" << std::endl;
    Timer t;
    base_->sub(a, b, out, info);
    check("sub", t.elapsed_us(), &out);
  }

  void mul(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    MUNET_DEBUG << "mul (broadcast)" << std::endl;
    Timer t;
    base_->mul(a, b, out, info);
    check("mul", t.elapsed_us(), &out);
  }

  void div(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    MUNET_DEBUG << "div (broadcast)" << std::endl;
    Timer t;
    base_->div(a, b, out, info);
    check("div", t.elapsed_us(), &out);
  }

  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) override {
    MUNET_DEBUG << "matmul | " << M << "x" << K << "x" << N << " matrix"
                << (transA ? " (transposed)" : "")
                << (transB ? " (transposed)" : "") << std::endl;
    Timer t;
    base_->matmul(a, b, out, M, K, N, transA, transB);
    check("matmul", t.elapsed_us(), &out);
  }

  void adam_step(Storage &params, const Storage &grads, Storage &exp_avg,
                 Storage &exp_avg_sq, float lr, float beta1, float beta2,
                 float eps, int step, size_t num_elements) override {
    MUNET_DEBUG << "adam_step | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->adam_step(params, grads, exp_avg, exp_avg_sq, lr, beta1, beta2, eps,
                     step, num_elements);
    check("adam_step", t.elapsed_us(), &params);
  }

  void broadcast_row(const Storage &src, Storage &dst, int rows,
                     int cols) override {
    MUNET_DEBUG << "broadcast_row | " << rows << "x" << cols << " matrix"
                << std::endl;
    Timer t;
    base_->broadcast_row(src, dst, rows, cols);
    check("broadcast_row", t.elapsed_us(), &dst);
  }

  void concat(const std::vector<Storage *> &inputs, Storage &out, int dim,
              const std::vector<Shape> &shapes) override {
    MUNET_DEBUG << "concat | " << dim << " dimension" << std::endl;
    Timer t;
    base_->concat(inputs, out, dim, shapes);
    check("concat", t.elapsed_us(), &out);
  }
  void concat_backward(const Storage &grad_out,
                       std::vector<Storage *> &grad_inputs, int dim,
                       const std::vector<Shape> &shapes) override {
    MUNET_DEBUG << "concat_backward | " << dim << " dimension" << std::endl;
    Timer t;
    base_->concat_backward(grad_out, grad_inputs, dim, shapes);
    check("concat_backward", t.elapsed_us(), &grad_out);
  }
  void relu(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "relu | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->relu(in, out, num_elements);
    check("relu", t.elapsed_us(), &out);
  }
  void relu_backward(const Storage &grad_out, const Storage &input,
                     Storage &grad_in, size_t num_elements) override {
    MUNET_DEBUG << "relu_backward | " << num_elements << " elements"
                << std::endl;
    Timer t;
    base_->relu_backward(grad_out, input, grad_in, num_elements);
    check("relu_backward", t.elapsed_us(), &grad_in);
  }
  void sigmoid(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "sigmoid | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->sigmoid(in, out, num_elements);
    check("sigmoid", t.elapsed_us(), &out);
  }
  void sigmoid_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, size_t num_elements) override {
    MUNET_DEBUG << "sigmoid_backward | " << num_elements << " elements"
                << std::endl;
    Timer t;
    base_->sigmoid_backward(grad_out, out, grad_in, num_elements);
    check("sigmoid_backward", t.elapsed_us(), &grad_in);
  }
  void exp(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "exp | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->exp(in, out, num_elements);
    check("exp", t.elapsed_us(), &out);
  }
  void log(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "log | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->log(in, out, num_elements);
    check("log", t.elapsed_us(), &out);
  }
  void sqrt(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "sqrt | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->sqrt(in, out, num_elements);
    check("sqrt", t.elapsed_us(), &out);
  }
  void rsqrt(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "rsqrt | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->rsqrt(in, out, num_elements);
    check("rsqrt", t.elapsed_us(), &out);
  }
  void sin(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "sin | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->sin(in, out, num_elements);
    check("sin", t.elapsed_us(), &out);
  }
  void cos(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "cos | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->cos(in, out, num_elements);
    check("cos", t.elapsed_us(), &out);
  }
  void softmax(const Storage &in, Storage &out, int batch_size,
               int num_classes) override {
    MUNET_DEBUG << "softmax | " << batch_size << " batches, " << num_classes
                << " classes" << std::endl;
    Timer t;
    base_->softmax(in, out, batch_size, num_classes);
    check("softmax", t.elapsed_us(), &out);
  }
  void softmax_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, int batch_size,
                        int num_classes) override {
    MUNET_DEBUG << "softmax_backward | " << batch_size << " batches, "
                << num_classes << " classes" << std::endl;
    Timer t;
    base_->softmax_backward(grad_out, out, grad_in, batch_size, num_classes);
    check("softmax_backward", t.elapsed_us(), &grad_in);
  }
  void cross_entropy(const Storage &logits, const Storage &targets,
                     Storage &out_loss, int batch_size, int num_classes,
                     int spatial) override {
    MUNET_DEBUG << "cross_entropy | " << batch_size << " batches, "
                << num_classes << " classes" << std::endl;
    Timer t;
    base_->cross_entropy(logits, targets, out_loss, batch_size, num_classes,
                         spatial);
    check("cross_entropy", t.elapsed_us(), &out_loss);
  }
  void cross_entropy_backward(const Storage &grad_out, const Storage &logits,
                              const Storage &targets, Storage &grad_in,
                              int batch_size, int num_classes,
                              int spatial) override {
    MUNET_DEBUG << "cross_entropy_backward | " << batch_size << " batches, "
                << num_classes << " classes" << std::endl;
    Timer t;
    base_->cross_entropy_backward(grad_out, logits, targets, grad_in,
                                  batch_size, num_classes, spatial);
    check("cross_entropy_backward", t.elapsed_us(), &grad_in);
  }
  void mse_loss(const Storage &pred, const Storage &target, Storage &out_loss,
                size_t num_elements) override {
    MUNET_DEBUG << "mse_loss | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->mse_loss(pred, target, out_loss, num_elements);
    check("mse_loss", t.elapsed_us(), &out_loss);
  }
  void mse_loss_backward(const Storage &grad_out, const Storage &pred,
                         const Storage &target, Storage &grad_in,
                         size_t num_elements) override {
    MUNET_DEBUG << "mse_loss_backward | " << num_elements << " elements"
                << std::endl;
    Timer t;
    base_->mse_loss_backward(grad_out, pred, target, grad_in, num_elements);
    check("mse_loss_backward", t.elapsed_us(), &grad_in);
  }
  void conv2d(const Storage &in, const Storage &weight, const Storage *bias,
              Storage &out, int B, int iC, int iH, int iW, int oC, int kH,
              int kW, int s, int p) override {
    MUNET_DEBUG << "conv2d | " << B << " batches, " << iC << " input channels, "
                << iH << " input height, " << iW << " input width, " << oC
                << " output channels, " << kH << " kernel height, " << kW
                << " kernel width, " << s << " stride, " << p << " padding"
                << std::endl;
    Timer t;
    base_->conv2d(in, weight, bias, out, B, iC, iH, iW, oC, kH, kW, s, p);
    check("conv2d", t.elapsed_us(), &out);
  }
  void conv2d_backward(const Storage &grad_out, const Storage &in,
                       const Storage &weight, Storage &grad_in, Storage &grad_w,
                       Storage *grad_b, int B, int iC, int iH, int iW, int oC,
                       int kH, int kW, int s, int p) override {
    MUNET_DEBUG << "conv2d_backward | " << B << " batches, " << iC
                << " input channels, " << iH << " input height, " << iW
                << " input width, " << oC << " output channels, " << kH
                << " kernel height, " << kW << " kernel width, " << s
                << " stride, " << p << " padding" << std::endl;
    Timer t;
    base_->conv2d_backward(grad_out, in, weight, grad_in, grad_w, grad_b, B, iC,
                           iH, iW, oC, kH, kW, s, p);
    check("conv2d_backward", t.elapsed_us(), &grad_in);
  }
  void max_pool2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int k, int s, int p) override {
    MUNET_DEBUG << "max_pool2d | " << B << " batches, " << C << " channels, "
                << iH << " input height, " << iW << " input width, " << k
                << " kernel size, " << s << " stride, " << p << " padding"
                << std::endl;
    Timer t;
    base_->max_pool2d(in, out, B, C, iH, iW, k, s, p);
    check("max_pool2d", t.elapsed_us(), &out);
  }
  void max_pool2d_backward(const Storage &grad_out, const Storage &in,
                           Storage &grad_in, int B, int C, int iH, int iW,
                           int k, int s, int p) override {
    MUNET_DEBUG << "max_pool2d_backward | " << B << " batches, " << C
                << " channels, " << iH << " input height, " << iW
                << " input width, " << k << " kernel size, " << s << " stride, "
                << p << " padding" << std::endl;
    Timer t;
    base_->max_pool2d_backward(grad_out, in, grad_in, B, C, iH, iW, k, s, p);
    check("max_pool2d_backward", t.elapsed_us(), &grad_in);
  }
  void upsample2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int scale) override {
    MUNET_DEBUG << "upsample2d | " << B << " batches, " << C << " channels, "
                << iH << " input height, " << iW << " input width, " << scale
                << " scale factor" << std::endl;
    Timer t;
    base_->upsample2d(in, out, B, C, iH, iW, scale);
    check("upsample2d", t.elapsed_us(), &out);
  }
  void upsample2d_backward(const Storage &grad_out, Storage &grad_in, int B,
                           int C, int iH, int iW, int scale) override {
    MUNET_DEBUG << "upsample2d_backward | " << B << " batches, " << C
                << " channels, " << iH << " input height, " << iW
                << " input width, " << scale << " scale factor" << std::endl;
    Timer t;
    base_->upsample2d_backward(grad_out, grad_in, B, C, iH, iW, scale);
    check("upsample2d_backward", t.elapsed_us(), &grad_in);
  }
  void batch_norm(const Storage &in, const Storage &scale, const Storage &bias,
                  Storage &running_mean, Storage &running_var,
                  Storage &save_mean, Storage &save_var, Storage &out, int B,
                  int C, int H, int W, float momentum, float eps,
                  bool training) override {
    MUNET_DEBUG << "batch_norm | " << B << " batches, " << C << " channels, "
                << H << " height, " << W << " width" << std::endl;
    Timer t;
    base_->batch_norm(in, scale, bias, running_mean, running_var, save_mean,
                      save_var, out, B, C, H, W, momentum, eps, training);
    check("batch_norm", t.elapsed_us(), &out);
  }
  void batch_norm_backward(const Storage &grad_out, const Storage &in,
                           const Storage &scale, const Storage &save_mean,
                           const Storage &save_var, Storage &grad_in,
                           Storage &grad_scale, Storage &grad_bias, int B,
                           int C, int H, int W, float eps) override {
    MUNET_DEBUG << "batch_norm_backward | " << B << " batches, " << C
                << " channels, " << H << " height, " << W << " width"
                << std::endl;
    Timer t;
    base_->batch_norm_backward(grad_out, in, scale, save_mean, save_var,
                               grad_in, grad_scale, grad_bias, B, C, H, W, eps);
    check("batch_norm_backward", t.elapsed_us(), &grad_in);
  }
  void update(Storage &weight, const Storage &grad, float lr,
              size_t num_elements) override {
    MUNET_DEBUG << "update | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->update(weight, grad, lr, num_elements);
    check("update", t.elapsed_us(), &weight);
  }
  void fill_uniform(Storage &out, float low, float high,
                    size_t num_elements) override {
    MUNET_DEBUG << "fill_uniform | " << num_elements << " elements"
                << std::endl;
    Timer t;
    base_->fill_uniform(out, low, high, num_elements);
    check("fill_uniform", t.elapsed_us(), &out);
  }
  void sum(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "sum | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->sum(in, out, num_elements);
    check("sum", t.elapsed_us(), &out);
  }

  void sum_to_shape(const Storage &in, Storage &out, const Shape &in_shape,
                    const Shape &out_shape) override {
    MUNET_DEBUG << "sum_to_shape | " << to_string(in_shape) << " -> "
                << to_string(out_shape) << std::endl;
    Timer t;
    base_->sum_to_shape(in, out, in_shape, out_shape);
    check("sum_to_shape", t.elapsed_us(), &out);
  }
  void mean_last_dim(const Storage &in, Storage &out, int outer_size,
                     int dim_size) override {
    MUNET_DEBUG << "mean_last_dim | outer=" << outer_size
                << " dim=" << dim_size << std::endl;
    Timer t;
    base_->mean_last_dim(in, out, outer_size, dim_size);
    check("mean_last_dim", t.elapsed_us(), &out);
  }
};

} // anonymous namespace

std::shared_ptr<Backend> wrap_with_debug_backend(std::shared_ptr<Backend> base) {
  return std::make_shared<DebugBackend>(std::move(base));
}

} // namespace munet
