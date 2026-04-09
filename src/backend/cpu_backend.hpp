#pragma once
#include "core/backend.hpp"
#include "core/all_reduce_runtime.hpp"
#include "core/util.hpp"
#include "storage.hpp"
#include "core/ops/common.hpp"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace munet {

class ThreadPool {
public:
  ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i)
      workers.emplace_back([this] {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(
                lock, [this] { return this->stop || !this->tasks.empty(); });
            if (this->stop && this->tasks.empty())
              return;
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          task();
        }
      });
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers)
      worker.join();
  }

  template <class F> auto enqueue(F &&f) -> std::future<void> {
    auto task =
        std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
    std::future<void> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");
      tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
  }

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

class CPUBackend : public Backend,
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
private:
  // Caching
  std::unordered_map<size_t, std::vector<void *>> free_blocks_;
  std::unordered_map<void *, size_t> alloc_sizes_;
  std::mutex mem_mutex_;

  double last_kernel_time_us_ = 0.0;

  static ThreadPool &get_pool() {
    static ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
  }

  template <typename Func>
  void parallel_for(size_t start, size_t end, Func func) {
    Timer t;
    size_t len = end - start;
    static const size_t num_threads = std::thread::hardware_concurrency();

    if (len < 1024 || num_threads <= 1) {
      func(start, end);
      return;
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);
    size_t block_size = (len + num_threads - 1) / num_threads;

    for (size_t i = 0; i < num_threads; ++i) {
      size_t s = start + i * block_size;
      if (s >= end)
        break;
      size_t e = std::min(s + block_size, end);
      futures.push_back(get_pool().enqueue([func, s, e]() { func(s, e); }));
    }
    for (auto &f : futures)
      f.wait();

    last_kernel_time_us_ = t.elapsed_us();
  }

public:
  const char *name() const override { return "cpu"; }

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
  BackendReductionCapability *reduction_capability() override { return this; }
  const BackendReductionCapability *reduction_capability() const override {
    return this;
  }
  BackendBlasCapability *blas_capability() override { return this; }
  const BackendBlasCapability *blas_capability() const override { return this; }
  BackendShapeCapability *shape_capability() override { return this; }
  const BackendShapeCapability *shape_capability() const override {
    return this;
  }
  BackendLossCapability *loss_capability() override { return this; }
  const BackendLossCapability *loss_capability() const override { return this; }
  BackendSpatialCapability *spatial_capability() override { return this; }
  const BackendSpatialCapability *spatial_capability() const override {
    return this;
  }
  BackendNormalizationCapability *normalization_capability() override {
    return this;
  }
  const BackendNormalizationCapability *
  normalization_capability() const override {
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

  ~CPUBackend() override {
    for (auto &kv : free_blocks_) {
      for (void *ptr : kv.second) {
        std::free(ptr);
      }
    }
  }

  double get_last_kernel_time_us() override { return last_kernel_time_us_; }

  void *allocate(size_t bytes) override {
    std::lock_guard<std::mutex> lock(mem_mutex_);
    if (!free_blocks_[bytes].empty()) {
      void *ptr = free_blocks_[bytes].back();
      free_blocks_[bytes].pop_back();
      return ptr;
    }
    void *ptr = std::malloc(bytes);
    if (!ptr)
      throw std::bad_alloc();
    alloc_sizes_[ptr] = bytes;
    return ptr;
  }

  void deallocate(void *ptr) override {
    std::lock_guard<std::mutex> lock(mem_mutex_);
    if (alloc_sizes_.count(ptr)) {
      free_blocks_[alloc_sizes_[ptr]].push_back(ptr);
    } else {
      std::free(ptr);
    }
  }

  void memset(void *ptr, int value, size_t bytes) override {
    std::memset(ptr, value, bytes);
  }
  void copy(const void *src, void *dst, size_t bytes, Device src_dev,
            Device dst_dev) override {
    if (bytes > 0 && (src == nullptr || dst == nullptr)) {
      throw std::runtime_error("cpu copy: null pointer with non-zero byte count");
    }
    const auto unsupported_endpoint = [](DeviceType type) {
      return type == DeviceType::CUDA || type == DeviceType::VULKAN;
    };
    if (unsupported_endpoint(src_dev.type) || unsupported_endpoint(dst_dev.type)) {
      throw std::runtime_error(
          "cpu copy: CPU backend cannot service CUDA/Vulkan transfer endpoints");
    }
    if ((src_dev.type == DeviceType::CPU && src_dev.index != 0) ||
        (dst_dev.type == DeviceType::CPU && dst_dev.index != 0)) {
      throw std::runtime_error(
          "cpu copy: CPU endpoint device index must be 0");
    }
    std::memcpy(dst, src, bytes);
  }
  void synchronize() override {}
  void all_reduce(Storage &buffer, size_t num_elements) override {
    detail::all_reduce_via_host(buffer, num_elements, *this, buffer.device());
  }

  void add(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    const float *ap = (const float *)a.data();
    const float *bp = (const float *)b.data();
    float *op = (float *)out.data();
    size_t total = numel(info.out_shape);
    int ndim = (int)info.out_shape.size();

    // Fast Path: Identical shapes and contiguous
    if (info.strides_a == default_strides(info.out_shape) &&
        info.strides_b == default_strides(info.out_shape)) {
      parallel_for(0, total, [&](size_t s, size_t e) {
        for (size_t i = s; i < e; ++i)
          op[i] = ap[i] + bp[i];
      });
      return;
    }

    // General Broadcast Path
    parallel_for(0, total, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i) {
        size_t off_a = 0, off_b = 0, curr = i;
        for (int d = ndim - 1; d >= 0; --d) {
          size_t coord = curr % info.out_shape[d];
          off_a += coord * info.strides_a[d];
          off_b += coord * info.strides_b[d];
          curr /= info.out_shape[d];
        }
        op[i] = ap[off_a] + bp[off_b];
      }
    });
  }

  void sub(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    const float *ap = (const float *)a.data(), *bp = (const float *)b.data();
    float *op = (float *)out.data();
    size_t total = numel(info.out_shape);
    int ndim = (int)info.out_shape.size();

    parallel_for(0, total, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i) {
        size_t off_a = 0, off_b = 0, curr = i;
        for (int d = ndim - 1; d >= 0; --d) {
          size_t coord = curr % info.out_shape[d];
          off_a += coord * info.strides_a[d];
          off_b += coord * info.strides_b[d];
          curr /= info.out_shape[d];
        }
        op[i] = ap[off_a] - bp[off_b];
      }
    });
  }

  void mul(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    const float *ap = (const float *)a.data(), *bp = (const float *)b.data();
    float *op = (float *)out.data();
    size_t total = numel(info.out_shape);
    int ndim = (int)info.out_shape.size();

    parallel_for(0, total, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i) {
        size_t off_a = 0, off_b = 0, curr = i;
        for (int d = ndim - 1; d >= 0; --d) {
          size_t coord = curr % info.out_shape[d];
          off_a += coord * info.strides_a[d];
          off_b += coord * info.strides_b[d];
          curr /= info.out_shape[d];
        }
        op[i] = ap[off_a] * bp[off_b];
      }
    });
  }

  void div(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    const float *ap = (const float *)a.data(), *bp = (const float *)b.data();
    float *op = (float *)out.data();
    size_t total = numel(info.out_shape);
    int ndim = (int)info.out_shape.size();

    parallel_for(0, total, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i) {
        size_t off_a = 0, off_b = 0, curr = i;
        for (int d = ndim - 1; d >= 0; --d) {
          size_t coord = curr % info.out_shape[d];
          off_a += coord * info.strides_a[d];
          off_b += coord * info.strides_b[d];
          curr /= info.out_shape[d];
        }
        op[i] = ap[off_a] / bp[off_b];
      }
    });
  }

  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) override {
    if (a.dtype() == DataType::Float16) {
      const uint16_t *ap = (const uint16_t *)a.data();
      const uint16_t *bp = (const uint16_t *)b.data();
      uint16_t *cp = (uint16_t *)out.data();
      parallel_for(0, M, [&](size_t start_m, size_t end_m) {
        for (int m = start_m; m < end_m; ++m) {
          for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
              float a_val = transA ? half_bits_to_float(ap[k * M + m])
                                   : half_bits_to_float(ap[m * K + k]);
              float b_val = transB ? half_bits_to_float(bp[n * K + k])
                                   : half_bits_to_float(bp[k * N + n]);
              sum += a_val * b_val;
            }
            cp[m * N + n] = float_to_half_bits(sum);
          }
        }
      });
      return;
    }

    const float *ap = (const float *)a.data();
    const float *bp = (const float *)b.data();
    float *cp = (float *)out.data();
    parallel_for(0, M, [&](size_t start_m, size_t end_m) {
      for (int m = start_m; m < end_m; ++m) {
        for (int n = 0; n < N; ++n) {
          float sum = 0.0f;
          for (int k = 0; k < K; ++k) {
            float a_val = transA ? ap[k * M + m] : ap[m * K + k];
            float b_val = transB ? bp[n * K + k] : bp[k * N + n];
            sum += a_val * b_val;
          }
          cp[m * N + n] = sum;
        }
      }
    });
  }

  void batched_matmul(const Storage &a, const Storage &b, Storage &out,
                      int batch_size, int M, int K, int N, bool transA, bool transB,
                      int64_t stride_a, int64_t stride_b, int64_t stride_out) override {
    if (a.dtype() == DataType::Float16) {
      const uint16_t *ap = (const uint16_t *)a.data();
      const uint16_t *bp = (const uint16_t *)b.data();
      uint16_t *cp = (uint16_t *)out.data();
      for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
        const uint16_t *a_batch = ap + b_idx * stride_a;
        const uint16_t *b_batch = bp + b_idx * stride_b;
        uint16_t *out_batch = cp + b_idx * stride_out;
        for (int m = 0; m < M; ++m) {
          for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
              const float a_val =
                  transA ? half_bits_to_float(a_batch[k * M + m])
                         : half_bits_to_float(a_batch[m * K + k]);
              const float b_val =
                  transB ? half_bits_to_float(b_batch[n * K + k])
                         : half_bits_to_float(b_batch[k * N + n]);
              sum += a_val * b_val;
            }
            out_batch[m * N + n] = float_to_half_bits(sum);
          }
        }
      }
      return;
    }

    const float *ap = (const float *)a.data();
    const float *bp = (const float *)b.data();
    float *cp = (float *)out.data();
    for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
      ops::detail::batched_matmul_cpu_fallback(
          ap + b_idx * stride_a, bp + b_idx * stride_b, cp + b_idx * stride_out,
          M, K, N, transA, transB);
    }
  }

  void concat(const std::vector<Storage *> &inputs, Storage &out, int dim,
              const std::vector<Shape> &shapes) override {
    float *out_ptr = (float *)out.data();
    int outer_size = 1;
    // Calculate the outer dimensions up to the concatenation axis.
    for (int i = 0; i < dim; ++i) {
      outer_size *= shapes[0][i];
    }

    int inner_size = 1;
    // Calculate the inner dimensions after the concatenation axis.
    for (int i = dim + 1; i < shapes[0].size(); ++i) {
      inner_size *= shapes[0][i];
    }

    int out_dim_size = 0;
    // Calculate the total size of the concatenation dimension.
    for (const auto &s : shapes) {
      out_dim_size += s[dim];
    }

    // Perform the concatenation operation across all input tensors.
    for (int i = 0; i < outer_size; ++i) {
      int out_offset = i * out_dim_size * inner_size; // Set the output offset.
      for (size_t j = 0; j < inputs.size(); ++j) {
        int dim_size =
            shapes[j][dim]; // Get the size of the current input tensor along
                            // the concatenation dimension.
        int copy_bytes =
            dim_size * inner_size * sizeof(float); // Number of bytes to copy.
        std::memcpy(out_ptr + out_offset,
                    (float *)inputs[j]->data() + i * dim_size * inner_size,
                    copy_bytes);             // Copy from input to output.
        out_offset += dim_size * inner_size; // Increment the output offset.
      }
    }
  }

  void concat_backward(const Storage &grad_out,
                       std::vector<Storage *> &grad_inputs, int dim,
                       const std::vector<Shape> &shapes) override {
    float *go_ptr = (float *)grad_out.data();
    int outer_size = 1;
    // Calculate the outer dimensions up to the concatenation axis.
    for (int i = 0; i < dim; ++i) {
      outer_size *= shapes[0][i];
    }

    int inner_size = 1;
    // Calculate the inner dimensions after the concatenation axis.
    for (int i = dim + 1; i < shapes[0].size(); ++i) {
      inner_size *= shapes[0][i];
    }

    int out_dim_size = 0;
    // Calculate the total size of the concatenation dimension.
    for (const auto &s : shapes) {
      out_dim_size += s[dim];
    }

    // Perform the backward gradient propagation operation across all input
    // tensors.
    for (int i = 0; i < outer_size; ++i) {
      int out_offset = i * out_dim_size * inner_size; // Set the output offset.
      for (size_t j = 0; j < grad_inputs.size(); ++j) {
        int dim_size =
            shapes[j][dim]; // Get the size of the current input tensor along
                            // the concatenation dimension.
        int copy_bytes =
            dim_size * inner_size * sizeof(float); // Number of bytes to copy.
        std::memcpy((float *)grad_inputs[j]->data() + i * dim_size * inner_size,
                    go_ptr + out_offset, copy_bytes); // Copy gradient to input.
        out_offset += dim_size * inner_size; // Increment the output offset.
      }
    }
  }

  void relu(const Storage &in, Storage &out, size_t num_elements) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        op[i] = ip[i] > 0 ? ip[i] : 0;
    });
  }

  void relu_backward(const Storage &grad_out, const Storage &input,
                     Storage &grad_in, size_t num_elements) override {
    const float *go = (const float *)grad_out.data();
    const float *in = (const float *)input.data();
    float *gi = (float *)grad_in.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        gi[i] = (in[i] > 0) ? go[i] : 0.0f;
    });
  }

  void sigmoid(const Storage &in, Storage &out, size_t num_elements) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        op[i] = 1.0f / (1.0f + std::exp(-ip[i]));
    });
  }

  void sigmoid_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, size_t num_elements) override {
    const float *go = (const float *)grad_out.data();
    const float *o = (const float *)out.data();
    float *gi = (float *)grad_in.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        gi[i] = go[i] * o[i] * (1.0f - o[i]);
    });
  }

  void exp(const Storage &in, Storage &out, size_t num_elements) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        op[i] = std::exp(ip[i]);
    });
  }

  void log(const Storage &in, Storage &out, size_t num_elements) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        op[i] = std::log(ip[i]);
    });
  }

  void sqrt(const Storage &in, Storage &out, size_t num_elements) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        op[i] = std::sqrt(ip[i]);
    });
  }

  void rsqrt(const Storage &in, Storage &out, size_t num_elements) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        op[i] = 1.0f / std::sqrt(ip[i]);
    });
  }

  void sin(const Storage &in, Storage &out, size_t num_elements) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        op[i] = std::sin(ip[i]);
    });
  }

  void cos(const Storage &in, Storage &out, size_t num_elements) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        op[i] = std::cos(ip[i]);
    });
  }

  void softmax(const Storage &in, Storage &out, int batch_size,
               int num_classes) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();
    parallel_for(0, batch_size, [&](size_t s, size_t e) {
      for (size_t b = s; b < e; ++b) {
        const float *in_row = ip + b * num_classes;
        float *out_row = op + b * num_classes;
        float max_val = in_row[0];
        for (int i = 1; i < num_classes; ++i)
          if (in_row[i] > max_val)
            max_val = in_row[i];

        // Use double for higher precision accumulation
        double sum_exp = 0.0;
        for (int i = 0; i < num_classes; ++i) {
          sum_exp += std::exp((double)in_row[i] - (double)max_val);
        }
        for (int i = 0; i < num_classes; ++i) {
          out_row[i] =
              (float)(std::exp((double)in_row[i] - (double)max_val) / sum_exp);
        }
      }
    });
  }

  void log_softmax(const Storage &in, Storage &out, int batch_size,
                   int num_classes) override {
    softmax(in, out, batch_size, num_classes);
    log(out, out, static_cast<size_t>(batch_size) * num_classes);
  }

  void softmax_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, int batch_size,
                        int num_classes) override {
    const float *go = (const float *)grad_out.data();
    const float *o = (const float *)out.data();
    float *gi = (float *)grad_in.data();
    parallel_for(0, batch_size, [&](size_t s, size_t e) {
      for (size_t b = s; b < e; ++b) {
        const float *go_row = go + b * num_classes;
        const float *out_row = o + b * num_classes;
        float *gi_row = gi + b * num_classes;

        double sum_out_go = 0.0;
        for (int i = 0; i < num_classes; ++i)
          sum_out_go += (double)out_row[i] * (double)go_row[i];

        for (int i = 0; i < num_classes; ++i)
          gi_row[i] =
              (float)((double)out_row[i] * ((double)go_row[i] - sum_out_go));
      }
    });
  }

  void mse_loss(const Storage &pred, const Storage &target, Storage &out_loss,
                size_t num_elements) override {
    const float *p = (const float *)pred.data();
    const float *t = (const float *)target.data();
    float *out = (float *)out_loss.data();

    // Sequential reduction for simplicity and thread safety
    float sum = 0.0f;
    for (size_t i = 0; i < num_elements; ++i) {
      float diff = p[i] - t[i];
      sum += diff * diff;
    }
    out[0] = sum / (float)num_elements;
  }

  void mse_loss_backward(const Storage &grad_out, const Storage &pred,
                         const Storage &target, Storage &grad_in,
                         size_t num_elements) override {
    const float *go = (const float *)grad_out.data();
    const float *p = (const float *)pred.data();
    const float *t = (const float *)target.data();
    float *gi = (float *)grad_in.data();

    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      float go_val = go[0]; // Loss is scalar
      float scale = 2.0f / (float)num_elements;
      for (size_t i = s; i < e; ++i) {
        gi[i] = go_val * scale * (p[i] - t[i]);
      }
    });
  }

  void cross_entropy(const Storage &logits, const Storage &targets,
                     Storage &out_loss, int batch_size, int num_classes,
                     int spatial) override {
    const float *l = (const float *)logits.data();
    const float *t = (const float *)targets.data();
    float *out = (float *)out_loss.data();

    // Sum reduction requires lock or atomic. We use a thread-local accumulation
    // strategy via sequential loop for simplicity in this fallback backend.
    double total_loss = 0.0;

    // Iterate over N samples
    for (int b = 0; b < batch_size; ++b) {
      // Iterate over spatial locations (pixels)
      for (int s = 0; s < spatial; ++s) {
        // Find Max for stability (over classes)
        float max_val = -1e30f;
        for (int c = 0; c < num_classes; ++c) {
          // NCHW offset: b * (C*S) + c * S + s
          int idx = b * (num_classes * spatial) + c * spatial + s;
          if (l[idx] > max_val)
            max_val = l[idx];
        }

        // Sum Exp
        double sum_exp = 0.0;
        for (int c = 0; c < num_classes; ++c) {
          int idx = b * (num_classes * spatial) + c * spatial + s;
          sum_exp += std::exp((double)l[idx] - (double)max_val);
        }

        // Compute Loss
        for (int c = 0; c < num_classes; ++c) {
          int idx = b * (num_classes * spatial) + c * spatial + s;
          float prob =
              (float)(std::exp((double)l[idx] - (double)max_val) / sum_exp);
          float tgt = t[idx];
          if (tgt > 0.0f) {
            total_loss -= tgt * std::log(prob + 1e-9f);
          }
        }
      }
    }
    // Mean over Batch (standard definition)
    out[0] = (float)(total_loss / (double)batch_size);
  }

  void cross_entropy_backward(const Storage &grad_out, const Storage &logits,
                              const Storage &targets, Storage &grad_in,
                              int batch_size, int num_classes,
                              int spatial) override {
    const float *go = (const float *)grad_out.data();
    const float *l = (const float *)logits.data();
    const float *t = (const float *)targets.data();
    float *gi = (float *)grad_in.data();

    float go_val = go[0];

    parallel_for(0, batch_size * spatial, [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        int b = i / spatial;
        int s = i % spatial;

        // Find Max
        float max_val = -1e30f;
        for (int c = 0; c < num_classes; ++c) {
          int idx = b * (num_classes * spatial) + c * spatial + s;
          if (l[idx] > max_val)
            max_val = l[idx];
        }

        // Sum Exp
        double sum_exp = 0.0;
        for (int c = 0; c < num_classes; ++c) {
          int idx = b * (num_classes * spatial) + c * spatial + s;
          sum_exp += std::exp((double)l[idx] - (double)max_val);
        }

        // Gradient
        for (int c = 0; c < num_classes; ++c) {
          int idx = b * (num_classes * spatial) + c * spatial + s;
          float prob =
              (float)(std::exp((double)l[idx] - (double)max_val) / sum_exp);
          // Gradient is (p - t) / N
          gi[idx] = go_val * (prob - t[idx]) / (float)batch_size;
        }
      }
    });
  }

  void update(Storage &weight, const Storage &grad, float lr,
              size_t num_elements) override {
    float *w = (float *)weight.data();
    const float *g = (const float *)grad.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        w[i] -= lr * g[i];
    });
  }

  void conv2d(const Storage &in, const Storage &weight, const Storage *bias,
              Storage &out, int B, int iC, int iH, int iW, int oC, int kH,
              int kW, int s, int p) override {
    int oH = (iH + 2 * p - kH) / s + 1;
    int oW = (iW + 2 * p - kW) / s + 1;
    const float *in_p = (const float *)in.data(),
                *w_p = (const float *)weight.data();
    const float *b_p = bias ? (const float *)bias->data() : nullptr;
    float *out_p = (float *)out.data();
    parallel_for(0, B * oC * oH * oW, [&](size_t start, size_t end) {
      for (size_t idx = start; idx < end; ++idx) {
        int ow = idx % oW, oh = (idx / oW) % oH, oc = (idx / (oW * oH)) % oC,
            b = idx / (oW * oH * oC);
        float val = b_p ? b_p[oc] : 0.0f;
        for (int ic = 0; ic < iC; ++ic) {
          for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
              int ih = oh * s - p + kh, iw = ow * s - p + kw;
              if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                val += in_p[((b * iC + ic) * iH + ih) * iW + iw] *
                       w_p[((oc * iC + ic) * kH + kh) * kW + kw];
              }
            }
          }
        }
        out_p[idx] = val;
      }
    });
  }
  void conv2d_backward(const Storage &grad_out, const Storage &in,
                       const Storage &weight, Storage &grad_in, Storage &grad_w,
                       Storage *grad_b, int B, int iC, int iH, int iW, int oC,
                       int kH, int kW, int s, int p) override {
    int oH = (iH + 2 * p - kH) / s + 1, oW = (iW + 2 * p - kW) / s + 1;
    const float *go_p = (const float *)grad_out.data(),
                *in_p = (const float *)in.data(),
                *w_p = (const float *)weight.data();
    float *gi_p = (float *)grad_in.data(), *gw_p = (float *)grad_w.data();
    float *gb_p = grad_b ? (float *)grad_b->data() : nullptr;
    // Single threaded accumulation to avoid atomic locks for now
    for (int b = 0; b < B; ++b) {
      for (int oc = 0; oc < oC; ++oc) {
        for (int oh = 0; oh < oH; ++oh) {
          for (int ow = 0; ow < oW; ++ow) {
            float go_val = go_p[((b * oC + oc) * oH + oh) * oW + ow];
            if (gb_p)
              gb_p[oc] += go_val;
            for (int ic = 0; ic < iC; ++ic) {
              for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                  int ih = oh * s - p + kh, iw = ow * s - p + kw;
                  if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                    gi_p[((b * iC + ic) * iH + ih) * iW + iw] +=
                        go_val * w_p[((oc * iC + ic) * kH + kh) * kW + kw];
                    gw_p[((oc * iC + ic) * kH + kh) * kW + kw] +=
                        go_val * in_p[((b * iC + ic) * iH + ih) * iW + iw];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  void max_pool2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int k, int s, int p) override {
    int oH = (iH + 2 * p - k) / s + 1, oW = (iW + 2 * p - k) / s + 1;
    const float *in_p = (const float *)in.data();
    float *out_p = (float *)out.data();
    parallel_for(0, B * C * oH * oW, [&](size_t start, size_t end) {
      for (size_t idx = start; idx < end; ++idx) {
        int ow = idx % oW, oh = (idx / oW) % oH, c = (idx / (oW * oH)) % C,
            b = idx / (oW * oH * C);
        float max_val = -1e9f;
        for (int kh = 0; kh < k; ++kh) {
          for (int kw = 0; kw < k; ++kw) {
            int ih = oh * s - p + kh, iw = ow * s - p + kw;
            if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
              max_val =
                  std::max(max_val, in_p[((b * C + c) * iH + ih) * iW + iw]);
            }
          }
        }
        out_p[idx] = max_val;
      }
    });
  }
  void max_pool2d_backward(const Storage &grad_out, const Storage &in,
                           Storage &grad_in, int B, int C, int iH, int iW,
                           int k, int s, int p) override {
    int oH = (iH + 2 * p - k) / s + 1, oW = (iW + 2 * p - k) / s + 1;
    const float *go_p = (const float *)grad_out.data(),
                *in_p = (const float *)in.data();
    float *gi_p = (float *)grad_in.data();
    // Recompute max to route gradient
    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < oH; ++oh) {
          for (int ow = 0; ow < oW; ++ow) {
            float max_val = -1e9f;
            int max_idx = -1;
            for (int kh = 0; kh < k; ++kh) {
              for (int kw = 0; kw < k; ++kw) {
                int ih = oh * s - p + kh, iw = ow * s - p + kw;
                if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                  int idx = ((b * C + c) * iH + ih) * iW + iw;
                  if (*in_p > max_val) {
                    max_val = *in_p;
                    max_idx = idx;
                  }
                }
              }
            }
            if (max_idx != -1)
              gi_p[max_idx] += go_p[((b * C + c) * oH + oh) * oW + ow];
          }
        }
      }
    }
  }
  void upsample2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int scale) override {
    int oH = iH * scale, oW = iW * scale;
    const float *in_p = (const float *)in.data();
    float *out_p = (float *)out.data();
    parallel_for(0, B * C * oH * oW, [&](size_t start, size_t end) {
      for (size_t idx = start; idx < end; ++idx) {
        int ow = idx % oW, oh = (idx / oW) % oH, c = (idx / (oW * oH)) % C,
            b = idx / (oW * oH * C);
        out_p[idx] =
            in_p[((b * C + c) * iH + (oh / scale)) * iW + (ow / scale)];
      }
    });
  }
  void upsample2d_backward(const Storage &grad_out, Storage &grad_in, int B,
                           int C, int iH, int iW, int scale) override {
    int oH = iH * scale, oW = iW * scale;
    const float *go_p = (const float *)grad_out.data();
    float *gi_p = (float *)grad_in.data();
    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < oH; ++oh) {
          for (int ow = 0; ow < oW; ++ow) {
            gi_p[((b * C + c) * iH + (oh / scale)) * iW + (ow / scale)] +=
                go_p[((b * C + c) * oH + oh) * oW + ow];
          }
        }
      }
    }
  }

  void batch_norm(const Storage &in, const Storage &scale, const Storage &bias,
                  Storage &running_mean, Storage &running_var,
                  Storage &save_mean, Storage &save_var, Storage &out, int B,
                  int C, int H, int W, float momentum, float eps,
                  bool training) override {
    const float *x = (const float *)in.data();
    const float *g = (const float *)scale.data();
    const float *b = (const float *)bias.data();
    float *rm = (float *)running_mean.data();
    float *rv = (float *)running_var.data();
    float *sm = (float *)save_mean.data();
    float *sv = (float *)save_var.data();
    float *y = (float *)out.data();
    size_t spatial = H * W;

    if (training) {
      // 1. Calculate Mean/Var per channel
      for (int c = 0; c < C; ++c) {
        float sum = 0, sq_sum = 0;
        for (int batch = 0; batch < B; ++batch) {
          for (int s = 0; s < spatial; ++s) {
            float val = x[(batch * C + c) * spatial + s];
            sum += val;
            sq_sum += val * val;
          }
        }
        float n = B * spatial;
        float mu = sum / n;
        float var = (sq_sum / n) - (mu * mu);
        sm[c] = mu;
        sv[c] = var; // save variance
        // Update running
        rm[c] = (1 - momentum) * rm[c] + momentum * mu;
        rv[c] = (1 - momentum) * rv[c] + momentum * var;
      }
    }

    // 2. Normalize
    parallel_for(0, B * C * spatial, [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        int tmp = i / spatial;
        int c = tmp % C;
        float mu = training ? sm[c] : rm[c];
        float var = training ? sv[c] : rv[c];
        float inv_std = 1.0f / std::sqrt(var + eps);
        y[i] = g[c] * (x[i] - mu) * inv_std + b[c];
      }
    });
  }

  void batch_norm_backward(const Storage &grad_out, const Storage &in,
                           const Storage &scale, const Storage &save_mean,
                           const Storage &save_var, Storage &grad_in,
                           Storage &grad_scale, Storage &grad_bias, int B,
                           int C, int H, int W, float eps) override {
    const float *dy = (const float *)grad_out.data();
    const float *x = (const float *)in.data();
    const float *gamma = (const float *)scale.data();
    const float *mu = (const float *)save_mean.data();
    const float *var = (const float *)save_var.data();
    float *dx = (float *)grad_in.data();
    float *dg = (float *)grad_scale.data();
    float *db = (float *)grad_bias.data();

    size_t spatial = H * W;
    float m = B * spatial;

    // Zero out gradients first (since we might accumulate if reusing buffers,
    // though here we write directly)
    std::memset(dg, 0, C * sizeof(float));
    std::memset(db, 0, C * sizeof(float));

    // 1. Compute dGamma, dBeta
    for (int c = 0; c < C; ++c) {
      float sum_dy = 0;
      float sum_dy_xhat = 0;
      float inv_std = 1.0f / std::sqrt(var[c] + eps);

      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < spatial; ++s) {
          int idx = (b * C + c) * spatial + s;
          float x_hat = (x[idx] - mu[c]) * inv_std;
          sum_dy += dy[idx];
          sum_dy_xhat += dy[idx] * x_hat;
        }
      }
      dg[c] = sum_dy_xhat;
      db[c] = sum_dy;

      // 2. Compute dx (Simplified standard BN backward)
      float factor = gamma[c] * inv_std / m;
      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < spatial; ++s) {
          int idx = (b * C + c) * spatial + s;
          float x_hat = (x[idx] - mu[c]) * inv_std;
          dx[idx] = factor * (m * dy[idx] - sum_dy - x_hat * sum_dy_xhat);
        }
      }
    }
  }

  void fill_uniform(Storage &out, float low, float high,
                    size_t num_elements) override {
    if (!is_floating(out.dtype())) {
      throw std::runtime_error(
          "fill_uniform only supports floating-point tensors");
    }

    char *ptr = static_cast<char *>(out.data());
    const size_t element_size = dtype_size(out.dtype());
    // Not strictly thread-safe to share generator, creating local one
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      std::mt19937 gen(42 + s); // Seed offset by index
      std::uniform_real_distribution<float> dis(low, high);
      for (size_t i = s; i < e; ++i) {
        write_scalar_to_buffer(ptr + i * element_size, out.dtype(), dis(gen));
      }
    });
  }

  void sum(const Storage &in, Storage &out, size_t num_elements) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();

    // Simple sequential sum for now to avoid atomic overheads on CPU
    float total = 0.0f;
    for (size_t i = 0; i < num_elements; ++i)
      total += ip[i];
    op[0] = total;
  }

  void broadcast_row(const Storage &src, Storage &dst, int rows,
                     int cols) override {
    const float *sp = (const float *)src.data();
    float *dp = (float *)dst.data();
    parallel_for(0, rows, [&](size_t s, size_t e) {
      for (size_t r = s; r < e; ++r) {
        std::memcpy(dp + r * cols, sp, cols * sizeof(float));
      }
    });
  }

  void adam_step(Storage &params, const Storage &grads, Storage &exp_avg,
                 Storage &exp_avg_sq, float lr, float beta1, float beta2,
                 float eps, int step, size_t num_elements) override {
    float *p = (float *)params.data();
    const float *g = (const float *)grads.data();
    float *m = (float *)exp_avg.data();
    float *v = (float *)exp_avg_sq.data();

    float bias_correction1 = 1.0f - std::pow(beta1, step);
    float bias_correction2 = 1.0f - std::pow(beta2, step);
    float step_size = lr / bias_correction1;

    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * (g[i] * g[i]);
        float denom = (std::sqrt(v[i]) / std::sqrt(bias_correction2)) + eps;
        p[i] -= step_size * m[i] / denom;
      }
    });
  }

  void sum_to_shape(const Storage &in, Storage &out, const Shape &in_shape,
                    const Shape &out_shape) override {

    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();

    // Zero output
    std::memset(op, 0, out.size_bytes());

    int ndim = (int)in_shape.size();
    int out_ndim = (int)out_shape.size();

    Strides in_strides = default_strides(in_shape);
    Strides out_strides = default_strides(out_shape);

    size_t total = numel(in_shape);

    for (size_t i = 0; i < total; ++i) {

      size_t out_off = 0;
      size_t curr = i;

      for (int d = ndim - 1; d >= 0; --d) {

        size_t coord = curr % in_shape[d];
        curr /= in_shape[d];

        int out_d_idx = d - (ndim - out_ndim);

        if (out_d_idx >= 0) {
          if (out_shape[out_d_idx] != 1) {
            out_off += coord * out_strides[out_d_idx];
          }
        }
      }

      op[out_off] += ip[i];
    }
  }

  void mean_last_dim(const Storage &in, Storage &out, int outer_size,
                     int dim_size) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();
    parallel_for(0, static_cast<size_t>(outer_size), [&](size_t s, size_t e) {
      for (size_t row = s; row < e; ++row) {
        double total = 0.0;
        const size_t base = row * static_cast<size_t>(dim_size);
        for (int col = 0; col < dim_size; ++col)
          total += ip[base + static_cast<size_t>(col)];
        op[row] = static_cast<float>(total / static_cast<double>(dim_size));
      }
    });
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
};
} // namespace munet
