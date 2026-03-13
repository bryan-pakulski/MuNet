#pragma once
#include "backend.hpp"
#include "storage.hpp"
#include "util.hpp"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
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

class CPUBackend : public Backend {
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

  // ---- DType conversion helpers (Phase 1 scaffolding) ----
  struct ComputePlan {
    DataType compute_dtype = DataType::Float32;
    bool fallback_applied = false;
  };

  static bool supports_compute_dtype(DataType dt) {
    switch (dt) {
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Float16:
    case DataType::BFloat16:
    case DataType::Int8:
    case DataType::Int4:
      return true;
    default:
      return false;
    }
  }

  static ComputePlan resolve_compute_plan(DataType preferred) {
    ComputePlan plan;
    DTypeDispatchConfig cfg = DTypeDispatch::current();
    DataType requested = cfg.has_compute_dtype ? cfg.compute_dtype : preferred;

    if (supports_compute_dtype(requested)) {
      plan.compute_dtype = requested;
      return plan;
    }

    if (cfg.fallback_mode == KernelFallbackMode::Error) {
      throw std::runtime_error("CPUBackend: requested compute dtype not supported: " +
                               std::string(dtype_name(requested)));
    }

    plan.compute_dtype = DataType::Float32;
    plan.fallback_applied = true;
    MUNET_WARNING << "CPUBackend: fallback compute dtype "
                  << dtype_name(requested)
                  << " -> float32 (warn_and_upcast mode)" << std::endl;
    return plan;
  }

  // Storage dtype may differ from compute dtype. For low precision and quantized
  // dtypes we currently dequantize/convert into the selected compute dtype.
  static double load_as_compute(const Storage &s, size_t idx) {
    return load_scalar_as_double(s.data(), s.dtype(), idx);
  }

  static void store_from_compute(Storage &s, size_t idx, double v) {
    store_scalar_from_double(s.data(), s.dtype(), idx, v);
  }

public:
  ~CPUBackend() override {
    for (auto &kv : free_blocks_) {
      for (void *ptr : kv.second) {
        std::free(ptr);
      }
    }
  }

  double get_last_kernel_time_us() override { return last_kernel_time_us_; }

  bool supports_non_finite_check() const override { return true; }

  bool has_non_finite(const Storage &s, size_t num_elements) const override {
    for (size_t i = 0; i < num_elements; ++i) {
      if (!std::isfinite(load_as_compute(s, i)))
        return true;
    }
    return false;
  }

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
    std::memcpy(dst, src, bytes);
  }
  void synchronize() override {}
  void all_reduce(Storage &buffer, size_t num_elements) override {}

  void add(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    (void)resolve_compute_plan(accumulation_dtype(a.dtype()));
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
        float v = load_as_compute(a, off_a) + load_as_compute(b, off_b);
        store_from_compute(out, i, v);
      }
    });
  }

  void sub(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    (void)resolve_compute_plan(accumulation_dtype(a.dtype()));
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
        double v = load_as_compute(a, off_a) - load_as_compute(b, off_b);
        store_from_compute(out, i, v);
      }
    });
  }

  void mul(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    (void)resolve_compute_plan(accumulation_dtype(a.dtype()));
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
        float v = load_as_compute(a, off_a) * load_as_compute(b, off_b);
        store_from_compute(out, i, v);
      }
    });
  }

  void div(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override {
    (void)resolve_compute_plan(accumulation_dtype(a.dtype()));
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
        double denom = load_as_compute(b, off_b);
        double v = load_as_compute(a, off_a) / denom;
        store_from_compute(out, i, v);
      }
    });
  }

  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) override {
    (void)resolve_compute_plan(accumulation_dtype(a.dtype()));
    parallel_for(0, M, [&](size_t start_m, size_t end_m) {
      for (int m = start_m; m < end_m; ++m) {
        for (int n = 0; n < N; ++n) {
          float acc = 0.0f;
          for (int k = 0; k < K; ++k) {
            size_t a_idx = transA ? (size_t)k * M + m : (size_t)m * K + k;
            size_t b_idx = transB ? (size_t)n * K + k : (size_t)k * N + n;
            acc += load_as_compute(a, a_idx) * load_as_compute(b, b_idx);
          }
          store_from_compute(out, (size_t)m * N + n, acc);
        }
      }
    });
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

  void softmax(const Storage &in, Storage &out, int batch_size,
               int num_classes) override {
    (void)resolve_compute_plan(accumulation_dtype(in.dtype()));
    parallel_for(0, batch_size, [&](size_t s, size_t e) {
      for (size_t b = s; b < e; ++b) {
        float max_val = load_as_compute(in, b * num_classes);
        for (int i = 1; i < num_classes; ++i) {
          float v = load_as_compute(in, b * num_classes + i);
          if (v > max_val)
            max_val = v;
        }

        double sum_exp = 0.0;
        for (int i = 0; i < num_classes; ++i) {
          sum_exp += std::exp((double)load_as_compute(in, b * num_classes + i) -
                              (double)max_val);
        }
        for (int i = 0; i < num_classes; ++i) {
          float o = (float)(std::exp((double)load_as_compute(in, b * num_classes + i) -
                                     (double)max_val) /
                            sum_exp);
          store_from_compute(out, b * num_classes + i, o);
        }
      }
    });
  }

  void softmax_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, int batch_size,
                        int num_classes) override {
    (void)resolve_compute_plan(accumulation_dtype(out.dtype()));
    parallel_for(0, batch_size, [&](size_t s, size_t e) {
      for (size_t b = s; b < e; ++b) {
        double sum_out_go = 0.0;
        for (int i = 0; i < num_classes; ++i) {
          double out_v = load_as_compute(out, b * num_classes + i);
          double go_v = load_as_compute(grad_out, b * num_classes + i);
          sum_out_go += out_v * go_v;
        }

        for (int i = 0; i < num_classes; ++i) {
          double out_v = load_as_compute(out, b * num_classes + i);
          double go_v = load_as_compute(grad_out, b * num_classes + i);
          double gi = out_v * (go_v - sum_out_go);
          store_from_compute(grad_in, b * num_classes + i, gi);
        }
      }
    });
  }

  void mse_loss(const Storage &pred, const Storage &target, Storage &out_loss,
                size_t num_elements) override {
    (void)resolve_compute_plan(accumulation_dtype(pred.dtype()));
    double sum = 0.0;
    for (size_t i = 0; i < num_elements; ++i) {
      float diff = load_as_compute(pred, i) - load_as_compute(target, i);
      sum += (double)diff * (double)diff;
    }
    store_from_compute(out_loss, 0, (float)(sum / (double)num_elements));
  }

  void mse_loss_backward(const Storage &grad_out, const Storage &pred,
                         const Storage &target, Storage &grad_in,
                         size_t num_elements) override {
    (void)resolve_compute_plan(accumulation_dtype(pred.dtype()));
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      double go_val = load_as_compute(grad_out, 0); // Loss is scalar
      double scale = 2.0 / (double)num_elements;
      for (size_t i = s; i < e; ++i) {
        double gi = go_val * scale *
                    (load_as_compute(pred, i) - load_as_compute(target, i));
        store_from_compute(grad_in, i, gi);
      }
    });
  }

  void cross_entropy(const Storage &logits, const Storage &targets,
                     Storage &out_loss, int batch_size, int num_classes,
                     int spatial) override {
    (void)resolve_compute_plan(accumulation_dtype(logits.dtype()));
    double total_loss = 0.0;

    for (int b = 0; b < batch_size; ++b) {
      for (int s = 0; s < spatial; ++s) {
        float max_val = -1e30f;
        for (int c = 0; c < num_classes; ++c) {
          int idx = b * (num_classes * spatial) + c * spatial + s;
          float lv = load_as_compute(logits, idx);
          if (lv > max_val)
            max_val = lv;
        }

        double sum_exp = 0.0;
        for (int c = 0; c < num_classes; ++c) {
          int idx = b * (num_classes * spatial) + c * spatial + s;
          sum_exp += std::exp((double)load_as_compute(logits, idx) -
                              (double)max_val);
        }

        for (int c = 0; c < num_classes; ++c) {
          int idx = b * (num_classes * spatial) + c * spatial + s;
          float prob = (float)(std::exp((double)load_as_compute(logits, idx) -
                                        (double)max_val) /
                               sum_exp);
          float tgt = load_as_compute(targets, idx);
          if (tgt > 0.0f) {
            total_loss -= tgt * std::log(prob + 1e-9f);
          }
        }
      }
    }
    store_from_compute(out_loss, 0, (float)(total_loss / (double)batch_size));
  }

  void cross_entropy_backward(const Storage &grad_out, const Storage &logits,
                              const Storage &targets, Storage &grad_in,
                              int batch_size, int num_classes,
                              int spatial) override {
    (void)resolve_compute_plan(accumulation_dtype(logits.dtype()));
    double go_val = load_as_compute(grad_out, 0);

    parallel_for(0, batch_size * spatial, [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        int b = i / spatial;
        int s = i % spatial;

        double max_val = -1e30;
        for (int c = 0; c < num_classes; ++c) {
          int idx = b * (num_classes * spatial) + c * spatial + s;
          max_val = std::max(max_val, load_as_compute(logits, idx));
        }

        double sum_exp = 0.0;
        for (int c = 0; c < num_classes; ++c) {
          int idx = b * (num_classes * spatial) + c * spatial + s;
          sum_exp += std::exp(load_as_compute(logits, idx) - max_val);
        }

        for (int c = 0; c < num_classes; ++c) {
          int idx = b * (num_classes * spatial) + c * spatial + s;
          double prob = std::exp(load_as_compute(logits, idx) - max_val) / sum_exp;
          double tgt = load_as_compute(targets, idx);
          double gi = go_val * (prob - tgt) / (double)batch_size;
          store_from_compute(grad_in, idx, gi);
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
    (void)resolve_compute_plan(accumulation_dtype(in.dtype()));
    int oH = (iH + 2 * p - kH) / s + 1;
    int oW = (iW + 2 * p - kW) / s + 1;
    parallel_for(0, B * oC * oH * oW, [&](size_t start, size_t end) {
      for (size_t idx = start; idx < end; ++idx) {
        int ow = idx % oW, oh = (idx / oW) % oH, oc = (idx / (oW * oH)) % oC,
            b = idx / (oW * oH * oC);
        double val = bias ? load_as_compute(*bias, oc) : 0.0;
        for (int ic = 0; ic < iC; ++ic) {
          for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
              int ih = oh * s - p + kh, iw = ow * s - p + kw;
              if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                size_t in_idx = ((b * iC + ic) * iH + ih) * iW + iw;
                size_t w_idx = ((oc * iC + ic) * kH + kh) * kW + kw;
                val += load_as_compute(in, in_idx) * load_as_compute(weight, w_idx);
              }
            }
          }
        }
        store_from_compute(out, idx, val);
      }
    });
  }
  void conv2d_backward(const Storage &grad_out, const Storage &in,
                       const Storage &weight, Storage &grad_in, Storage &grad_w,
                       Storage *grad_b, int B, int iC, int iH, int iW, int oC,
                       int kH, int kW, int s, int p) override {
    (void)resolve_compute_plan(accumulation_dtype(grad_out.dtype()));
    int oH = (iH + 2 * p - kH) / s + 1, oW = (iW + 2 * p - kW) / s + 1;
    for (int b = 0; b < B; ++b) {
      for (int oc = 0; oc < oC; ++oc) {
        for (int oh = 0; oh < oH; ++oh) {
          for (int ow = 0; ow < oW; ++ow) {
            size_t go_idx = ((b * oC + oc) * oH + oh) * oW + ow;
            double go_val = load_as_compute(grad_out, go_idx);
            if (grad_b) {
              double gb = load_as_compute(*grad_b, oc) + go_val;
              store_from_compute(*grad_b, oc, gb);
            }
            for (int ic = 0; ic < iC; ++ic) {
              for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                  int ih = oh * s - p + kh, iw = ow * s - p + kw;
                  if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                    size_t gi_idx = ((b * iC + ic) * iH + ih) * iW + iw;
                    size_t gw_idx = ((oc * iC + ic) * kH + kh) * kW + kw;
                    double gi = load_as_compute(grad_in, gi_idx) +
                                go_val * load_as_compute(weight, gw_idx);
                    double gw = load_as_compute(grad_w, gw_idx) +
                                go_val * load_as_compute(in, gi_idx);
                    store_from_compute(grad_in, gi_idx, gi);
                    store_from_compute(grad_w, gw_idx, gw);
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
    (void)resolve_compute_plan(accumulation_dtype(in.dtype()));
    int oH = (iH + 2 * p - k) / s + 1, oW = (iW + 2 * p - k) / s + 1;
    parallel_for(0, B * C * oH * oW, [&](size_t start, size_t end) {
      for (size_t idx = start; idx < end; ++idx) {
        int ow = idx % oW, oh = (idx / oW) % oH, c = (idx / (oW * oH)) % C,
            b = idx / (oW * oH * C);
        double max_val = -1e300;
        for (int kh = 0; kh < k; ++kh) {
          for (int kw = 0; kw < k; ++kw) {
            int ih = oh * s - p + kh, iw = ow * s - p + kw;
            if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
              size_t src_idx = ((b * C + c) * iH + ih) * iW + iw;
              max_val = std::max(max_val, load_as_compute(in, src_idx));
            }
          }
        }
        store_from_compute(out, idx, max_val);
      }
    });
  }
  void max_pool2d_backward(const Storage &grad_out, const Storage &in,
                           Storage &grad_in, int B, int C, int iH, int iW,
                           int k, int s, int p) override {
    (void)resolve_compute_plan(accumulation_dtype(grad_out.dtype()));
    int oH = (iH + 2 * p - k) / s + 1, oW = (iW + 2 * p - k) / s + 1;
    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < oH; ++oh) {
          for (int ow = 0; ow < oW; ++ow) {
            double max_val = -1e300;
            int max_idx = -1;
            for (int kh = 0; kh < k; ++kh) {
              for (int kw = 0; kw < k; ++kw) {
                int ih = oh * s - p + kh, iw = ow * s - p + kw;
                if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                  int idx = ((b * C + c) * iH + ih) * iW + iw;
                  double v = load_as_compute(in, static_cast<size_t>(idx));
                  if (v > max_val) {
                    max_val = v;
                    max_idx = idx;
                  }
                }
              }
            }
            if (max_idx != -1) {
              size_t go_idx = ((b * C + c) * oH + oh) * oW + ow;
              double accum = load_as_compute(grad_in, static_cast<size_t>(max_idx)) +
                             load_as_compute(grad_out, go_idx);
              store_from_compute(grad_in, static_cast<size_t>(max_idx), accum);
            }
          }
        }
      }
    }
  }
  void upsample2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int scale) override {
    (void)resolve_compute_plan(accumulation_dtype(in.dtype()));
    int oH = iH * scale, oW = iW * scale;
    parallel_for(0, B * C * oH * oW, [&](size_t start, size_t end) {
      for (size_t idx = start; idx < end; ++idx) {
        int ow = idx % oW, oh = (idx / oW) % oH, c = (idx / (oW * oH)) % C,
            b = idx / (oW * oH * C);
        size_t src_idx = ((b * C + c) * iH + (oh / scale)) * iW + (ow / scale);
        store_from_compute(out, idx, load_as_compute(in, src_idx));
      }
    });
  }
  void upsample2d_backward(const Storage &grad_out, Storage &grad_in, int B,
                           int C, int iH, int iW, int scale) override {
    (void)resolve_compute_plan(accumulation_dtype(grad_out.dtype()));
    int oH = iH * scale, oW = iW * scale;
    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < oH; ++oh) {
          for (int ow = 0; ow < oW; ++ow) {
            size_t gi_idx = ((b * C + c) * iH + (oh / scale)) * iW + (ow / scale);
            size_t go_idx = ((b * C + c) * oH + oh) * oW + ow;
            double accum = load_as_compute(grad_in, gi_idx) +
                           load_as_compute(grad_out, go_idx);
            store_from_compute(grad_in, gi_idx, accum);
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
    (void)resolve_compute_plan(accumulation_dtype(in.dtype()));
    size_t spatial = H * W;

    if (training) {
      for (int c = 0; c < C; ++c) {
        double sum = 0.0, sq_sum = 0.0;
        for (int batch = 0; batch < B; ++batch) {
          for (int s = 0; s < static_cast<int>(spatial); ++s) {
            size_t idx = (batch * C + c) * spatial + s;
            double val = load_as_compute(in, idx);
            sum += val;
            sq_sum += val * val;
          }
        }
        double n = static_cast<double>(B) * static_cast<double>(spatial);
        double mu = sum / n;
        double var = (sq_sum / n) - (mu * mu);
        store_from_compute(save_mean, c, mu);
        store_from_compute(save_var, c, var);

        double rm = load_as_compute(running_mean, c);
        double rv = load_as_compute(running_var, c);
        store_from_compute(running_mean, c,
                           (1.0 - momentum) * rm + momentum * mu);
        store_from_compute(running_var, c,
                           (1.0 - momentum) * rv + momentum * var);
      }
    }

    parallel_for(0, B * C * spatial, [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        int c = (i / spatial) % C;
        double mu = training ? load_as_compute(save_mean, c)
                             : load_as_compute(running_mean, c);
        double var = training ? load_as_compute(save_var, c)
                              : load_as_compute(running_var, c);
        double gamma = load_as_compute(scale, c);
        double beta = load_as_compute(bias, c);
        double x = load_as_compute(in, i);
        double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));
        store_from_compute(out, i, gamma * (x - mu) * inv_std + beta);
      }
    });
  }

  void batch_norm_backward(const Storage &grad_out, const Storage &in,
                           const Storage &scale, const Storage &save_mean,
                           const Storage &save_var, Storage &grad_in,
                           Storage &grad_scale, Storage &grad_bias, int B,
                           int C, int H, int W, float eps) override {
    (void)resolve_compute_plan(accumulation_dtype(grad_out.dtype()));
    size_t spatial = H * W;
    double m = static_cast<double>(B) * static_cast<double>(spatial);

    for (int c = 0; c < C; ++c) {
      store_from_compute(grad_scale, c, 0.0);
      store_from_compute(grad_bias, c, 0.0);
    }

    for (int c = 0; c < C; ++c) {
      double sum_dy = 0.0;
      double sum_dy_xhat = 0.0;
      double inv_std =
          1.0 / std::sqrt(load_as_compute(save_var, c) + static_cast<double>(eps));
      double mu = load_as_compute(save_mean, c);

      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < static_cast<int>(spatial); ++s) {
          size_t idx = (b * C + c) * spatial + s;
          double dy = load_as_compute(grad_out, idx);
          double x_hat = (load_as_compute(in, idx) - mu) * inv_std;
          sum_dy += dy;
          sum_dy_xhat += dy * x_hat;
        }
      }

      store_from_compute(grad_scale, c, sum_dy_xhat);
      store_from_compute(grad_bias, c, sum_dy);

      double gamma = load_as_compute(scale, c);
      double factor = gamma * inv_std / m;
      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < static_cast<int>(spatial); ++s) {
          size_t idx = (b * C + c) * spatial + s;
          double dy = load_as_compute(grad_out, idx);
          double x_hat = (load_as_compute(in, idx) - mu) * inv_std;
          double dx = factor * (m * dy - sum_dy - x_hat * sum_dy_xhat);
          store_from_compute(grad_in, idx, dx);
        }
      }
    }
  }
  void fill_uniform(Storage &out, float low, float high,
                    size_t num_elements) override {
    // Not strictly thread-safe to share generator, creating local one
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      std::mt19937 gen(42 + s); // Seed offset by index
      std::uniform_real_distribution<float> dis(low, high);
      for (size_t i = s; i < e; ++i) {
        store_from_compute(out, i, static_cast<double>(dis(gen)));
      }
    });
  }

  void sum(const Storage &in, Storage &out, size_t num_elements) override {
    (void)resolve_compute_plan(accumulation_dtype(in.dtype()));
    double total = 0.0;
    for (size_t i = 0; i < num_elements; ++i)
      total += (double)load_as_compute(in, i);
    store_from_compute(out, 0, (float)total);
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
};
} // namespace munet
