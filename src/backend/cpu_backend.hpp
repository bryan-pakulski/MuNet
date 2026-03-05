#pragma once
#include "../backend.hpp"
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

class CPUBackend : public Backend {
private:
  // Caching
  std::unordered_map<size_t, std::vector<void *>> free_blocks_;
  std::unordered_map<void *, size_t> alloc_sizes_;
  std::mutex mem_mutex_;

  static ThreadPool &get_pool() {
    static ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
  }

  template <typename Func>
  void parallel_for(size_t start, size_t end, Func func) {
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
  }

public:
  ~CPUBackend() override {
    for (auto &kv : free_blocks_) {
      for (void *ptr : kv.second) {
        std::free(ptr);
      }
    }
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
           size_t num_elements) override {
    const float *ap = (const float *)a.data();
    const float *bp = (const float *)b.data();
    float *op = (float *)out.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        op[i] = ap[i] + bp[i];
    });
  }

  void mul(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override {
    const float *ap = (const float *)a.data();
    const float *bp = (const float *)b.data();
    float *op = (float *)out.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        op[i] = ap[i] * bp[i];
    });
  }

  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) override {
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

  void sub(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override {
    const float *ap = (const float *)a.data(), *bp = (const float *)b.data();
    float *op = (float *)out.data();
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      for (size_t i = s; i < e; ++i)
        op[i] = ap[i] - bp[i];
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
        int s_idx = i % spatial;
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
    float *ptr = (float *)out.data();
    // Not strictly thread-safe to share generator, creating local one
    parallel_for(0, num_elements, [&](size_t s, size_t e) {
      std::mt19937 gen(42 + s); // Seed offset by index
      std::uniform_real_distribution<float> dis(low, high);
      for (size_t i = s; i < e; ++i) {
        ptr[i] = dis(gen);
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
};
} // namespace munet
