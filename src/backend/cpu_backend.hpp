#pragma once
#include "../backend.hpp"
#include "../profiler.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
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
  template <typename F> void profile(const char *name, F func) {
#ifdef ENABLE_PROFILING
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count();
    Profiler::get().log(name, "cpu", us);
#else
    func();
#endif
  }

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
    profile("add", [&]() {
      const float *ap = (const float *)a.data();
      const float *bp = (const float *)b.data();
      float *op = (float *)out.data();
      parallel_for(0, num_elements, [&](size_t s, size_t e) {
        for (size_t i = s; i < e; ++i)
          op[i] = ap[i] + bp[i];
      });
    });
  }

  void mul(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override {
    profile("mul", [&]() {
      const float *ap = (const float *)a.data();
      const float *bp = (const float *)b.data();
      float *op = (float *)out.data();
      parallel_for(0, num_elements, [&](size_t s, size_t e) {
        for (size_t i = s; i < e; ++i)
          op[i] = ap[i] * bp[i];
      });
    });
  }

  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) override {
    profile("matmul", [&]() {
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
    });
  }

  void relu(const Storage &in, Storage &out, size_t num_elements) override {
    profile("relu", [&]() {
      const float *ip = (const float *)in.data();
      float *op = (float *)out.data();
      parallel_for(0, num_elements, [&](size_t s, size_t e) {
        for (size_t i = s; i < e; ++i)
          op[i] = ip[i] > 0 ? ip[i] : 0;
      });
    });
  }

  void relu_backward(const Storage &grad_out, const Storage &input,
                     Storage &grad_in, size_t num_elements) override {
    profile("relu_backward", [&]() {
      const float *go = (const float *)grad_out.data();
      const float *in = (const float *)input.data();
      float *gi = (float *)grad_in.data();
      parallel_for(0, num_elements, [&](size_t s, size_t e) {
        for (size_t i = s; i < e; ++i)
          gi[i] = (in[i] > 0) ? go[i] : 0.0f;
      });
    });
  }

  void sub(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override {
    profile("sub", [&]() {
      const float *ap = (const float *)a.data(), *bp = (const float *)b.data();
      float *op = (float *)out.data();
      parallel_for(0, num_elements, [&](size_t s, size_t e) {
        for (size_t i = s; i < e; ++i)
          op[i] = ap[i] - bp[i];
      });
    });
  }

  void update(Storage &weight, const Storage &grad, float lr,
              size_t num_elements) override {
    profile("update", [&]() {
      float *w = (float *)weight.data();
      const float *g = (const float *)grad.data();
      parallel_for(0, num_elements, [&](size_t s, size_t e) {
        for (size_t i = s; i < e; ++i)
          w[i] -= lr * g[i];
      });
    });
  }
};

} // namespace munet
