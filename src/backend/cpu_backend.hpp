#pragma once
#include "../backend.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace munet {

class CPUBackend : public Backend {
public:
  void *allocate(size_t bytes) override {
    void *ptr = std::malloc(bytes);
    if (!ptr)
      throw std::bad_alloc();
    return ptr;
  }

  void deallocate(void *ptr) override { std::free(ptr); }

  void memset(void *ptr, int value, size_t bytes) override {
    std::memset(ptr, value, bytes);
  }

  void copy(const void *src, void *dst, size_t bytes, Device src_dev,
            Device dst_dev) override {
    // In reality, check if dst_dev or src_dev is CUDA and route accordingly.
    std::memcpy(dst, src, bytes);
  }

  void synchronize() override {
    // CPU is synchronous by default
  }

  void all_reduce(Storage &buffer, size_t num_elements) override {
    // No-op for single CPU.
  }

  // --- Compute Kernels ---

  void add(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override {
    const float *ap = (const float *)a.data();
    const float *bp = (const float *)b.data();
    float *op = (float *)out.data();
    for (size_t i = 0; i < num_elements; ++i)
      op[i] = ap[i] + bp[i];
  }

  void mul(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override {
    const float *ap = (const float *)a.data();
    const float *bp = (const float *)b.data();
    float *op = (float *)out.data();
    for (size_t i = 0; i < num_elements; ++i)
      op[i] = ap[i] * bp[i];
  }

  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) override {
    const float *ap = (const float *)a.data();
    const float *bp = (const float *)b.data();
    float *cp = (float *)out.data();

    std::memset(cp, 0, M * N * sizeof(float));

    for (int m = 0; m < M; ++m) {
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
  }

  void relu(const Storage &in, Storage &out, size_t num_elements) override {
    const float *ip = (const float *)in.data();
    float *op = (float *)out.data();
    for (size_t i = 0; i < num_elements; ++i)
      op[i] = ip[i] > 0 ? ip[i] : 0;
  }

  void relu_backward(const Storage &grad_out, const Storage &input,
                     Storage &grad_in, size_t num_elements) override {
    const float *go = (const float *)grad_out.data();
    const float *in = (const float *)input.data();
    float *gi = (float *)grad_in.data();
    for (size_t i = 0; i < num_elements; ++i)
      gi[i] = (in[i] > 0) ? go[i] : 0.0f;
  }

  void sub(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override {
    const float *ap = (const float *)a.data(), *bp = (const float *)b.data();
    float *op = (float *)out.data();
    for (size_t i = 0; i < num_elements; ++i)
      op[i] = ap[i] - bp[i]; // Fixed: access index i
  }

  void update(Storage &weight, const Storage &grad, float lr,
              size_t num_elements) override {
    float *w = (float *)weight.data();
    const float *g = (const float *)grad.data();
    for (size_t i = 0; i < num_elements; ++i)
      w[i] -= lr * g[i]; // Fixed: access index i
  }
};

} // namespace munet
