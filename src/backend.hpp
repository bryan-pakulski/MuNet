#pragma once
#include "types.hpp"
#include <memory>

namespace munet {

class Storage;

class Backend {
public:
  virtual ~Backend() = default;

  virtual void *allocate(size_t bytes) = 0;
  virtual void deallocate(void *ptr) = 0;
  virtual void memset(void *ptr, int value, size_t bytes) = 0;

  virtual void copy(const void *src, void *dst, size_t bytes, Device src_dev,
                    Device dst_dev) = 0;
  virtual void synchronize() = 0;
  virtual void all_reduce(Storage &buffer, size_t num_elements) = 0;

  // --- Compute Operations ---
  virtual void add(const Storage &a, const Storage &b, Storage &out,
                   size_t num_elements) = 0;
  virtual void sub(const Storage &a, const Storage &b, Storage &out,
                   size_t num_elements) = 0;
  virtual void mul(const Storage &a, const Storage &b, Storage &out,
                   size_t num_elements) = 0;

  virtual void matmul(const Storage &a, const Storage &b, Storage &out, int M,
                      int K, int N, bool transA, bool transB) = 0;

  virtual void relu(const Storage &in, Storage &out, size_t num_elements) = 0;
  virtual void relu_backward(const Storage &grad_out, const Storage &input,
                             Storage &grad_in, size_t num_elements) = 0;

  // --- Spatial Compute ---
  virtual void conv2d(const Storage &in, const Storage &weight,
                      const Storage *bias, Storage &out, int B, int iC, int iH,
                      int iW, int oC, int kH, int kW, int s, int p) = 0;
  virtual void conv2d_backward(const Storage &grad_out, const Storage &in,
                               const Storage &weight, Storage &grad_in,
                               Storage &grad_w, Storage *grad_b, int B, int iC,
                               int iH, int iW, int oC, int kH, int kW, int s,
                               int p) = 0;
  virtual void max_pool2d(const Storage &in, Storage &out, int B, int C, int iH,
                          int iW, int k, int s, int p) = 0;
  virtual void max_pool2d_backward(const Storage &grad_out, const Storage &in,
                                   Storage &grad_in, int B, int C, int iH,
                                   int iW, int k, int s, int p) = 0;
  virtual void upsample2d(const Storage &in, Storage &out, int B, int C, int iH,
                          int iW, int scale) = 0;
  virtual void upsample2d_backward(const Storage &grad_out, Storage &grad_in,
                                   int B, int C, int iH, int iW, int scale) = 0;

  // --- Optimizers ---
  // In-place SGD update: w = w - lr * grad
  virtual void update(Storage &weight, const Storage &grad, float lr,
                      size_t num_elements) = 0;
};

class BackendManager {
public:
  static std::shared_ptr<Backend> get(Device device);
};

} // namespace munet
