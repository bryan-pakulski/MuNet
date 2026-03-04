#pragma once
#include "../backend.hpp"

namespace munet {

class CUDABackend : public Backend {
public:
  CUDABackend();
  ~CUDABackend() override;

  void *allocate(size_t bytes) override;
  void deallocate(void *ptr) override;
  void memset(void *ptr, int value, size_t bytes) override;

  void copy(const void *src, void *dst, size_t bytes, Device src_dev,
            Device dst_dev) override;
  void synchronize() override;
  void all_reduce(Storage &buffer, size_t num_elements) override;

  // --- Compute Kernels ---
  void add(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override;
  void mul(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override;
  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) override;
  void relu(const Storage &in, Storage &out, size_t num_elements) override;
  void relu_backward(const Storage &grad_out, const Storage &input,
                     Storage &grad_in, size_t num_elements) override;

  void sigmoid(const Storage &in, Storage &out, size_t num_elements) override;
  void sigmoid_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, size_t num_elements) override;
  void softmax(const Storage &in, Storage &out, int batch_size,
               int num_classes) override;
  void softmax_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, int batch_size,
                        int num_classes) override;

  void cross_entropy(const Storage &logits, const Storage &targets,
                     Storage &out_loss, int batch_size,
                     int num_classes) override;
  void cross_entropy_backward(const Storage &grad_out, const Storage &logits,
                              const Storage &targets, Storage &grad_in,
                              int batch_size, int num_classes) override;

  void mse_loss(const Storage &pred, const Storage &target, Storage &out_loss,
                size_t num_elements) override;
  void mse_loss_backward(const Storage &grad_out, const Storage &pred,
                         const Storage &target, Storage &grad_in,
                         size_t num_elements) override;

  void sub(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override;
  void update(Storage &weight, const Storage &grad, float lr,
              size_t num_elements) override;

  void conv2d(const Storage &in, const Storage &weight, const Storage *bias,
              Storage &out, int B, int iC, int iH, int iW, int oC, int kH,
              int kW, int s, int p) override;
  void conv2d_backward(const Storage &grad_out, const Storage &in,
                       const Storage &weight, Storage &grad_in, Storage &grad_w,
                       Storage *grad_b, int B, int iC, int iH, int iW, int oC,
                       int kH, int kW, int s, int p) override;
  void max_pool2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int k, int s, int p) override;
  void max_pool2d_backward(const Storage &grad_out, const Storage &in,
                           Storage &grad_in, int B, int C, int iH, int iW,
                           int k, int s, int p) override;
  void upsample2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int scale) override;
  void upsample2d_backward(const Storage &grad_out, Storage &grad_in, int B,
                           int C, int iH, int iW, int scale) override;

  void batch_norm(const Storage &in, const Storage &scale, const Storage &bias,
                  Storage &running_mean, Storage &running_var,
                  Storage &save_mean, Storage &save_var, Storage &out, int B,
                  int C, int H, int W, float momentum, float eps,
                  bool training) override;
  void batch_norm_backward(const Storage &grad_out, const Storage &in,
                           const Storage &scale, const Storage &save_mean,
                           const Storage &save_var, Storage &grad_in,
                           Storage &grad_scale, Storage &grad_bias, int B,
                           int C, int H, int W, float eps) override;
};

} // namespace munet
