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

  virtual void broadcast_row(const Storage &src, Storage &dst, int rows,
                             int cols) = 0;

  // Timing retrieval
  virtual double get_last_kernel_time_us() = 0;

  // --- Compute Operations ---
  virtual void add(const Storage &a, const Storage &b, Storage &out,
                   const BroadcastInfo &info) = 0;
  virtual void sub(const Storage &a, const Storage &b, Storage &out,
                   const BroadcastInfo &info) = 0;
  virtual void mul(const Storage &a, const Storage &b, Storage &out,
                   const BroadcastInfo &info) = 0;

  virtual void matmul(const Storage &a, const Storage &b, Storage &out, int M,
                      int K, int N, bool transA, bool transB) = 0;

  virtual void relu(const Storage &in, Storage &out, size_t num_elements) = 0;
  virtual void relu_backward(const Storage &grad_out, const Storage &input,
                             Storage &grad_in, size_t num_elements) = 0;

  virtual void sigmoid(const Storage &in, Storage &out,
                       size_t num_elements) = 0;
  virtual void sigmoid_backward(const Storage &grad_out, const Storage &out,
                                Storage &grad_in, size_t num_elements) = 0;

  virtual void softmax(const Storage &in, Storage &out, int batch_size,
                       int num_classes) = 0;
  virtual void softmax_backward(const Storage &grad_out, const Storage &out,
                                Storage &grad_in, int batch_size,
                                int num_classes) = 0;

  virtual void concat(const std::vector<Storage *> &inputs, Storage &out,
                      int dim, const std::vector<Shape> &shapes) = 0;
  virtual void concat_backward(const Storage &grad_out,
                               std::vector<Storage *> &grad_inputs, int dim,
                               const std::vector<Shape> &shapes) = 0;

  // --- Loss Functions ---
  // Updated signatures to support spatial dimensions
  virtual void cross_entropy(const Storage &logits, const Storage &targets,
                             Storage &out_loss, int batch_size, int num_classes,
                             int spatial) = 0;
  virtual void cross_entropy_backward(const Storage &grad_out,
                                      const Storage &logits,
                                      const Storage &targets, Storage &grad_in,
                                      int batch_size, int num_classes,
                                      int spatial) = 0;

  virtual void mse_loss(const Storage &pred, const Storage &target,
                        Storage &out_loss, size_t num_elements) = 0;
  virtual void mse_loss_backward(const Storage &grad_out, const Storage &pred,
                                 const Storage &target, Storage &grad_in,
                                 size_t num_elements) = 0;

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

  // --- Normalization ---
  virtual void batch_norm(const Storage &in, const Storage &scale,
                          const Storage &bias, Storage &running_mean,
                          Storage &running_var, Storage &save_mean,
                          Storage &save_var, Storage &out, int B, int C, int H,
                          int W, float momentum, float eps, bool training) = 0;
  virtual void batch_norm_backward(const Storage &grad_out, const Storage &in,
                                   const Storage &scale,
                                   const Storage &save_mean,
                                   const Storage &save_var, Storage &grad_in,
                                   Storage &grad_scale, Storage &grad_bias,
                                   int B, int C, int H, int W, float eps) = 0;

  // --- Optimizers ---
  virtual void adam_step(Storage &params, const Storage &grads,
                         Storage &exp_avg, Storage &exp_avg_sq, float lr,
                         float beta1, float beta2, float eps, int step,
                         size_t num_elements) = 0;

  // update: w = w - lr * grad
  virtual void update(Storage &weight, const Storage &grad, float lr,
                      size_t num_elements) = 0;

  // --- Random ---
  virtual void fill_uniform(Storage &out, float low, float high,
                            size_t num_elements) = 0;

  // --- Reduction ---
  virtual void sum_to_shape(const Storage &in, Storage &out,
                            const Shape &in_shape, const Shape &out_shape) = 0;
  virtual void sum(const Storage &in, Storage &out, size_t num_elements) = 0;
};

class BackendManager {
public:
  static std::shared_ptr<Backend> get(Device device);
};

} // namespace munet
