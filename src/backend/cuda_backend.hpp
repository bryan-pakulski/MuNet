#pragma once
#include "../backend.hpp"

namespace munet {

class CUDABackend : public Backend {
private:
  void *start_event_; // Using void* to avoid nvcc requirement in header
  void *stop_event_;
  int device_index_ = 0;
  double last_kernel_us_ = 0.0;

public:
  CUDABackend(int device_index);
  ~CUDABackend() override;

  double get_last_kernel_time_us() override { return last_kernel_us_; }

  void *allocate(size_t bytes) override;
  void deallocate(void *ptr) override;
  void memset(void *ptr, int value, size_t bytes) override;

  void copy(const void *src, void *dst, size_t bytes, Device src_dev,
            Device dst_dev) override;
  void synchronize() override;
  void all_reduce(Storage &buffer, size_t num_elements) override;
  void broadcast_row(const Storage &src, Storage &dst, int rows,
                     int cols) override;

  // --- Compute Kernels ---
  void add(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override;
  void mul(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override;
  void div(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override;
  void sub(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) override;
  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) override;
  void relu(const Storage &in, Storage &out, size_t num_elements) override;
  void relu_backward(const Storage &grad_out, const Storage &input,
                     Storage &grad_in, size_t num_elements) override;

  void sigmoid(const Storage &in, Storage &out, size_t num_elements) override;
  void sigmoid_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, size_t num_elements) override;
  void log(const Storage &in, Storage &out, size_t num_elements) override;
  void sqrt(const Storage &in, Storage &out, size_t num_elements) override;
  void clip(const Storage &in, Storage &out, float min_value, float max_value,
            size_t num_elements) override;
  void erf(const Storage &in, Storage &out, size_t num_elements) override;
  void softmax(const Storage &in, Storage &out, int batch_size,
               int num_classes) override;
  void softmax_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, int batch_size,
                        int num_classes) override;
  void topk(const Storage &in, Storage &out_values, Storage &out_indices,
            int outer, int dim_size, int k, bool largest, bool sorted) override;
  void cross_entropy(const Storage &logits, const Storage &targets,
                     Storage &out_loss, int batch_size, int num_classes,
                     int spatial) override;
  void cross_entropy_backward(const Storage &grad_out, const Storage &logits,
                              const Storage &targets, Storage &grad_in,
                              int batch_size, int num_classes,
                              int spatial) override;

  void mse_loss(const Storage &pred, const Storage &target, Storage &out_loss,
                size_t num_elements) override;
  void mse_loss_backward(const Storage &grad_out, const Storage &pred,
                         const Storage &target, Storage &grad_in,
                         size_t num_elements) override;
  void update(Storage &weight, const Storage &grad, float lr,
              size_t num_elements) override;

  void concat(const std::vector<Storage *> &inputs, Storage &out, int dim,
              const std::vector<Shape> &shapes) override;
  void gather_elements(const Storage &data, const Storage &indices,
                       Storage &out, const Shape &shape, int axis) override;
  void concat_backward(const Storage &grad_out,
                       std::vector<Storage *> &grad_inputs, int dim,
                       const std::vector<Shape> &shapes) override;

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

  void fill_uniform(Storage &out, float low, float high,
                    size_t num_elements) override;
  void sum(const Storage &in, Storage &out, size_t num_elements) override;
  void sum_to_shape(const Storage &in, Storage &out, const Shape &in_shape,
                    const Shape &out_shape) override;
  void adam_step(Storage &params, const Storage &grads, Storage &exp_avg,
                 Storage &exp_avg_sq, float lr, float beta1, float beta2,
                 float eps, int step, size_t num_elements) override;
};

} // namespace munet
