#pragma once

#include "../types.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace munet {

class Storage;

enum class BackendFeature {
  ElementwiseBinary,
  BroadcastRow,
  Matmul,
  UnaryActivation,
  Softmax,
  Concat,
  Loss,
  Convolution,
  Pooling,
  BatchNorm,
  OptimizerStep,
  RandomFill,
  Reduction,
};

enum class BackendFallbackPolicy {
  ExplicitUnsupported,
  CPUFallback,
  ConversionFallback,
};

enum class BackendFeatureRuntimeRole {
  DeployRuntime,
  TrainingOnly,
};

struct BackendSupport {
  bool available = false;
  BackendFallbackPolicy fallback_policy =
      BackendFallbackPolicy::ExplicitUnsupported;
  DataType preferred_accumulation_dtype = DataType::Float32;
};

inline bool supports_backend_feature_dtype(BackendFeature feature,
                                           DataType dtype) {
  switch (feature) {
  case BackendFeature::RandomFill:
    return is_floating(dtype);
  case BackendFeature::ElementwiseBinary:
  case BackendFeature::BroadcastRow:
  case BackendFeature::Matmul:
  case BackendFeature::UnaryActivation:
  case BackendFeature::Softmax:
  case BackendFeature::Concat:
  case BackendFeature::Loss:
  case BackendFeature::Convolution:
  case BackendFeature::Pooling:
  case BackendFeature::BatchNorm:
  case BackendFeature::OptimizerStep:
  case BackendFeature::Reduction:
    return dtype == DataType::Float32;
  default:
    return false;
  }
}

inline bool supports_backend_feature_shape(BackendFeature, DataType,
                                           const Shape *) {
  return true;
}

inline const char *backend_feature_name(BackendFeature feature) {
  switch (feature) {
  case BackendFeature::ElementwiseBinary:
    return "elementwise_binary";
  case BackendFeature::BroadcastRow:
    return "broadcast_row";
  case BackendFeature::Matmul:
    return "matmul";
  case BackendFeature::UnaryActivation:
    return "unary_activation";
  case BackendFeature::Softmax:
    return "softmax";
  case BackendFeature::Concat:
    return "concat";
  case BackendFeature::Loss:
    return "loss";
  case BackendFeature::Convolution:
    return "convolution";
  case BackendFeature::Pooling:
    return "pooling";
  case BackendFeature::BatchNorm:
    return "batch_norm";
  case BackendFeature::OptimizerStep:
    return "optimizer_step";
  case BackendFeature::RandomFill:
    return "random_fill";
  case BackendFeature::Reduction:
    return "reduction";
  default:
    return "unknown";
  }
}

inline const char *backend_fallback_policy_name(BackendFallbackPolicy policy) {
  switch (policy) {
  case BackendFallbackPolicy::ExplicitUnsupported:
    return "explicit_unsupported";
  case BackendFallbackPolicy::CPUFallback:
    return "cpu_fallback";
  case BackendFallbackPolicy::ConversionFallback:
    return "conversion_fallback";
  default:
    return "unknown";
  }
}

inline BackendFeatureRuntimeRole
backend_feature_runtime_role(BackendFeature feature) {
  switch (feature) {
  case BackendFeature::OptimizerStep:
    return BackendFeatureRuntimeRole::TrainingOnly;
  case BackendFeature::ElementwiseBinary:
  case BackendFeature::BroadcastRow:
  case BackendFeature::Matmul:
  case BackendFeature::UnaryActivation:
  case BackendFeature::Softmax:
  case BackendFeature::Concat:
  case BackendFeature::Loss:
  case BackendFeature::Convolution:
  case BackendFeature::Pooling:
  case BackendFeature::BatchNorm:
  case BackendFeature::RandomFill:
  case BackendFeature::Reduction:
    return BackendFeatureRuntimeRole::DeployRuntime;
  default:
    return BackendFeatureRuntimeRole::DeployRuntime;
  }
}

inline bool is_training_only_backend_feature(BackendFeature feature) {
  return backend_feature_runtime_role(feature) ==
         BackendFeatureRuntimeRole::TrainingOnly;
}

inline DataType default_backend_accumulation_dtype(BackendFeature feature,
                                                   DataType dtype) {
  switch (feature) {
  case BackendFeature::ElementwiseBinary:
  case BackendFeature::Matmul:
  case BackendFeature::Reduction:
  case BackendFeature::Loss:
  case BackendFeature::BatchNorm:
  case BackendFeature::OptimizerStep:
    return is_floating(dtype) ? DataType::Float32 : dtype;
  default:
    return dtype;
  }
}

inline BackendFallbackPolicy
backend_feature_default_fallback_policy(BackendFeature feature) {
  switch (feature) {
  case BackendFeature::ElementwiseBinary:
  case BackendFeature::BroadcastRow:
  case BackendFeature::Matmul:
  case BackendFeature::UnaryActivation:
  case BackendFeature::Softmax:
  case BackendFeature::Reduction:
  case BackendFeature::RandomFill:
  case BackendFeature::Loss:
  case BackendFeature::Pooling:
  case BackendFeature::BatchNorm:
    return BackendFallbackPolicy::CPUFallback;
  case BackendFeature::Convolution:
  case BackendFeature::Concat:
  case BackendFeature::OptimizerStep:
    return BackendFallbackPolicy::ExplicitUnsupported;
  default:
    return BackendFallbackPolicy::ExplicitUnsupported;
  }
}

class BackendAllocationTransferCapability {
public:
  virtual ~BackendAllocationTransferCapability() = default;
  virtual void *allocate(size_t bytes) = 0;
  virtual void deallocate(void *ptr) = 0;
  virtual void memset(void *ptr, int value, size_t bytes) = 0;
  virtual void copy(const void *src, void *dst, size_t bytes, Device src_dev,
                    Device dst_dev) = 0;
  virtual void synchronize() = 0;
  virtual void all_reduce(Storage &buffer, size_t num_elements) = 0;
  virtual double get_last_kernel_time_us() = 0;
  virtual bool reports_gpu_kernel_time() const { return false; }
  virtual void to_contiguous(const Storage &src, Storage &dst,
                             const Shape &shape, const Strides &strides,
                             size_t offset) = 0;
};

class BackendElementwiseCapability {
public:
  virtual ~BackendElementwiseCapability() = default;
  virtual void add(const Storage &a, const Storage &b, Storage &out,
                   const BroadcastInfo &info) = 0;
  virtual void sub(const Storage &a, const Storage &b, Storage &out,
                   const BroadcastInfo &info) = 0;
  virtual void mul(const Storage &a, const Storage &b, Storage &out,
                   const BroadcastInfo &info) = 0;
  virtual void div(const Storage &a, const Storage &b, Storage &out,
                   const BroadcastInfo &info) = 0;
  virtual void broadcast_row(const Storage &src, Storage &dst, int rows,
                             int cols) = 0;
  virtual void relu(const Storage &in, Storage &out, size_t num_elements) = 0;
  virtual void relu_backward(const Storage &grad_out, const Storage &input,
                             Storage &grad_in, size_t num_elements) = 0;
  virtual void sigmoid(const Storage &in, Storage &out,
                       size_t num_elements) = 0;
  virtual void sigmoid_backward(const Storage &grad_out, const Storage &out,
                                Storage &grad_in, size_t num_elements) = 0;
  virtual void exp(const Storage &in, Storage &out, size_t num_elements) = 0;
  virtual void log(const Storage &in, Storage &out, size_t num_elements) = 0;
  virtual void sqrt(const Storage &in, Storage &out, size_t num_elements) = 0;
  virtual void rsqrt(const Storage &in, Storage &out, size_t num_elements) = 0;
  virtual void sin(const Storage &in, Storage &out, size_t num_elements) = 0;
  virtual void cos(const Storage &in, Storage &out, size_t num_elements) = 0;
  virtual void softmax(const Storage &in, Storage &out, int batch_size,
                       int num_classes) = 0;
  virtual void log_softmax(const Storage &, Storage &, int, int) {
    throw std::runtime_error("log_softmax not implemented");
  }
  virtual void softmax_backward(const Storage &grad_out, const Storage &out,
                                Storage &grad_in, int batch_size,
                                int num_classes) = 0;
};

class BackendReductionCapability {
public:
  virtual ~BackendReductionCapability() = default;
  virtual void sum_to_shape(const Storage &in, Storage &out,
                            const Shape &in_shape, const Shape &out_shape) = 0;
  virtual void sum(const Storage &in, Storage &out, size_t num_elements) = 0;
  virtual void mean_last_dim(const Storage &in, Storage &out, int outer_size,
                             int dim_size) = 0;
};

class BackendBlasCapability {
public:
  virtual ~BackendBlasCapability() = default;
  virtual void matmul(const Storage &a, const Storage &b, Storage &out, int M,
                      int K, int N, bool transA, bool transB) = 0;
  virtual void batched_matmul(const Storage &a, const Storage &b, Storage &out,
                              int batch_size, int M, int K, int N,
                              bool transA, bool transB,
                              int64_t stride_a, int64_t stride_b,
                              int64_t stride_out) = 0;
};


class BackendShapeCapability {
public:
  virtual ~BackendShapeCapability() = default;
  virtual void concat(const std::vector<Storage *> &inputs, Storage &out,
                      int dim, const std::vector<Shape> &shapes) = 0;
  virtual void concat_backward(const Storage &grad_out,
                               std::vector<Storage *> &grad_inputs, int dim,
                               const std::vector<Shape> &shapes) = 0;
};

class BackendLossCapability {
public:
  virtual ~BackendLossCapability() = default;
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
};

class BackendSpatialCapability {
public:
  virtual ~BackendSpatialCapability() = default;
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
};

class BackendNormalizationCapability {
public:
  virtual ~BackendNormalizationCapability() = default;
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
};

class BackendOptimizerCapability {
public:
  virtual ~BackendOptimizerCapability() = default;
  virtual void adam_step(Storage &params, const Storage &grads,
                         Storage &exp_avg, Storage &exp_avg_sq, float lr,
                         float beta1, float beta2, float eps, int step,
                         size_t num_elements) = 0;
  virtual void update(Storage &weight, const Storage &grad, float lr,
                      size_t num_elements) = 0;
};

class BackendRandomCapability {
public:
  virtual ~BackendRandomCapability() = default;
  virtual void fill_uniform(Storage &out, float low, float high,
                            size_t num_elements) = 0;
};

class Backend {
public:
  virtual ~Backend() = default;
  virtual const char *name() const = 0;

  virtual BackendAllocationTransferCapability *
  allocation_transfer_capability() {
    return nullptr;
  }
  virtual const BackendAllocationTransferCapability *
  allocation_transfer_capability() const {
    return nullptr;
  }

  virtual BackendElementwiseCapability *elementwise_capability() {
    return nullptr;
  }
  virtual const BackendElementwiseCapability *elementwise_capability() const {
    return nullptr;
  }

  virtual BackendReductionCapability *reduction_capability() { return nullptr; }
  virtual const BackendReductionCapability *reduction_capability() const {
    return nullptr;
  }

  virtual BackendBlasCapability *blas_capability() { return nullptr; }
  virtual const BackendBlasCapability *blas_capability() const {
    return nullptr;
  }

  virtual BackendShapeCapability *shape_capability() { return nullptr; }
  virtual const BackendShapeCapability *shape_capability() const {
    return nullptr;
  }

  virtual BackendLossCapability *loss_capability() { return nullptr; }
  virtual const BackendLossCapability *loss_capability() const {
    return nullptr;
  }

  virtual BackendSpatialCapability *spatial_capability() { return nullptr; }
  virtual const BackendSpatialCapability *spatial_capability() const {
    return nullptr;
  }

  virtual BackendNormalizationCapability *normalization_capability() {
    return nullptr;
  }
  virtual const BackendNormalizationCapability *
  normalization_capability() const {
    return nullptr;
  }

  virtual BackendOptimizerCapability *optimizer_capability() { return nullptr; }
  virtual const BackendOptimizerCapability *optimizer_capability() const {
    return nullptr;
  }

  virtual BackendRandomCapability *random_capability() { return nullptr; }
  virtual const BackendRandomCapability *random_capability() const {
    return nullptr;
  }

  BackendSupport query_support(BackendFeature feature, DataType dtype) const {
    return query_support(feature, dtype, nullptr);
  }

  virtual BackendSupport query_support(BackendFeature feature, DataType dtype,
                                       const Shape *shape) const {
    BackendSupport support;
    support.available = has_capability(feature) &&
                        supports_backend_feature_dtype(feature, dtype) &&
                        supports_backend_feature_shape(feature, dtype, shape);
    support.preferred_accumulation_dtype =
        preferred_accumulation_dtype(feature, dtype);
    support.fallback_policy = preferred_fallback_policy(feature, dtype);
    return support;
  }

  virtual bool supports(BackendFeature feature, DataType dtype) const {
    return query_support(feature, dtype).available;
  }

  virtual bool supports(BackendFeature feature, DataType dtype,
                        const Shape &shape) const {
    return query_support(feature, dtype, &shape).available;
  }

  virtual DataType preferred_accumulation_dtype(BackendFeature feature,
                                                DataType dtype) const {
    return default_backend_accumulation_dtype(feature, dtype);
  }

  virtual BackendFallbackPolicy
  preferred_fallback_policy(BackendFeature feature, DataType dtype) const {
    (void)dtype;
    return backend_feature_default_fallback_policy(feature);
  }

  void *allocate(size_t bytes) {
    return require_capability(allocation_transfer_capability(),
                              "allocation_transfer", "allocate")
        ->allocate(bytes);
  }
  void deallocate(void *ptr) {
    require_capability(allocation_transfer_capability(), "allocation_transfer",
                       "deallocate")
        ->deallocate(ptr);
  }
  void memset(void *ptr, int value, size_t bytes) {
    require_capability(allocation_transfer_capability(), "allocation_transfer",
                       "memset")
        ->memset(ptr, value, bytes);
  }
  void copy(const void *src, void *dst, size_t bytes, Device src_dev,
            Device dst_dev) {
    require_capability(allocation_transfer_capability(), "allocation_transfer",
                       "copy")
        ->copy(src, dst, bytes, src_dev, dst_dev);
  }
  void synchronize() {
    require_capability(allocation_transfer_capability(), "allocation_transfer",
                       "synchronize")
        ->synchronize();
  }
  void all_reduce(Storage &buffer, size_t num_elements) {
    require_capability(allocation_transfer_capability(), "allocation_transfer",
                       "all_reduce")
        ->all_reduce(buffer, num_elements);
  }
  double get_last_kernel_time_us() {
    return require_capability(allocation_transfer_capability(),
                              "allocation_transfer", "get_last_kernel_time_us")
        ->get_last_kernel_time_us();
  }
  bool reports_gpu_kernel_time() const {
    const auto *cap = allocation_transfer_capability();
    return cap ? cap->reports_gpu_kernel_time() : false;
  }

  void broadcast_row(const Storage &src, Storage &dst, int rows, int cols) {
    require_capability(elementwise_capability(), "elementwise", "broadcast_row")
        ->broadcast_row(src, dst, rows, cols);
  }
  void add(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) {
    require_capability(elementwise_capability(), "elementwise", "add")
        ->add(a, b, out, info);
  }
  void sub(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) {
    require_capability(elementwise_capability(), "elementwise", "sub")
        ->sub(a, b, out, info);
  }
  void mul(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) {
    require_capability(elementwise_capability(), "elementwise", "mul")
        ->mul(a, b, out, info);
  }
  void div(const Storage &a, const Storage &b, Storage &out,
           const BroadcastInfo &info) {
    require_capability(elementwise_capability(), "elementwise", "div")
        ->div(a, b, out, info);
  }
  void relu(const Storage &in, Storage &out, size_t num_elements) {
    require_capability(elementwise_capability(), "elementwise", "relu")
        ->relu(in, out, num_elements);
  }
  void relu_backward(const Storage &grad_out, const Storage &input,
                     Storage &grad_in, size_t num_elements) {
    require_capability(elementwise_capability(), "elementwise", "relu_backward")
        ->relu_backward(grad_out, input, grad_in, num_elements);
  }
  void sigmoid(const Storage &in, Storage &out, size_t num_elements) {
    require_capability(elementwise_capability(), "elementwise", "sigmoid")
        ->sigmoid(in, out, num_elements);
  }
  void sigmoid_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, size_t num_elements) {
    require_capability(elementwise_capability(), "elementwise",
                       "sigmoid_backward")
        ->sigmoid_backward(grad_out, out, grad_in, num_elements);
  }
  void exp(const Storage &in, Storage &out, size_t num_elements) {
    require_capability(elementwise_capability(), "elementwise", "exp")
        ->exp(in, out, num_elements);
  }
  void log(const Storage &in, Storage &out, size_t num_elements) {
    require_capability(elementwise_capability(), "elementwise", "log")
        ->log(in, out, num_elements);
  }
  void sqrt(const Storage &in, Storage &out, size_t num_elements) {
    require_capability(elementwise_capability(), "elementwise", "sqrt")
        ->sqrt(in, out, num_elements);
  }
  void rsqrt(const Storage &in, Storage &out, size_t num_elements) {
    require_capability(elementwise_capability(), "elementwise", "rsqrt")
        ->rsqrt(in, out, num_elements);
  }
  void sin(const Storage &in, Storage &out, size_t num_elements) {
    require_capability(elementwise_capability(), "elementwise", "sin")
        ->sin(in, out, num_elements);
  }
  void cos(const Storage &in, Storage &out, size_t num_elements) {
    require_capability(elementwise_capability(), "elementwise", "cos")
        ->cos(in, out, num_elements);
  }
  void softmax(const Storage &in, Storage &out, int batch_size,
               int num_classes) {
    require_capability(elementwise_capability(), "elementwise", "softmax")
        ->softmax(in, out, batch_size, num_classes);
  }
  void log_softmax(const Storage &in, Storage &out, int batch_size,
                   int num_classes) {
    require_capability(elementwise_capability(), "elementwise", "log_softmax")
        ->log_softmax(in, out, batch_size, num_classes);
  }
  void softmax_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, int batch_size, int num_classes) {
    require_capability(elementwise_capability(), "elementwise",
                       "softmax_backward")
        ->softmax_backward(grad_out, out, grad_in, batch_size, num_classes);
  }

  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) {
    require_capability(blas_capability(), "blas", "matmul")
        ->matmul(a, b, out, M, K, N, transA, transB);
  }
  void batched_matmul(const Storage &a, const Storage &b, Storage &out,
                      int batch_size, int M, int K, int N,
                      bool transA = false, bool transB = false,
                      int64_t stride_a = 0, int64_t stride_b = 0,
                      int64_t stride_c = 0) {
    require_capability(blas_capability(), "blas", "batched_matmul")
        ->batched_matmul(a, b, out, batch_size, M, K, N, transA, transB, stride_a, stride_b, stride_c);
  }



  void concat(const std::vector<Storage *> &inputs, Storage &out, int dim,
              const std::vector<Shape> &shapes) {
    require_capability(shape_capability(), "shape", "concat")
        ->concat(inputs, out, dim, shapes);
  }
  void concat_backward(const Storage &grad_out,
                       std::vector<Storage *> &grad_inputs, int dim,
                       const std::vector<Shape> &shapes) {
    require_capability(shape_capability(), "shape", "concat_backward")
        ->concat_backward(grad_out, grad_inputs, dim, shapes);
  }

  void cross_entropy(const Storage &logits, const Storage &targets,
                     Storage &out_loss, int batch_size, int num_classes,
                     int spatial) {
    require_capability(loss_capability(), "loss", "cross_entropy")
        ->cross_entropy(logits, targets, out_loss, batch_size, num_classes,
                        spatial);
  }
  void cross_entropy_backward(const Storage &grad_out, const Storage &logits,
                              const Storage &targets, Storage &grad_in,
                              int batch_size, int num_classes, int spatial) {
    require_capability(loss_capability(), "loss", "cross_entropy_backward")
        ->cross_entropy_backward(grad_out, logits, targets, grad_in, batch_size,
                                 num_classes, spatial);
  }
  void mse_loss(const Storage &pred, const Storage &target, Storage &out_loss,
                size_t num_elements) {
    require_capability(loss_capability(), "loss", "mse_loss")
        ->mse_loss(pred, target, out_loss, num_elements);
  }
  void mse_loss_backward(const Storage &grad_out, const Storage &pred,
                         const Storage &target, Storage &grad_in,
                         size_t num_elements) {
    require_capability(loss_capability(), "loss", "mse_loss_backward")
        ->mse_loss_backward(grad_out, pred, target, grad_in, num_elements);
  }

  void conv2d(const Storage &in, const Storage &weight, const Storage *bias,
              Storage &out, int B, int iC, int iH, int iW, int oC, int kH,
              int kW, int s, int p) {
    require_capability(spatial_capability(), "spatial", "conv2d")
        ->conv2d(in, weight, bias, out, B, iC, iH, iW, oC, kH, kW, s, p);
  }
  void conv2d_backward(const Storage &grad_out, const Storage &in,
                       const Storage &weight, Storage &grad_in, Storage &grad_w,
                       Storage *grad_b, int B, int iC, int iH, int iW, int oC,
                       int kH, int kW, int s, int p) {
    require_capability(spatial_capability(), "spatial", "conv2d_backward")
        ->conv2d_backward(grad_out, in, weight, grad_in, grad_w, grad_b, B, iC,
                          iH, iW, oC, kH, kW, s, p);
  }
  void max_pool2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int k, int s, int p) {
    require_capability(spatial_capability(), "spatial", "max_pool2d")
        ->max_pool2d(in, out, B, C, iH, iW, k, s, p);
  }
  void max_pool2d_backward(const Storage &grad_out, const Storage &in,
                           Storage &grad_in, int B, int C, int iH, int iW,
                           int k, int s, int p) {
    require_capability(spatial_capability(), "spatial", "max_pool2d_backward")
        ->max_pool2d_backward(grad_out, in, grad_in, B, C, iH, iW, k, s, p);
  }
  void upsample2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int scale) {
    require_capability(spatial_capability(), "spatial", "upsample2d")
        ->upsample2d(in, out, B, C, iH, iW, scale);
  }
  void upsample2d_backward(const Storage &grad_out, Storage &grad_in, int B,
                           int C, int iH, int iW, int scale) {
    require_capability(spatial_capability(), "spatial", "upsample2d_backward")
        ->upsample2d_backward(grad_out, grad_in, B, C, iH, iW, scale);
  }

  void batch_norm(const Storage &in, const Storage &scale, const Storage &bias,
                  Storage &running_mean, Storage &running_var,
                  Storage &save_mean, Storage &save_var, Storage &out, int B,
                  int C, int H, int W, float momentum, float eps,
                  bool training) {
    require_capability(normalization_capability(), "normalization",
                       "batch_norm")
        ->batch_norm(in, scale, bias, running_mean, running_var, save_mean,
                     save_var, out, B, C, H, W, momentum, eps, training);
  }
  void batch_norm_backward(const Storage &grad_out, const Storage &in,
                           const Storage &scale, const Storage &save_mean,
                           const Storage &save_var, Storage &grad_in,
                           Storage &grad_scale, Storage &grad_bias, int B,
                           int C, int H, int W, float eps) {
    require_capability(normalization_capability(), "normalization",
                       "batch_norm_backward")
        ->batch_norm_backward(grad_out, in, scale, save_mean, save_var, grad_in,
                              grad_scale, grad_bias, B, C, H, W, eps);
  }

  void adam_step(Storage &params, const Storage &grads, Storage &exp_avg,
                 Storage &exp_avg_sq, float lr, float beta1, float beta2,
                 float eps, int step, size_t num_elements) {
    require_capability(optimizer_capability(), "optimizer", "adam_step")
        ->adam_step(params, grads, exp_avg, exp_avg_sq, lr, beta1, beta2, eps,
                    step, num_elements);
  }
  void update(Storage &weight, const Storage &grad, float lr,
              size_t num_elements) {
    require_capability(optimizer_capability(), "optimizer", "update")
        ->update(weight, grad, lr, num_elements);
  }

  void fill_uniform(Storage &out, float low, float high, size_t num_elements) {
    require_capability(random_capability(), "random", "fill_uniform")
        ->fill_uniform(out, low, high, num_elements);
  }

  void sum_to_shape(const Storage &in, Storage &out, const Shape &in_shape,
                    const Shape &out_shape) {
    require_capability(reduction_capability(), "reduction", "sum_to_shape")
        ->sum_to_shape(in, out, in_shape, out_shape);
  }
  void sum(const Storage &in, Storage &out, size_t num_elements) {
    require_capability(reduction_capability(), "reduction", "sum")
        ->sum(in, out, num_elements);
  }
  void mean_last_dim(const Storage &in, Storage &out, int outer_size,
                     int dim_size) {
    require_capability(reduction_capability(), "reduction", "mean_last_dim")
        ->mean_last_dim(in, out, outer_size, dim_size);
  }
  void to_contiguous(const Storage &src, Storage &dst, const Shape &shape,
                     const Strides &strides, size_t offset) {
    require_capability(allocation_transfer_capability(), "allocation_transfer",
                       "to_contiguous")
        ->to_contiguous(src, dst, shape, strides, offset);
  }

protected:
  template <typename Capability>
  Capability *require_capability(Capability *capability,
                                 const char *capability_group,
                                 const char *operation) const {
    if (capability != nullptr) {
      return capability;
    }
    throw std::runtime_error(std::string("Backend '") + name() +
                             "' does not provide the '" + capability_group +
                             "' capability required for '" + operation + "'.");
  }

private:
  bool has_capability(BackendFeature feature) const {
    switch (feature) {
    case BackendFeature::ElementwiseBinary:
    case BackendFeature::BroadcastRow:
    case BackendFeature::UnaryActivation:
    case BackendFeature::Softmax:
      return elementwise_capability() != nullptr;
    case BackendFeature::Matmul:
      return blas_capability() != nullptr;
    case BackendFeature::Concat:
      return shape_capability() != nullptr;
    case BackendFeature::Loss:
      return loss_capability() != nullptr;
    case BackendFeature::Convolution:
    case BackendFeature::Pooling:
      return spatial_capability() != nullptr;
    case BackendFeature::BatchNorm:
      return normalization_capability() != nullptr;
    case BackendFeature::OptimizerStep:
      return optimizer_capability() != nullptr;
    case BackendFeature::RandomFill:
      return random_capability() != nullptr;
    case BackendFeature::Reduction:
      return reduction_capability() != nullptr;
    default:
      return false;
    }
  }
};

class BackendRegistry {
public:
  using BackendFactory = std::function<std::shared_ptr<Backend>(Device)>;
  using BackendDecorator =
      std::function<std::shared_ptr<Backend>(std::shared_ptr<Backend>)>;

  void register_backend(DeviceType type, BackendFactory factory);
  std::shared_ptr<Backend> get(Device device);
  void clear_cache(DeviceType type);
  void clear_all();
  void set_decorator(BackendDecorator decorator);

private:
  std::unordered_map<int, std::shared_ptr<Backend>> cache_;
  std::unordered_map<DeviceType, BackendFactory> factories_;
  BackendDecorator decorator_;
  std::mutex mutex_;
};

BackendRegistry &default_backend_registry();


struct BackendRuntimeStatus {
  std::string name;
  std::string source;
  bool discovered = false;
  bool loadable = false;
  bool active = false;
  std::string reason_code;
  std::string detail;
  std::string plugin_path;
  uint32_t plugin_abi_version = 0;
  uint32_t core_abi_version = 0;
  uint64_t capability_flags = 0;
};

class BackendManager {
public:
  using BackendFactory = BackendRegistry::BackendFactory;

  static BackendRegistry &registry();
  static void register_backend(DeviceType type, BackendFactory factory);
  static std::shared_ptr<Backend> get(Device device);
  static std::vector<std::string> list_available_backends();
  static std::vector<BackendRuntimeStatus> backend_status();
};

} // namespace munet
