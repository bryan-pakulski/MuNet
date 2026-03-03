#pragma once
#include "layer.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace munet {

class Flatten : public Layer {
public:
  inline Tensor forward(const Tensor &input) override;
  inline Tensor backward(const Tensor &grad_output) override;
  inline std::string get_onnx_op_type() const override { return "Flatten"; }

private:
  std::vector<int> input_shape_;
};

inline Tensor Flatten::forward(const Tensor &input) {
  input_shape_ = input.shape();
  int batch_size = input_shape_.empty() ? 1 : input_shape_[0];
  int flat_size = 1;
  for (size_t i = 1; i < input_shape_.size(); ++i)
    flat_size *= input_shape_[i];

  Tensor output({batch_size, flat_size}, input.device_, input.dtype_);

#ifdef MUNET_USE_CUDA
  if (input.device_ == Device::CUDA) {
    cudaMemcpy(output.data(), input.data(), input.bytes(),
               cudaMemcpyDeviceToDevice);
    return output;
  }
#endif
  std::memcpy(output.data(), input.data(), input.bytes());
  return output;
}

inline Tensor Flatten::backward(const Tensor &grad_output) {
  Tensor grad_input(input_shape_, grad_output.device_, grad_output.dtype_);
#ifdef MUNET_USE_CUDA
  if (grad_output.device_ == Device::CUDA) {
    cudaMemcpy(grad_input.data(), grad_output.data(), grad_output.bytes(),
               cudaMemcpyDeviceToDevice);
    return grad_input;
  }
#endif
  std::memcpy(grad_input.data(), grad_output.data(), grad_output.bytes());
  return grad_input;
}

class MaxPool2D : public Layer {
public:
  inline MaxPool2D(int kernel_size, int stride);
  inline Tensor forward(const Tensor &input) override;
  inline Tensor backward(const Tensor &grad_output) override;
  inline std::string get_onnx_op_type() const override { return "MaxPool"; }

private:
  int kernel_size_;
  int stride_;
  std::vector<int> input_shape_;
  Tensor max_indices_{
      std::vector<int>{}}; // Changed from std::vector to Tensor for GPU support
};

inline MaxPool2D::MaxPool2D(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {}

inline Tensor MaxPool2D::forward(const Tensor &input) {
  input_shape_ = input.shape();
  int N = input_shape_[0], C = input_shape_[1], H = input_shape_[2],
      W = input_shape_[3];
  int OH = (H - kernel_size_) / stride_ + 1;
  int OW = (W - kernel_size_) / stride_ + 1;

  Tensor output({N, C, OH, OW}, input.device_, input.dtype_);
  // Allocate indices on same device. Treat as FP32 size-wise (4 bytes per
  // index)
  max_indices_ = Tensor({(int)output.size()}, input.device_, DataType::FP32);

  const float *in_ptr = static_cast<const float *>(input.data());
  float *out_ptr = static_cast<float *>(output.data());
  int *idx_ptr = static_cast<int *>(max_indices_.data());

#ifdef MUNET_USE_CUDA
  if (input.device_ == Device::CUDA) {
    cuda_kernels::maxpool_forward(in_ptr, out_ptr, idx_ptr, N, C, H, W, OH, OW,
                                  kernel_size_, stride_);
    return output;
  }
#endif

  for (int b = 0; b < N; ++b) {
    for (int c = 0; c < C; ++c) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float max_val = std::numeric_limits<float>::lowest();
          int max_idx = -1;

          for (int kh = 0; kh < kernel_size_; ++kh) {
            for (int kw = 0; kw < kernel_size_; ++kw) {
              int ih = oh * stride_ + kh;
              int iw = ow * stride_ + kw;
              int in_idx = b * (C * H * W) + c * (H * W) + ih * W + iw;

              if (in_ptr[in_idx] > max_val) {
                max_val = in_ptr[in_idx];
                max_idx = in_idx;
              }
            }
          }
          int out_idx = b * (C * OH * OW) + c * (OH * OW) + oh * OW + ow;
          out_ptr[out_idx] = max_val;
          idx_ptr[out_idx] = max_idx;
        }
      }
    }
  }
  return output;
}

inline Tensor MaxPool2D::backward(const Tensor &grad_output) {
  Tensor grad_input(input_shape_, grad_output.device_, grad_output.dtype_);
  grad_input.zero();

  const float *go_ptr = static_cast<const float *>(grad_output.data());
  float *gi_ptr = static_cast<float *>(grad_input.data());
  const int *idx_ptr = static_cast<const int *>(max_indices_.data());

#ifdef MUNET_USE_CUDA
  if (grad_output.device_ == Device::CUDA) {
    cuda_kernels::maxpool_backward(go_ptr, gi_ptr, idx_ptr, grad_output.size());
    return grad_input;
  }
#endif

  for (size_t i = 0; i < grad_output.size(); ++i) {
    int in_idx = idx_ptr[i];
    if (in_idx != -1) {
      gi_ptr[in_idx] += go_ptr[i];
    }
  }
  return grad_input;
}

class Conv2D : public Layer {
public:
  inline Conv2D(int in_channels, int out_channels, int kernel_size,
                int stride = 1, int padding = 0);
  inline Tensor forward(const Tensor &input) override;
  inline Tensor backward(const Tensor &grad_output) override;

  inline std::unordered_map<std::string, Tensor *> get_parameters() override {
    return {{"weight", &weight_}, {"bias", &bias_}};
  }
  inline std::unordered_map<std::string, Tensor *> get_gradients() override {
    return {{"weight", weight_.grad()}, {"bias", bias_.grad()}};
  }
  inline std::string get_onnx_op_type() const override { return "Conv"; }

private:
  int in_channels_, out_channels_, kernel_size_, stride_, padding_;
  Tensor weight_, bias_;
  Tensor input_cache_{std::vector<int>{}};
};

inline Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size,
                      int stride, int padding)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      weight_({out_channels, in_channels, kernel_size, kernel_size}),
      bias_({out_channels}) {
  float scale = std::sqrt(2.0f / (in_channels_ * kernel_size_ * kernel_size_));
  float *w_ptr = static_cast<float *>(weight_.data());
  for (size_t i = 0; i < weight_.size(); ++i) {
    float r = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
    w_ptr[i] = r * scale;
  }

  bias_.zero();
  weight_.allocate_grad();
  bias_.allocate_grad();
}

inline Tensor Conv2D::forward(const Tensor &input) {
  input_cache_ = input.clone();
  int N = input.shape()[0], C = input.shape()[1], H = input.shape()[2],
      W = input.shape()[3];
  int OH = (H + 2 * padding_ - kernel_size_) / stride_ + 1;
  int OW = (W + 2 * padding_ - kernel_size_) / stride_ + 1;

  Tensor output({N, out_channels_, OH, OW}, input.device_, input.dtype_);

  const float *in_ptr = static_cast<const float *>(input.data());
  const float *w_ptr = static_cast<const float *>(weight_.data());
  const float *b_ptr = static_cast<const float *>(bias_.data());
  float *out_ptr = static_cast<float *>(output.data());

#ifdef MUNET_USE_CUDA
  if (input.device_ == Device::CUDA) {
    cuda_kernels::conv2d_forward(in_ptr, w_ptr, b_ptr, out_ptr, N, C,
                                 out_channels_, H, W, OH, OW, kernel_size_,
                                 stride_, padding_);
    return output;
  }
#endif

  for (int b = 0; b < N; ++b) {
    for (int oc = 0; oc < out_channels_; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float sum = b_ptr[oc];

          for (int ic = 0; ic < in_channels_; ++ic) {
            for (int kh = 0; kh < kernel_size_; ++kh) {
              for (int kw = 0; kw < kernel_size_; ++kw) {
                int ih = oh * stride_ - padding_ + kh;
                int iw = ow * stride_ - padding_ + kw;

                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                  int in_idx = b * (C * H * W) + ic * (H * W) + ih * W + iw;
                  int w_idx = oc * (C * kernel_size_ * kernel_size_) +
                              ic * (kernel_size_ * kernel_size_) +
                              kh * kernel_size_ + kw;
                  sum += in_ptr[in_idx] * w_ptr[w_idx];
                }
              }
            }
          }
          out_ptr[b * (out_channels_ * OH * OW) + oc * (OH * OW) + oh * OW +
                  ow] = sum;
        }
      }
    }
  }
  return output;
}

inline Tensor Conv2D::backward(const Tensor &grad_output) {
  int N = input_cache_.shape()[0], C = input_cache_.shape()[1];
  int H = input_cache_.shape()[2], W = input_cache_.shape()[3];
  int OH = grad_output.shape()[2], OW = grad_output.shape()[3];

  Tensor grad_input(input_cache_.shape(), input_cache_.device_,
                    input_cache_.dtype_);
  grad_input.zero();

  const float *go_ptr = static_cast<const float *>(grad_output.data());
  const float *in_ptr = static_cast<const float *>(input_cache_.data());
  const float *w_ptr = static_cast<const float *>(weight_.data());

  float *gi_ptr = static_cast<float *>(grad_input.data());
  float *gw_ptr = static_cast<float *>(weight_.grad()->data());
  float *gb_ptr = static_cast<float *>(bias_.grad()->data());

#ifdef MUNET_USE_CUDA
  if (grad_output.device_ == Device::CUDA) {
    cuda_kernels::conv2d_backward(go_ptr, in_ptr, w_ptr, gi_ptr, gw_ptr, gb_ptr,
                                  N, C, out_channels_, H, W, OH, OW,
                                  kernel_size_, stride_, padding_);
    return grad_input;
  }
#endif

  for (int b = 0; b < N; ++b) {
    for (int oc = 0; oc < out_channels_; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float go_val = go_ptr[b * (out_channels_ * OH * OW) + oc * (OH * OW) +
                                oh * OW + ow];
          gb_ptr[oc] += go_val; // d_bias

          for (int ic = 0; ic < in_channels_; ++ic) {
            for (int kh = 0; kh < kernel_size_; ++kh) {
              for (int kw = 0; kw < kernel_size_; ++kw) {
                int ih = oh * stride_ - padding_ + kh;
                int iw = ow * stride_ - padding_ + kw;

                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                  int in_idx = b * (C * H * W) + ic * (H * W) + ih * W + iw;
                  int w_idx = oc * (C * kernel_size_ * kernel_size_) +
                              ic * (kernel_size_ * kernel_size_) +
                              kh * kernel_size_ + kw;

                  gw_ptr[w_idx] += in_ptr[in_idx] * go_val; // d_weight
                  gi_ptr[in_idx] += w_ptr[w_idx] * go_val;  // d_input
                }
              }
            }
          }
        }
      }
    }
  }
  return grad_input;
}

class BatchNorm2D : public Layer {
public:
  inline BatchNorm2D(int num_features, float eps = 1e-5f,
                     float momentum = 0.1f);
  inline Tensor forward(const Tensor &input) override;
  inline Tensor backward(const Tensor &grad_output) override;

  inline std::unordered_map<std::string, Tensor *> get_parameters() override {
    return {{"weight", &weight_},
            {"bias", &bias_},
            {"running_mean", &running_mean_},
            {"running_var", &running_var_}};
  }
  inline std::unordered_map<std::string, Tensor *> get_gradients() override {
    return {{"weight", weight_.grad()}, {"bias", bias_.grad()}};
  }
  inline std::string get_onnx_op_type() const override {
    return "BatchNormalization";
  }

  inline void train() { training_ = true; }
  inline void eval() { training_ = false; }

private:
  int num_features_;
  float eps_;
  float momentum_;
  bool training_{true};

  Tensor weight_, bias_;
  Tensor running_mean_, running_var_;

  Tensor save_mean_{std::vector<int>{}};
  Tensor save_var_{std::vector<int>{}};
  Tensor save_inv_std_{std::vector<int>{}};
  Tensor input_cache_{std::vector<int>{}};
};

inline BatchNorm2D::BatchNorm2D(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum),
      weight_({num_features}), bias_({num_features}),
      running_mean_({num_features}), running_var_({num_features}) {
  float *w_ptr = static_cast<float *>(weight_.data());
  float *b_ptr = static_cast<float *>(bias_.data());
  float *rm_ptr = static_cast<float *>(running_mean_.data());
  float *rv_ptr = static_cast<float *>(running_var_.data());

  for (int i = 0; i < num_features; ++i) {
    w_ptr[i] = 1.0f;
    b_ptr[i] = 0.0f;
    rm_ptr[i] = 0.0f;
    rv_ptr[i] = 1.0f;
  }

  weight_.allocate_grad();
  bias_.allocate_grad();
  // Do NOT allocate grad for running stats so SGD ignores them
}

inline Tensor BatchNorm2D::forward(const Tensor &input) {
  input_cache_ = input.clone();
  int N = input.shape()[0], C = input.shape()[1], H = input.shape()[2],
      W = input.shape()[3];

  Tensor output(input.shape(), input.device_, input.dtype_);

  if (training_) {
    save_mean_ = Tensor({C}, input.device_, input.dtype_);
    save_var_ = Tensor({C}, input.device_, input.dtype_);
    save_inv_std_ = Tensor({C}, input.device_, input.dtype_);
  }

  const float *in_ptr = static_cast<const float *>(input.data());
  float *out_ptr = static_cast<float *>(output.data());
  float *rm_ptr = static_cast<float *>(running_mean_.data());
  float *rv_ptr = static_cast<float *>(running_var_.data());
  const float *w_ptr = static_cast<const float *>(weight_.data());
  const float *b_ptr = static_cast<const float *>(bias_.data());

#ifdef MUNET_USE_CUDA
  if (input.device_ == Device::CUDA) {
    float *sm_ptr =
        training_ ? static_cast<float *>(save_mean_.data()) : nullptr;
    float *sv_ptr =
        training_ ? static_cast<float *>(save_var_.data()) : nullptr;
    float *sis_ptr =
        training_ ? static_cast<float *>(save_inv_std_.data()) : nullptr;

    cuda_kernels::batchnorm2d_forward(in_ptr, out_ptr, sm_ptr, sv_ptr, sis_ptr,
                                      rm_ptr, rv_ptr, w_ptr, b_ptr, N, C, H, W,
                                      eps_, momentum_, training_);
    return output;
  }
#endif

  int M = N * H * W;
  if (training_) {
    float *sm_ptr = static_cast<float *>(save_mean_.data());
    float *sv_ptr = static_cast<float *>(save_var_.data());
    float *sis_ptr = static_cast<float *>(save_inv_std_.data());

    for (int c = 0; c < C; ++c) {
      float sum = 0.0f;
      for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w)
            sum += in_ptr[((n * C + c) * H + h) * W + w];
        }
      }
      float mean = sum / M;

      float var_sum = 0.0f;
      for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            float diff = in_ptr[((n * C + c) * H + h) * W + w] - mean;
            var_sum += diff * diff;
          }
        }
      }
      float var = var_sum / M;
      float inv_std = 1.0f / std::sqrt(var + eps_);

      sm_ptr[c] = mean;
      sv_ptr[c] = var;
      sis_ptr[c] = inv_std;

      rm_ptr[c] = (1.0f - momentum_) * rm_ptr[c] + momentum_ * mean;
      rv_ptr[c] = (1.0f - momentum_) * rv_ptr[c] +
                  momentum_ * (var * M / (M > 1 ? M - 1 : 1));

      for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            int idx = ((n * C + c) * H + h) * W + w;
            out_ptr[idx] = (in_ptr[idx] - mean) * inv_std * w_ptr[c] + b_ptr[c];
          }
        }
      }
    }
  } else {
    for (int c = 0; c < C; ++c) {
      float inv_std = 1.0f / std::sqrt(rv_ptr[c] + eps_);
      for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            int idx = ((n * C + c) * H + h) * W + w;
            out_ptr[idx] =
                (in_ptr[idx] - rm_ptr[c]) * inv_std * w_ptr[c] + b_ptr[c];
          }
        }
      }
    }
  }
  return output;
}

inline Tensor BatchNorm2D::backward(const Tensor &grad_output) {
  if (!training_)
    throw std::runtime_error("Cannot backward BatchNorm in eval mode");

  int N = input_cache_.shape()[0], C = input_cache_.shape()[1];
  int H = input_cache_.shape()[2], W = input_cache_.shape()[3];
  int M = N * H * W;

  Tensor grad_input(input_cache_.shape(), input_cache_.device_,
                    input_cache_.dtype_);
  grad_input.zero();

  const float *go_ptr = static_cast<const float *>(grad_output.data());
  const float *in_ptr = static_cast<const float *>(input_cache_.data());
  float *gi_ptr = static_cast<float *>(grad_input.data());

  const float *sm_ptr = static_cast<const float *>(save_mean_.data());
  const float *sis_ptr = static_cast<const float *>(save_inv_std_.data());
  const float *w_ptr = static_cast<const float *>(weight_.data());

  float *gw_ptr = static_cast<float *>(weight_.grad()->data());
  float *gb_ptr = static_cast<float *>(bias_.grad()->data());

#ifdef MUNET_USE_CUDA
  if (grad_output.device_ == Device::CUDA) {
    cuda_kernels::batchnorm2d_backward(go_ptr, in_ptr, gi_ptr, gw_ptr, gb_ptr,
                                       sm_ptr, sis_ptr, w_ptr, N, C, H, W);
    return grad_input;
  }
#endif

  for (int c = 0; c < C; ++c) {
    float sum_go = 0.0f;
    float sum_go_xhat = 0.0f;
    float mean = sm_ptr[c];
    float inv_std = sis_ptr[c];
    float w_c = w_ptr[c];

    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          int idx = ((n * C + c) * H + h) * W + w;
          float go_val = go_ptr[idx];
          float xhat = (in_ptr[idx] - mean) * inv_std;
          sum_go += go_val;
          sum_go_xhat += go_val * xhat;
        }
      }
    }

    gw_ptr[c] += sum_go_xhat;
    gb_ptr[c] += sum_go;

    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          int idx = ((n * C + c) * H + h) * W + w;
          float go_val = go_ptr[idx];
          float xhat = (in_ptr[idx] - mean) * inv_std;
          gi_ptr[idx] =
              (w_c * inv_std / M) * (M * go_val - sum_go - xhat * sum_go_xhat);
        }
      }
    }
  }
  return grad_input;
}

class Upsample2D : public Layer {
public:
  inline Upsample2D(int scale_factor) : scale_factor_(scale_factor) {}
  inline Tensor forward(const Tensor &input) override;
  inline Tensor backward(const Tensor &grad_output) override;
  inline std::string get_onnx_op_type() const override { return "Upsample"; }

private:
  int scale_factor_;
  std::vector<int> input_shape_;
};

inline Tensor Upsample2D::forward(const Tensor &input) {
  input_shape_ = input.shape();
  int N = input_shape_[0], C = input_shape_[1], H = input_shape_[2],
      W = input_shape_[3];
  int OH = H * scale_factor_;
  int OW = W * scale_factor_;

  Tensor output({N, C, OH, OW}, input.device_, input.dtype_);
  const float *in_ptr = static_cast<const float *>(input.data());
  float *out_ptr = static_cast<float *>(output.data());

#ifdef MUNET_USE_CUDA
  if (input.device_ == Device::CUDA) {
    cuda_kernels::upsample_forward(in_ptr, out_ptr, N, C, H, W, OH, OW,
                                   scale_factor_);
    return output;
  }
#endif

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          int ih = oh / scale_factor_;
          int iw = ow / scale_factor_;
          out_ptr[n * (C * OH * OW) + c * (OH * OW) + oh * OW + ow] =
              in_ptr[n * (C * H * W) + c * (H * W) + ih * W + iw];
        }
      }
    }
  }
  return output;
}

inline Tensor Upsample2D::backward(const Tensor &grad_output) {
  Tensor grad_input(input_shape_, grad_output.device_, grad_output.dtype_);
  grad_input.zero();

  int N = input_shape_[0], C = input_shape_[1], H = input_shape_[2],
      W = input_shape_[3];
  int OH = grad_output.shape()[2], OW = grad_output.shape()[3];

  const float *go_ptr = static_cast<const float *>(grad_output.data());
  float *gi_ptr = static_cast<float *>(grad_input.data());

#ifdef MUNET_USE_CUDA
  if (grad_output.device_ == Device::CUDA) {
    cuda_kernels::upsample_backward(go_ptr, gi_ptr, N, C, H, W, OH, OW,
                                    scale_factor_);
    return grad_input;
  }
#endif

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          int ih = oh / scale_factor_;
          int iw = ow / scale_factor_;
          gi_ptr[n * (C * H * W) + c * (H * W) + ih * W + iw] +=
              go_ptr[n * (C * OH * OW) + c * (OH * OW) + oh * OW + ow];
        }
      }
    }
  }
  return grad_input;
}
} // namespace munet
