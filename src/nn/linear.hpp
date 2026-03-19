#pragma once

#include "module.hpp"
#include <cmath>

namespace munet {
namespace nn {

class Linear : public Module {
public:
  Linear(int in_features, int out_features, bool bias = true,
         TensorOptions options = TensorOptions{}) {
    Tensor w({in_features, out_features}, parameter_options(options));
    float limit = 1.0f / std::sqrt((float)in_features);
    w.uniform_(-limit, limit);
    weight = w;
    register_parameter("weight", weight);

    if (bias) {
      Tensor b({out_features}, parameter_options(options));
      b.uniform_(-limit, limit);
      this->bias = b;
      register_parameter("bias", this->bias);
    }
  }

  Tensor forward(Tensor x) override {
    Tensor out = x.matmul(weight);
    if (bias.size() > 0) {
      out = out + bias;
    }
    return out;
  }

  Tensor weight, bias;
};

class Conv2d : public Module {
public:
  Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1,
         int padding = 0, TensorOptions options = TensorOptions{})
      : stride_(stride), padding_(padding) {
    Tensor w({out_channels, in_channels, kernel_size, kernel_size},
             parameter_options(options));
    float n = in_channels * kernel_size * kernel_size;
    float limit = std::sqrt(3.0f / n);
    w.uniform_(-limit, limit);
    weight = w;
    register_parameter("weight", weight);

    Tensor b({out_channels}, parameter_options(options));
    b.uniform_(-limit, limit);
    bias = b;
    register_parameter("bias", bias);
  }

  Tensor forward(Tensor x) override {
    return x.conv2d(weight, bias, stride_, padding_);
  }

  Tensor weight, bias;
  int stride_, padding_;
};

class Flatten : public Module {
public:
  Tensor forward(Tensor x) override {
    int batch_size = x.shape()[0];
    size_t total = x.size();
    return x.reshape({batch_size, (int)(total / batch_size)});
  }
};

class Embedding : public Module {
public:
  Embedding(int num_embeddings, int embedding_dim,
            TensorOptions options = TensorOptions{})
      : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    if (num_embeddings_ <= 0 || embedding_dim_ <= 0)
      throw std::runtime_error(
          "Embedding expects positive num_embeddings and embedding_dim");

    Tensor w({num_embeddings_, embedding_dim_}, parameter_options(options));
    float limit = 1.0f / std::sqrt((float)embedding_dim_);
    w.uniform_(-limit, limit);
    weight = w;
    register_parameter("weight", weight);
  }

  Tensor forward(Tensor x) override {
    auto s = x.shape();

    if (s.size() == 2 && !weight.requires_grad() && !x.requires_grad()) {
      Device cpu{DeviceType::CPU, 0};
      Tensor x_cpu = x.to(cpu);
      Tensor w_cpu = weight.to(cpu);
      int B = s[0], T = s[1];

      Tensor out_cpu({B, T, embedding_dim_}, cpu, weight.dtype(), false);
      const char *idx = static_cast<const char *>(x_cpu.data());
      const char *wv = static_cast<const char *>(w_cpu.data());
      char *ov = static_cast<char *>(out_cpu.data());
      const size_t idx_stride = dtype_size(x_cpu.dtype());
      const size_t weight_stride = dtype_size(w_cpu.dtype());
      const size_t out_stride = dtype_size(out_cpu.dtype());

      for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
          int token = static_cast<int>(read_scalar_from_buffer(
                                           idx + (b * T + t) * idx_stride,
                                           x_cpu.dtype())
                                           .value);
          if (token < 0 || token >= num_embeddings_)
            throw std::runtime_error("Embedding index out of range");

          const char *src = wv + token * embedding_dim_ * weight_stride;
          char *dst = ov + (b * T + t) * embedding_dim_ * out_stride;
          for (int d = 0; d < embedding_dim_; ++d) {
            const ScalarValue value =
                read_scalar_from_buffer(src + d * weight_stride, w_cpu.dtype());
            write_scalar_to_buffer(dst + d * out_stride, out_cpu.dtype(),
                                   value.value);
          }
        }
      }

      return (x.device().type == DeviceType::CPU) ? out_cpu
                                                   : out_cpu.to(x.device());
    }

    if (s.size() != 3)
      throw std::runtime_error(
          "Embedding expects [B,T] indices or [B,T,V] one-hot/probabilities");
    if (s[2] != num_embeddings_)
      throw std::runtime_error("Embedding input vocab dimension must match "
                               "num_embeddings");

    int B = s[0], T = s[1], V = s[2];
    Tensor flat = x.reshape({B * T, V});
    Tensor out = flat.matmul(weight);
    return out.reshape({B, T, embedding_dim_});
  }

  Tensor weight;
  int num_embeddings_;
  int embedding_dim_;
};

} // namespace nn
} // namespace munet
