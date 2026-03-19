#pragma once

#include "module.hpp"
#include <cmath>

namespace munet {
namespace nn {

class Linear : public Module {
public:
  Linear(int in_features, int out_features, bool bias = true) {
    Tensor w({in_features, out_features}, Device{DeviceType::CPU, 0},
             DataType::Float32, true);
    float limit = 1.0f / std::sqrt((float)in_features);
    w.uniform_(-limit, limit);
    weight = w;
    register_parameter("weight", weight);

    if (bias) {
      Tensor b({out_features}, Device{DeviceType::CPU, 0}, DataType::Float32,
               true);
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
         int padding = 0)
      : stride_(stride), padding_(padding) {
    Tensor w({out_channels, in_channels, kernel_size, kernel_size},
             Device{DeviceType::CPU, 0}, DataType::Float32, true);
    float n = in_channels * kernel_size * kernel_size;
    float limit = std::sqrt(3.0f / n);
    w.uniform_(-limit, limit);
    weight = w;
    register_parameter("weight", weight);

    Tensor b({out_channels}, Device{DeviceType::CPU, 0}, DataType::Float32,
             true);
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
  Embedding(int num_embeddings, int embedding_dim)
      : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    if (num_embeddings_ <= 0 || embedding_dim_ <= 0)
      throw std::runtime_error(
          "Embedding expects positive num_embeddings and embedding_dim");

    Tensor w({num_embeddings_, embedding_dim_}, Device{DeviceType::CPU, 0},
             DataType::Float32, true);
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

      Tensor out_cpu({B, T, embedding_dim_}, cpu, x.dtype(), false);
      const float *idx = static_cast<const float *>(x_cpu.data());
      const float *wv = static_cast<const float *>(w_cpu.data());
      float *ov = static_cast<float *>(out_cpu.data());

      for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
          int token = static_cast<int>(idx[b * T + t]);
          if (token < 0 || token >= num_embeddings_)
            throw std::runtime_error("Embedding index out of range");

          const float *src = wv + token * embedding_dim_;
          float *dst = ov + (b * T + t) * embedding_dim_;
          for (int d = 0; d < embedding_dim_; ++d)
            dst[d] = src[d];
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
