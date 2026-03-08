#pragma once
#include "core/module.hpp"
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace munet {
namespace nn {

class Module : public core::Module {
public:
  virtual ~Module() = default;

  std::shared_ptr<Module> register_module(std::string name,
                                          std::shared_ptr<Module> m) {
    core::Module::register_module(name, m);
    return m;
  }

  std::map<std::string, std::shared_ptr<Module>>
  named_modules_typed(std::string prefix = "") {
    std::map<std::string, std::shared_ptr<Module>> mods;
    auto base_mods = core::Module::named_modules(prefix);
    for (auto &[name, m] : base_mods) {
      auto casted = std::dynamic_pointer_cast<Module>(m);
      if (casted) {
        mods[name] = casted;
      }
    }
    return mods;
  }
};

// --- Layers ---

class Linear : public Module {
public:
  Linear(int in_features, int out_features, bool bias = true) {
    Tensor w({in_features, out_features}, Device{DeviceType::CPU, 0},
             DataType::Float32, true);
    // Kaiming Init equivalent for linear: range = sqrt(1/in_features) (simple)
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
    // x: [B, I], w: [I, O] -> [B, O]
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
    // Shape: [Out, In, K, K]
    Tensor w({out_channels, in_channels, kernel_size, kernel_size},
             Device{DeviceType::CPU, 0}, DataType::Float32, true);
    // Kaiming Init
    float n = in_channels * kernel_size * kernel_size;
    float limit = std::sqrt(3.0f / n); // Uniform kaiming
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

class ReLU : public Module {
public:
  Tensor forward(Tensor x) override { return x.relu(); }
};

class Sigmoid : public Module {
public:
  Tensor forward(Tensor x) override { return x.sigmoid(); }
};

class Tanh : public Module {
public:
  Tensor forward(Tensor x) override {
    // tanh(x) = 2 * sigmoid(2x) - 1
    Tensor two({1}, x.device(), x.dtype(), false);
    two.uniform_(2.0f, 2.0f);
    Tensor one({1}, x.device(), x.dtype(), false);
    one.uniform_(1.0f, 1.0f);
    return (x * two).sigmoid() * two - one;
  }
};


class GELU : public Module {
public:
  Tensor forward(Tensor x) override {
    // Fast GELU approximation: x * sigmoid(1.702 * x)
    Tensor c({1}, x.device(), x.dtype(), false);
    c.uniform_(1.702f, 1.702f);
    return x * (x * c).sigmoid();
  }
};

class Dropout : public Module {
public:
  explicit Dropout(float p = 0.5f) : p_(p) {
    if (p_ < 0.0f || p_ >= 1.0f)
      throw std::runtime_error("Dropout probability must be in [0, 1)");
  }

  Tensor forward(Tensor x) override {
    if (!training_ || p_ == 0.0f)
      return x;

    Device cpu{DeviceType::CPU, 0};
    Tensor mask_cpu(x.shape(), cpu, x.dtype(), false);

    float keep_prob = 1.0f - p_;
    std::bernoulli_distribution keep(keep_prob);
    std::mt19937 rng(std::random_device{}());

    float *m = static_cast<float *>(mask_cpu.data());
    for (size_t i = 0; i < mask_cpu.size(); ++i) {
      m[i] = keep(rng) ? (1.0f / keep_prob) : 0.0f;
    }

    Tensor mask = (x.device().type == DeviceType::CPU) ? mask_cpu
                                                        : mask_cpu.to(x.device());
    return x * mask;
  }

  float p_;
};



class LayerNorm : public Module {
public:
  explicit LayerNorm(int normalized_shape, float eps = 1e-5f)
      : normalized_shape_(normalized_shape), eps_(eps) {
    if (normalized_shape_ <= 0)
      throw std::runtime_error("LayerNorm expects normalized_shape > 0");

    Tensor w({normalized_shape_}, Device{DeviceType::CPU, 0},
             DataType::Float32, true);
    w.uniform_(1.0f, 1.0f);
    weight = w;
    register_parameter("weight", weight);

    Tensor b({normalized_shape_}, Device{DeviceType::CPU, 0},
             DataType::Float32, true);
    b.uniform_(0.0f, 0.0f);
    bias = b;
    register_parameter("bias", bias);
  }

  Tensor forward(Tensor x) override {
    if (x.shape().empty() || x.shape().back() != normalized_shape_)
      throw std::runtime_error("LayerNorm expects last dim to match "
                               "normalized_shape");
    return x.layer_norm(weight, bias, eps_);
  }

  Tensor weight, bias;
  int normalized_shape_;
  float eps_;
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

    // Efficient index-based gather path for [B, T] token ids.
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

    // One-hot/probability path [B, T, V] (keeps autograd support).
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


class MultiHeadAttention : public Module {
public:
  MultiHeadAttention(int embed_dim, int num_heads, bool causal = true)
      : embed_dim_(embed_dim), num_heads_(num_heads), causal_(causal) {
    if (embed_dim_ <= 0 || num_heads_ <= 0 || (embed_dim_ % num_heads_) != 0)
      throw std::runtime_error(
          "MultiHeadAttention expects embed_dim > 0, num_heads > 0 and "
          "embed_dim % num_heads == 0");

    head_dim_ = embed_dim_ / num_heads_;
    q_proj = register_module("q_proj", std::make_shared<Linear>(embed_dim_, embed_dim_));
    k_proj = register_module("k_proj", std::make_shared<Linear>(embed_dim_, embed_dim_));
    v_proj = register_module("v_proj", std::make_shared<Linear>(embed_dim_, embed_dim_));
    out_proj = register_module("out_proj", std::make_shared<Linear>(embed_dim_, embed_dim_));
  }

  Tensor forward(Tensor x) override {
    auto s = x.shape();
    if (s.size() != 3)
      throw std::runtime_error("MultiHeadAttention expects input shape [B,T,E]");
    if (s[2] != embed_dim_)
      throw std::runtime_error("MultiHeadAttention expects last dim == embed_dim");

    // NOTE: current implementation is inference-focused CPU fallback.
    // For training workloads, a dedicated backend kernel path should be added.

    int B = s[0], T = s[1], E = s[2];

    Tensor x2d = x.reshape({B * T, E});
    Tensor q2d = std::dynamic_pointer_cast<Linear>(q_proj)->forward(x2d);
    Tensor k2d = std::dynamic_pointer_cast<Linear>(k_proj)->forward(x2d);
    Tensor v2d = std::dynamic_pointer_cast<Linear>(v_proj)->forward(x2d);

    Device cpu{DeviceType::CPU, 0};
    Tensor q_cpu = q2d.to(cpu);
    Tensor k_cpu = k2d.to(cpu);
    Tensor v_cpu = v2d.to(cpu);

    const float *qv = static_cast<const float *>(q_cpu.data());
    const float *kv = static_cast<const float *>(k_cpu.data());
    const float *vv = static_cast<const float *>(v_cpu.data());

    Tensor out_cpu({B, T, E}, cpu, x.dtype(), false);
    float *ov = static_cast<float *>(out_cpu.data());

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < num_heads_; ++h) {
        const int h_off = h * head_dim_;

        for (int t = 0; t < T; ++t) {
          std::vector<float> scores(T, -std::numeric_limits<float>::infinity());
          float max_score = -std::numeric_limits<float>::infinity();

          for (int j = 0; j < T; ++j) {
            if (causal_ && j > t)
              continue;

            float dot = 0.0f;
            for (int d = 0; d < head_dim_; ++d) {
              int q_idx = ((b * T + t) * E) + h_off + d;
              int k_idx = ((b * T + j) * E) + h_off + d;
              dot += qv[q_idx] * kv[k_idx];
            }
            scores[j] = dot * scale;
            if (scores[j] > max_score)
              max_score = scores[j];
          }

          float denom = 0.0f;
          for (int j = 0; j < T; ++j) {
            if (causal_ && j > t)
              continue;
            scores[j] = std::exp(scores[j] - max_score);
            denom += scores[j];
          }
          if (denom <= 0.0f)
            denom = 1.0f;

          for (int d = 0; d < head_dim_; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < T; ++j) {
              if (causal_ && j > t)
                continue;
              float p = scores[j] / denom;
              int v_idx = ((b * T + j) * E) + h_off + d;
              acc += p * vv[v_idx];
            }
            int o_idx = ((b * T + t) * E) + h_off + d;
            ov[o_idx] = acc;
          }
        }
      }
    }

    Tensor attn = (x.device().type == DeviceType::CPU) ? out_cpu
                                                        : out_cpu.to(x.device());
    Tensor out2d = attn.reshape({B * T, E});
    Tensor proj = std::dynamic_pointer_cast<Linear>(out_proj)->forward(out2d);
    return proj.reshape({B, T, E});
  }

  int embed_dim_;
  int num_heads_;
  int head_dim_;
  bool causal_;
  std::shared_ptr<Module> q_proj, k_proj, v_proj, out_proj;
};

class GlobalAvgPool2d : public Module {
public:
  Tensor forward(Tensor x) override {
    auto s = x.shape();
    if (s.size() != 4)
      throw std::runtime_error("GlobalAvgPool2d expects NCHW input");

    int B = s[0], C = s[1], H = s[2], W = s[3];
    int HW = H * W;

    Tensor flat = x.reshape({B * C, HW});
    Tensor weights({HW, 1}, x.device(), x.dtype(), false);
    float scale = 1.0f / static_cast<float>(HW);
    weights.uniform_(scale, scale);

    Tensor out = flat.matmul(weights);
    return out.reshape({B, C, 1, 1});
  }
};

class LeakyReLU : public Module {
public:
  explicit LeakyReLU(float negative_slope = 0.01f)
      : negative_slope_(negative_slope) {}

  Tensor forward(Tensor x) override {
    Tensor slope({1}, x.device(), x.dtype(), false);
    slope.uniform_(negative_slope_, negative_slope_);
    return x.relu() + (x - x.relu()) * slope;
  }

  float negative_slope_;
};

class MaxPool2d : public Module {
public:
  MaxPool2d(int kernel_size, int stride = 2, int padding = 0)
      : k_(kernel_size), s_(stride), p_(padding) {}
  Tensor forward(Tensor x) override { return x.max_pool2d(k_, s_, p_); }

  int k_, s_, p_;
};

class Upsample : public Module {
public:
  Upsample(int scale_factor) : scale_(scale_factor) {}
  Tensor forward(Tensor x) override { return x.upsample2d(scale_); }

  int scale_;
};

class BatchNorm2d : public Module {
public:
  BatchNorm2d(int num_features, float eps = 1e-5f, float momentum = 0.1f)
      : eps_(eps), momentum_(momentum) {
    Tensor w({num_features}, Device{DeviceType::CPU, 0}, DataType::Float32,
             true);
    w.uniform_(1.0f, 1.0f); // Init to 1
    weight = w;
    register_parameter("weight", weight);

    Tensor b({num_features}, Device{DeviceType::CPU, 0}, DataType::Float32,
             true);
    b.uniform_(0.0f, 0.0f); // Init to 0
    bias = b;
    register_parameter("bias", bias);

    Tensor rm({num_features}, Device{DeviceType::CPU, 0}, DataType::Float32,
              false);
    rm.uniform_(0.0f, 0.0f);
    running_mean = rm;
    register_buffer("running_mean", running_mean);

    Tensor rv({num_features}, Device{DeviceType::CPU, 0}, DataType::Float32,
              false);
    rv.uniform_(1.0f, 1.0f);
    running_var = rv;
    register_buffer("running_var", running_var);
  }

  Tensor forward(Tensor x) override {
    return x.batch_norm(running_mean, running_var, weight, bias, training_,
                        momentum_, eps_);
  }

  Tensor weight, bias, running_mean, running_var;
  float eps_, momentum_;
};

class Sequential : public Module {
public:
  Sequential() = default;

  void add(std::shared_ptr<Module> m) {
    register_module(std::to_string(modules_.size()), m);
    ordered_modules_.push_back(m);
  }

  Tensor forward(Tensor x) override {
    for (auto &m : ordered_modules_) {
      x = m->forward(x);
    }
    return x;
  }

  std::vector<std::shared_ptr<Module>> ordered_modules_;
};

} // namespace nn
} // namespace munet
