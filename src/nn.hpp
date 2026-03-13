#pragma once
#include "amp.hpp"
#include "core/module.hpp"
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace munet {
namespace nn {

namespace {
inline Tensor maybe_autocast_module_input(const Tensor &x, amp::AutocastOp op) {
  if (!amp::AutocastMode::is_enabled() || !amp::should_autocast(op))
    return x;
  DataType target = amp::AutocastMode::dtype();
  if (!is_float_dtype(x.dtype()) || x.dtype() == target)
    return x;
  return x.to_dtype(target);
}

inline Tensor maybe_autocast_module_output(const Tensor &out, amp::AutocastOp op,
                                          bool autocast_active) {
  if (!autocast_active || !amp::should_autocast(op))
    return out;
  DataType target = amp::AutocastMode::dtype();
  if (!is_float_dtype(out.dtype()) || out.dtype() == target)
    return out;
  return out.to_dtype(target);
}
} // namespace

struct ScopedAutocastModeToggle {
  bool prev_enabled;
  explicit ScopedAutocastModeToggle(bool disable)
      : prev_enabled(amp::AutocastMode::is_enabled()) {
    if (disable && prev_enabled)
      amp::AutocastMode::set_enabled(false);
  }
  ~ScopedAutocastModeToggle() { amp::AutocastMode::set_enabled(prev_enabled); }
};

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
    bool autocast_active = amp::AutocastMode::is_enabled();
    ScopedAutocastModeToggle scoped(autocast_active);
    // tanh(x) = 2 * sigmoid(2x) - 1
    Tensor two({1}, x.device(), x.dtype(), false);
    two.uniform_(2.0f, 2.0f);
    Tensor one({1}, x.device(), x.dtype(), false);
    one.uniform_(1.0f, 1.0f);
    Tensor out = (x * two).sigmoid() * two - one;
    return maybe_autocast_module_output(out, amp::AutocastOp::Tanh, autocast_active);
  }
};


class GELU : public Module {
public:
  Tensor forward(Tensor x) override {
    bool autocast_active = amp::AutocastMode::is_enabled();
    ScopedAutocastModeToggle scoped(autocast_active);
    // Fast GELU approximation: x * sigmoid(1.702 * x)
    Tensor c({1}, x.device(), x.dtype(), false);
    c.uniform_(1.702f, 1.702f);
    Tensor out = x * (x * c).sigmoid();
    return maybe_autocast_module_output(out, amp::AutocastOp::GELU, autocast_active);
  }
};

class Dropout : public Module {
public:
  explicit Dropout(float p = 0.5f) : p_(p) {
    if (p_ < 0.0f || p_ >= 1.0f)
      throw std::runtime_error("Dropout probability must be in [0, 1)");
  }

  Tensor forward(Tensor x) override {
    bool autocast_active = amp::AutocastMode::is_enabled();
    ScopedAutocastModeToggle scoped(autocast_active);

    if (!training_ || p_ == 0.0f)
      return maybe_autocast_module_output(x, amp::AutocastOp::Dropout, autocast_active);

    Device cpu{DeviceType::CPU, 0};
    Tensor mask_cpu(x.shape(), cpu, x.dtype(), false);

    float keep_prob = 1.0f - p_;
    std::bernoulli_distribution keep(keep_prob);
    std::mt19937 rng(std::random_device{}());

    for (size_t i = 0; i < mask_cpu.size(); ++i) {
      double mv = keep(rng) ? (1.0 / static_cast<double>(keep_prob)) : 0.0;
      store_scalar_from_double(mask_cpu.data(), mask_cpu.dtype(), i, mv);
    }

    Tensor mask = (x.device().type == DeviceType::CPU) ? mask_cpu
                                                        : mask_cpu.to(x.device());
    Tensor out = x * mask;
    return maybe_autocast_module_output(out, amp::AutocastOp::Dropout, autocast_active);
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
    x = maybe_autocast_module_input(x, amp::AutocastOp::GlobalAvgPool2d);
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
    x = maybe_autocast_module_input(x, amp::AutocastOp::GlobalAvgPool2d);
    auto s = x.shape();
    if (s.size() != 3)
      throw std::runtime_error("MultiHeadAttention expects input shape [B,T,E]");
    if (s[2] != embed_dim_)
      throw std::runtime_error("MultiHeadAttention expects last dim == embed_dim");

    int B = s[0], T = s[1], E = s[2];
    int BH = B * num_heads_;

    // Project Q/K/V
    Tensor x2d = x.reshape({B * T, E});
    Tensor q2d = std::dynamic_pointer_cast<Linear>(q_proj)->forward(x2d);
    Tensor k2d = std::dynamic_pointer_cast<Linear>(k_proj)->forward(x2d);
    Tensor v2d = std::dynamic_pointer_cast<Linear>(v_proj)->forward(x2d);

    // [B,T,E] -> [B,H,T,D] -> [BH*T,D]
    Tensor q = q2d.reshape({B, T, num_heads_, head_dim_}).permute({0, 2, 1, 3})
                   .contiguous()
                   .reshape({BH * T, head_dim_});
    Tensor k = k2d.reshape({B, T, num_heads_, head_dim_}).permute({0, 2, 1, 3})
                   .contiguous()
                   .reshape({BH * T, head_dim_});
    Tensor v = v2d.reshape({B, T, num_heads_, head_dim_}).permute({0, 2, 1, 3})
                   .contiguous()
                   .reshape({BH * T, head_dim_});

    Tensor scores = q.matmul(k.transpose(0, 1)); // [BH*T, BH*T]
    Tensor scale({1}, scores.device(), scores.dtype(), false);
    scale.uniform_(1.0f / std::sqrt(static_cast<float>(head_dim_)),
                   1.0f / std::sqrt(static_cast<float>(head_dim_)));
    scores = scores * scale;

    // Block + causal mask on CPU, then move to target device.
    Device cpu{DeviceType::CPU, 0};
    Tensor mask_cpu({BH * T, BH * T}, cpu, scores.dtype(), false);
    float *m = static_cast<float *>(mask_cpu.data());
    for (int i = 0; i < BH * T; ++i) {
      int bh_i = i / T;
      int t_i = i % T;
      for (int j = 0; j < BH * T; ++j) {
        int bh_j = j / T;
        int t_j = j % T;
        bool masked = (bh_i != bh_j) || (causal_ && t_j > t_i);
        m[i * (BH * T) + j] = masked ? 1.0f : 0.0f;
      }
    }
    Tensor mask = (scores.device().type == DeviceType::CPU)
                      ? mask_cpu
                      : mask_cpu.to(scores.device());
    scores = scores.masked_fill(mask, -1e9f);

    Tensor probs = scores.softmax(-1);
    Tensor ctx = probs.matmul(v); // [BH*T, D]

    // [BH*T,D] -> [B,H,T,D] -> [B,T,E]
    Tensor merged = ctx.reshape({B, num_heads_, T, head_dim_})
                      .permute({0, 2, 1, 3})
                      .contiguous()
                      .reshape({B * T, E});

    Tensor out2d = std::dynamic_pointer_cast<Linear>(out_proj)->forward(merged);
    return out2d.reshape({B, T, E});
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
    x = maybe_autocast_module_input(x, amp::AutocastOp::GlobalAvgPool2d);
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
    bool autocast_active = amp::AutocastMode::is_enabled();
    ScopedAutocastModeToggle scoped(autocast_active);
    Tensor slope({1}, x.device(), x.dtype(), false);
    slope.uniform_(negative_slope_, negative_slope_);
    Tensor out = x.relu() + (x - x.relu()) * slope;
    return maybe_autocast_module_output(out, amp::AutocastOp::LeakyRelu, autocast_active);
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
