#pragma once

#include "linear.hpp"

namespace munet {
namespace nn {

class MultiHeadAttention : public Module {
public:
  MultiHeadAttention(int embed_dim, int num_heads, bool causal = true,
                     TensorOptions options = TensorOptions{})
      : Module(options), embed_dim_(embed_dim), num_heads_(num_heads), causal_(causal) {
    if (embed_dim_ <= 0 || num_heads_ <= 0 || (embed_dim_ % num_heads_) != 0)
      throw std::runtime_error(
          "MultiHeadAttention expects embed_dim > 0, num_heads > 0 and "
          "embed_dim % num_heads == 0");

    head_dim_ = embed_dim_ / num_heads_;
    q_proj = register_module(
        "q_proj",
        std::make_shared<Linear>(embed_dim_, embed_dim_, true, options));
    k_proj = register_module(
        "k_proj",
        std::make_shared<Linear>(embed_dim_, embed_dim_, true, options));
    v_proj = register_module(
        "v_proj",
        std::make_shared<Linear>(embed_dim_, embed_dim_, true, options));
    out_proj = register_module(
        "out_proj",
        std::make_shared<Linear>(embed_dim_, embed_dim_, true, options));
  }

  Tensor forward_impl(Tensor x) override {
    auto s = x.shape();
    if (s.size() != 3)
      throw std::runtime_error("MultiHeadAttention expects input shape [B,T,E]");
    if (s[2] != embed_dim_)
      throw std::runtime_error("MultiHeadAttention expects last dim == embed_dim");

    int B = s[0], T = s[1], E = s[2];
    int BH = B * num_heads_;

    Tensor x2d = x.reshape({B * T, E});
    Tensor q2d = std::dynamic_pointer_cast<Linear>(q_proj)->forward(x2d);
    Tensor k2d = std::dynamic_pointer_cast<Linear>(k_proj)->forward(x2d);
    Tensor v2d = std::dynamic_pointer_cast<Linear>(v_proj)->forward(x2d);

    Tensor q = q2d.reshape({B, T, num_heads_, head_dim_}).permute({0, 2, 1, 3})
                   .contiguous()
                   .reshape({BH * T, head_dim_});
    Tensor k = k2d.reshape({B, T, num_heads_, head_dim_}).permute({0, 2, 1, 3})
                   .contiguous()
                   .reshape({BH * T, head_dim_});
    Tensor v = v2d.reshape({B, T, num_heads_, head_dim_}).permute({0, 2, 1, 3})
                   .contiguous()
                   .reshape({BH * T, head_dim_});

    Tensor scores = q.matmul(k.transpose(0, 1));
    if (!is_floating(scores.dtype())) {
      throw std::runtime_error(
          "MultiHeadAttention requires floating-point score tensors");
    }
    Tensor scale({1}, scores.device(), scores.dtype(), false);
    scale.fill_(1.0f / std::sqrt(static_cast<float>(head_dim_)));
    scores = scores * scale;

    Device cpu{DeviceType::CPU, 0};
    Tensor mask_cpu({BH * T, BH * T}, cpu, DataType::Int32, false);
    for (int i = 0; i < BH * T; ++i) {
      int bh_i = i / T;
      int t_i = i % T;
      for (int j = 0; j < BH * T; ++j) {
        int bh_j = j / T;
        int t_j = j % T;
        bool masked = (bh_i != bh_j) || (causal_ && t_j > t_i);
        write_scalar_to_buffer(static_cast<char *>(mask_cpu.data()) +
                                   (i * (BH * T) + j) * dtype_size(mask_cpu.dtype()),
                               mask_cpu.dtype(), masked ? 1.0 : 0.0);
      }
    }
    Tensor mask = (scores.device().type == DeviceType::CPU)
                      ? mask_cpu
                      : mask_cpu.to(scores.device());
    scores = scores.masked_fill(mask, make_scalar(-1e9, scores.dtype()));

    Tensor probs = scores.softmax(-1);
    Tensor ctx = probs.matmul(v);

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

} // namespace nn
} // namespace munet
