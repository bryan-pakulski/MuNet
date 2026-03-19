#pragma once

#include "module.hpp"

namespace munet {
namespace nn {

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

} // namespace nn
} // namespace munet
