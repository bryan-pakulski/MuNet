#pragma once

#include "module.hpp"
#include <random>
#include <stdexcept>

namespace munet {
namespace nn {

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

} // namespace nn
} // namespace munet
