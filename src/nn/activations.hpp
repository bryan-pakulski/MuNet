#pragma once

#include "module.hpp"
#include <random>
#include <stdexcept>

namespace munet {
namespace nn {

class ReLU : public Module {
public:
  Tensor forward_impl(Tensor x) override { return x.relu(); }
};

class Sigmoid : public Module {
public:
  Tensor forward_impl(Tensor x) override { return x.sigmoid(); }
};

class Tanh : public Module {
public:
  Tensor forward_impl(Tensor x) override {
    Tensor two({1}, x.device(), x.dtype(), false);
    two.fill_(2.0f);
    Tensor one({1}, x.device(), x.dtype(), false);
    one.fill_(1.0f);
    return (x * two).sigmoid() * two - one;
  }
};

class GELU : public Module {
public:
  Tensor forward_impl(Tensor x) override {
    Tensor c({1}, x.device(), x.dtype(), false);
    c.fill_(1.702f);
    return x * (x * c).sigmoid();
  }
};

class Dropout : public Module {
public:
  explicit Dropout(float p = 0.5f) : p_(p) {
    if (p_ < 0.0f || p_ >= 1.0f)
      throw std::runtime_error("Dropout probability must be in [0, 1)");
  }

  Tensor forward_impl(Tensor x) override {
    if (!training_ || p_ == 0.0f)
      return x;
    if (!is_floating(x.dtype()))
      throw std::runtime_error("Dropout requires floating-point tensors");

    Device cpu{DeviceType::CPU, 0};
    Tensor mask_cpu(x.shape(), cpu, x.dtype(), false);

    float keep_prob = 1.0f - p_;
    std::bernoulli_distribution keep(keep_prob);
    std::mt19937 rng(std::random_device{}());

    for (size_t i = 0; i < mask_cpu.size(); ++i) {
      write_scalar_to_buffer(static_cast<char *>(mask_cpu.data()) +
                                 i * dtype_size(mask_cpu.dtype()),
                             mask_cpu.dtype(),
                             keep(rng) ? (1.0 / keep_prob) : 0.0);
    }

    Tensor mask = (x.device().type == DeviceType::CPU)
                      ? mask_cpu
                      : mask_cpu.to(x.device());
    return x * mask;
  }

  float p_;
};

class LeakyReLU : public Module {
public:
  explicit LeakyReLU(float negative_slope = 0.01f)
      : negative_slope_(negative_slope) {}

  Tensor forward_impl(Tensor x) override {
    Tensor slope({1}, x.device(), x.dtype(), false);
    slope.fill_(negative_slope_);
    return x.relu() + (x - x.relu()) * slope;
  }

  float negative_slope_;
};

class Softmax : public Module {
public:
  explicit Softmax(int dim = -1) : dim_(dim) {}

  Tensor forward_impl(Tensor x) override {
    return x.softmax(dim_);
  }

  int dim_;
};

} // namespace nn
} // namespace munet
