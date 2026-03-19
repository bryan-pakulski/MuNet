#pragma once

#include "module.hpp"

namespace munet {
namespace nn {

class LayerNorm : public Module {
public:
  explicit LayerNorm(int normalized_shape, float eps = 1e-5f,
                     TensorOptions options = TensorOptions{})
      : normalized_shape_(normalized_shape), eps_(eps) {
    if (normalized_shape_ <= 0)
      throw std::runtime_error("LayerNorm expects normalized_shape > 0");

    Tensor w({normalized_shape_}, parameter_options(options));
    w.fill_(1.0f);
    weight = w;
    register_parameter("weight", weight);

    Tensor b({normalized_shape_}, parameter_options(options));
    b.fill_(0.0f);
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

class BatchNorm2d : public Module {
public:
  BatchNorm2d(int num_features, float eps = 1e-5f, float momentum = 0.1f,
              TensorOptions options = TensorOptions{})
      : eps_(eps), momentum_(momentum) {
    Tensor w({num_features}, parameter_options(options));
    w.fill_(1.0f);
    weight = w;
    register_parameter("weight", weight);

    Tensor b({num_features}, parameter_options(options));
    b.fill_(0.0f);
    bias = b;
    register_parameter("bias", bias);

    const DataType stats_dtype =
        accumulation_type(AccumulationOp::Normalization, options.dtype);
    Tensor rm({num_features}, buffer_options(options, stats_dtype));
    rm.fill_(0.0f);
    running_mean = rm;
    register_buffer("running_mean", running_mean);

    Tensor rv({num_features}, buffer_options(options, stats_dtype));
    rv.fill_(1.0f);
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

} // namespace nn
} // namespace munet
