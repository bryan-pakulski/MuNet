#pragma once
#include "core/module.hpp"
#include <cmath>

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
