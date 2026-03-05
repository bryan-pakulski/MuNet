#pragma once
#include "tensor.hpp"
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace munet {
namespace nn {

class Module {
public:
  virtual ~Module() = default;

  // Generic forward for single-tensor layers
  virtual Tensor forward(Tensor x) = 0;

  virtual std::vector<Tensor> parameters() {
    std::vector<Tensor> params;
    for (auto &[name, p] : parameters_) {
      params.push_back(p);
    }
    for (auto &[name, m] : modules_) {
      auto sub_params = m->parameters();
      params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
  }

  virtual void train(bool mode = true) {
    training_ = mode;
    for (auto &[name, m] : modules_) {
      m->train(mode);
    }
  }

  void eval() { train(false); }

  virtual void to(Device device) {
    for (auto &[name, p] : parameters_) {
      parameters_[name] = p.to(device);
      if (p.requires_grad()) {
        parameters_[name].set_requires_grad(true);
      }
    }
    for (auto &[name, b] : buffers_) {
      buffers_[name] = b.to(device);
    }
    for (auto &[name, m] : modules_) {
      m->to(device);
    }
  }

  void zero_grad() {
    for (auto &p : parameters()) {
      p.zero_grad();
    }
  }

protected:
  Tensor &register_parameter(std::string name, Tensor t) {
    parameters_[name] = t;
    return parameters_[name];
  }

  Tensor &register_buffer(std::string name, Tensor t) {
    buffers_[name] = t;
    return buffers_[name];
  }

  std::shared_ptr<Module> register_module(std::string name,
                                          std::shared_ptr<Module> m) {
    modules_[name] = m;
    return m;
  }

  bool training_ = true;
  std::map<std::string, Tensor> parameters_;
  std::map<std::string, Tensor> buffers_;
  std::map<std::string, std::shared_ptr<Module>> modules_;
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
    weight = register_parameter("weight", w);

    if (bias) {
      Tensor b({out_features}, Device{DeviceType::CPU, 0}, DataType::Float32,
               true);
      b.uniform_(-limit, limit);
      this->bias = register_parameter("bias", b);
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
    weight = register_parameter("weight", w);

    Tensor b({out_channels}, Device{DeviceType::CPU, 0}, DataType::Float32,
             true);
    b.uniform_(-limit, limit);
    bias = register_parameter("bias", b);
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

class MaxPool2d : public Module {
public:
  MaxPool2d(int kernel_size, int stride = 2, int padding = 0)
      : k_(kernel_size), s_(stride), p_(padding) {}
  Tensor forward(Tensor x) override { return x.max_pool2d(k_, s_, p_); }

private:
  int k_, s_, p_;
};

class Upsample : public Module {
public:
  Upsample(int scale_factor) : scale_(scale_factor) {}
  Tensor forward(Tensor x) override { return x.upsample2d(scale_); }

private:
  int scale_;
};

class BatchNorm2d : public Module {
public:
  BatchNorm2d(int num_features, float eps = 1e-5f, float momentum = 0.1f)
      : eps_(eps), momentum_(momentum) {
    Tensor w({num_features}, Device{DeviceType::CPU, 0}, DataType::Float32,
             true);
    w.uniform_(1.0f, 1.0f); // Init to 1
    weight = register_parameter("weight", w);

    Tensor b({num_features}, Device{DeviceType::CPU, 0}, DataType::Float32,
             true);
    b.uniform_(0.0f, 0.0f); // Init to 0
    bias = register_parameter("bias", b);

    Tensor rm({num_features}, Device{DeviceType::CPU, 0}, DataType::Float32,
              false);
    rm.uniform_(0.0f, 0.0f);
    running_mean = register_buffer("running_mean", rm);

    Tensor rv({num_features}, Device{DeviceType::CPU, 0}, DataType::Float32,
              false);
    rv.uniform_(1.0f, 1.0f);
    running_var = register_buffer("running_var", rv);
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

private:
  std::vector<std::shared_ptr<Module>> ordered_modules_;
};

} // namespace nn
} // namespace munet
