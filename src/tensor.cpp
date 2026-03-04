#include "tensor.hpp"
#include "autograd/autograd.hpp"
#include "backend/cpu_backend.hpp"
#include "ops.hpp"
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef MUNET_USE_CUDA
#include "backend/cuda_backend.hpp"
#endif

#ifdef MUNET_USE_VULKAN
#include "backend/vulkan_backend.hpp"
#endif

namespace munet {

// --- Backend Management ---
std::shared_ptr<Backend> BackendManager::get(Device device) {
  if (device.type == DeviceType::CPU) {
    static auto cpu_backend = std::make_shared<CPUBackend>();
    return cpu_backend;
  }
#ifdef MUNET_USE_CUDA
  if (device.type == DeviceType::CUDA) {
    static auto cuda_backend = std::make_shared<CUDABackend>();
    return cuda_backend;
  }
#endif
#ifdef MUNET_USE_VULKAN
  if (device.type == DeviceType::VULKAN) {
    static auto vulkan_backend = std::make_shared<VulkanBackend>();
    return vulkan_backend;
  }
#endif
  throw std::runtime_error("Requested backend not compiled or implemented.");
}

// --- Autograd ---
void Tensor::backward(const Tensor &grad) {
  if (!impl_->grad_fn)
    return;
  Engine::execute(impl_->grad_fn.get(), grad);
}

void Tensor::backward() {
  if (!impl_->grad_fn)
    return;
  Tensor root_grad(shape(), device(), dtype());
  std::vector<float> ones(size(), 1.0f);
  impl_->backend().copy(ones.data(), root_grad.data(), bytes(),
                        Device{DeviceType::CPU, 0}, device());
  Engine::execute(impl_->grad_fn.get(), root_grad);
}

// --- Ops ---
Tensor Tensor::operator+(const Tensor &other) const {
  return ops::add(*this, other);
}

Tensor Tensor::matmul(const Tensor &other) const {
  return ops::matmul(*this, other);
}

Tensor Tensor::relu() const { return ops::relu(*this); }
Tensor Tensor::sigmoid() const { return ops::sigmoid(*this); }
Tensor Tensor::softmax() const { return ops::softmax(*this); }

Tensor Tensor::conv2d(const Tensor &weight, const Tensor &bias, int stride,
                      int padding) const {
  return ops::conv2d(*this, weight, bias, stride, padding);
}
Tensor Tensor::max_pool2d(int kernel_size, int stride, int padding) const {
  return ops::max_pool2d(*this, kernel_size, stride, padding);
}
Tensor Tensor::upsample2d(int scale_factor) const {
  return ops::upsample2d(*this, scale_factor);
}

// --- Utilities ---
struct ToBackward : public Node {
  Device src_device;
  ToBackward(Device dev) : src_device(dev) {}
  std::string name() const override { return "ToBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    return {grads[0].to(src_device)};
  }
};

Tensor Tensor::to(Device dev) const {
  if (device() == dev)
    return *this;

  Tensor out(shape(), dev, dtype(), requires_grad());

  // Safety routing matrix:
  if (device().type == DeviceType::CUDA || dev.type == DeviceType::CUDA) {
    Device cuda_dev = (device().type == DeviceType::CUDA) ? device() : dev;
    BackendManager::get(cuda_dev)->copy(data(), out.data(), bytes(), device(),
                                        dev);
  } else if (device().type == DeviceType::VULKAN ||
             dev.type == DeviceType::VULKAN) {
    Device vk_dev = (device().type == DeviceType::VULKAN) ? device() : dev;
    BackendManager::get(vk_dev)->copy(data(), out.data(), bytes(), device(),
                                      dev);
  } else {
    // CPU -> CPU routing
    out.impl_->backend().copy(data(), out.data(), bytes(), device(), dev);
  }

  if (requires_grad()) {
    auto fn = std::make_shared<ToBackward>(device());
    ops::link_backward_edges(fn.get(), {*this});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  ops::record_trace(out, "To", {*this});

  return out;
}

Tensor Tensor::operator-(const Tensor &other) const {
  return ops::sub(*this, other);
}
Tensor Tensor::operator*(const Tensor &other) const {
  return ops::mul(*this, other);
}
Tensor Tensor::sum() const { return ops::sum(*this); }

Tensor Tensor::reshape(Shape new_shape) const {
  return ops::reshape(*this, new_shape);
}

void Tensor::uniform_(float low, float high) {
  if (size() == 0)
    return;

  // Initialize a fresh CPU tensor to avoid copying uninitialized garbage from
  // GPU
  Tensor cpu_tensor(shape(), Device{DeviceType::CPU, 0}, dtype(), false);
  std::mt19937 gen(42); // fixed seed for predictability
  std::uniform_real_distribution<float> dis(low, high);
  float *ptr = (float *)cpu_tensor.data();
  for (size_t i = 0; i < size(); ++i)
    ptr[i] = dis(gen);

  if (device().type != DeviceType::CPU) {
    impl_->backend().copy(cpu_tensor.data(), data(), bytes(),
                          Device{DeviceType::CPU, 0}, device());
  } else {
    std::memcpy(data(), cpu_tensor.data(), bytes());
  }
}

void Tensor::step(float lr) {
  if (!impl_->grad)
    return;
  impl_->backend().update(*impl_->storage, *impl_->grad->storage, lr, size());
}

Tensor Tensor::batch_norm(Tensor &running_mean, Tensor &running_var,
                          const Tensor &weight, const Tensor &bias,
                          bool training, float momentum, float eps) const {
  return ops::batch_norm(*this, running_mean, running_var, weight, bias,
                         training, momentum, eps);
}

Tensor Tensor::mse_loss(const Tensor &target) const {
  return ops::mse_loss(*this, target);
}

Tensor Tensor::cross_entropy(const Tensor &target) const {
  return ops::cross_entropy(*this, target);
}

} // namespace munet
