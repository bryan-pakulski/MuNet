#include "tensor.hpp"
#include "autograd/engine.hpp"
#include "ops.hpp"
#include "core/util.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace munet {
namespace {

Tensor convert_tensor_dtype_cpu(const Tensor &input, DataType target_dtype) {
  Tensor out(input.shape(), Device{DeviceType::CPU, 0}, target_dtype,
             input.requires_grad());
  convert_buffer_dtype(input.data(), input.dtype(), out.data(), target_dtype,
                       input.size());
  return out;
}

} // namespace

// --- Autograd ---
void Tensor::backward(const Tensor &grad) {
  backward(grad, false);
}

void Tensor::backward(bool retain_graph) {
  if (!impl_->grad_fn)
    return;

  if (size() != 1)
    throw std::runtime_error("backward() requires scalar tensor");
  if (!is_floating(dtype())) {
    throw std::runtime_error("backward() requires a floating-point tensor");
  }

  Tensor root_grad(shape(), device(), dtype());
  Tensor root_cpu(shape(), Device{DeviceType::CPU, 0}, dtype());
  write_scalar_to_buffer(root_cpu.data(), dtype(), 1.0);
  impl_->backend().copy(root_cpu.data(), root_grad.data(), root_grad.bytes(),
                        root_cpu.device(), device());

  backward(root_grad, retain_graph);
}

void Tensor::backward(const Tensor &grad, bool retain_graph) {
  if (!impl_->grad_fn)
    return;
  Engine::get_default().execute(
      BackwardRequest{impl_->grad_fn, grad, retain_graph, false, {}});
}

void Tensor::backward() { backward(false); }

void Tensor::register_gradient_hook(GradientHook hook) const {
  if (!impl_ || !impl_->grad_fn) {
    throw std::runtime_error(
        "register_gradient_hook() requires a tensor with a grad_fn");
  }
  impl_->grad_fn->register_gradient_hook(std::move(hook));
}

Tensor Tensor::detach() const {
  Tensor out(shape(), device(), dtype(), false);
  out.impl_->storage = impl_->storage;
  return out;
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
Tensor Tensor::softmax(int dim) const { return ops::softmax(*this, dim); }
Tensor Tensor::log_softmax(int dim) const { return ops::log_softmax(*this, dim); }

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
  explicit ToBackward(Device dev) : src_device(dev) {}
  std::string name() const override { return "ToBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    return {grads[0].to(src_device)};
  }
};

struct ToDTypeBackward : public Node {
  DataType src_dtype;
  explicit ToDTypeBackward(DataType dtype) : src_dtype(dtype) {}
  std::string name() const override { return "ToDTypeBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    return {grads[0].to(src_dtype)};
  }
};

Tensor Tensor::to(Device dev) const {
  if (device() == dev)
    return *this;

  Tensor out(shape(), dev, dtype(), requires_grad());
  size_t byte_count = bytes();

  if (device().type == DeviceType::CUDA || dev.type == DeviceType::CUDA) {
#ifdef MUNET_USE_CUDA
    Device cuda_dev = (device().type == DeviceType::CUDA) ? device() : dev;
    BackendManager::get(cuda_dev)->copy(data(), out.data(), byte_count,
                                        device(), dev);
#else
    throw std::runtime_error("CUDA backend not compiled");
#endif
  } else if (device().type == DeviceType::VULKAN ||
             dev.type == DeviceType::VULKAN) {
#ifdef MUNET_USE_VULKAN
    Device vk_dev = (device().type == DeviceType::VULKAN) ? device() : dev;
    BackendManager::get(vk_dev)->copy(data(), out.data(), byte_count, device(),
                                      dev);
#else
    throw std::runtime_error("Vulkan backend not compiled");
#endif
  } else {
    impl_->backend().copy(data(), out.data(), byte_count, device(), dev);
  }

  if (GradMode::is_enabled() && requires_grad()) {
    if (impl_->grad_fn) {
      auto fn = std::make_shared<ToBackward>(device());
      ops::link_backward_edges(fn.get(), {*this});
      out.set_requires_grad(true);
      out.impl_->grad_fn = fn;
    } else {
      out.set_requires_grad(true);
    }
  }

  if (impl_->grad_fn) {
    ops::record_trace(out, "To", {*this});
  }

  return out;
}

Tensor Tensor::to(DataType target_dtype) const {
  if (dtype() == target_dtype)
    return *this;

  Device cpu{DeviceType::CPU, 0};
  Tensor cpu_src = (device().type == DeviceType::CPU) ? *this : to(cpu);
  Tensor cpu_out;
  if (is_profile_enabled()) {
    Timer timer;
    cpu_out = convert_tensor_dtype_cpu(cpu_src, target_dtype);
    Profiler::get().record("transfer.dtype_convert", timer.elapsed_us(), 0.0,
                           cpu_src.bytes(), to_string(cpu_src.shape()));
  } else {
    cpu_out = convert_tensor_dtype_cpu(cpu_src, target_dtype);
  }
  Tensor out = (device().type == DeviceType::CPU) ? cpu_out : cpu_out.to(device());

  if (GradMode::is_enabled() && requires_grad()) {
    if (impl_->grad_fn) {
      auto fn = std::make_shared<ToDTypeBackward>(dtype());
      ops::link_backward_edges(fn.get(), {*this});
      out.set_requires_grad(true);
      out.impl_->grad_fn = fn;
    } else {
      out.set_requires_grad(true);
    }
  }

  if (impl_->grad_fn) {
    ops::record_trace(out, "ToDType", {*this},
                      {{"dtype", {static_cast<int>(target_dtype)}}});
  }

  return out;
}

Tensor Tensor::to(const TensorOptions &target_options) const {
  Tensor out = *this;
  if (dtype() != target_options.dtype) {
    out = out.to(target_options.dtype);
  }
  if (out.device() != target_options.device) {
    out = out.to(target_options.device);
  }

  if (!target_options.requires_grad) {
    out = out.detach();
    out.set_requires_grad(false);
  } else {
    out.set_requires_grad(true);
  }

  return out;
}

Tensor Tensor::operator-(const Tensor &other) const {
  return ops::sub(*this, other);
}
Tensor Tensor::operator*(const Tensor &other) const {
  return ops::mul(*this, other);
}

Tensor Tensor::operator/(const Tensor &other) const {
  return ops::div(*this, other);
}

Tensor Tensor::cat(const std::vector<Tensor> &inputs, int dim) {
  return ops::cat(inputs, dim);
}

Tensor Tensor::sum() const { return ops::sum(*this); }

Tensor Tensor::reshape(Shape new_shape) const {
  return ops::reshape(*this, new_shape);
}

Tensor Tensor::masked_fill(const Tensor &mask, const ScalarValue &value) const {
  return ops::masked_fill(*this, mask, value);
}

Tensor Tensor::masked_fill(const Tensor &mask, float value) const {
  return masked_fill(mask, make_scalar(value));
}

ScalarValue Tensor::item_value() const {
  if (size() != 1) {
    throw std::runtime_error(
        "item_value() can only be called on tensors with 1 element");
  }

  if (device().type == DeviceType::CPU) {
    impl_->backend().synchronize();
    return read_scalar_from_buffer(data(), dtype());
  }

  return to(Device{DeviceType::CPU, 0}).item_value();
}

float Tensor::item() const { return item_value().as_float(); }

void Tensor::step(float lr) {
  if (!impl_ || !impl_->grad) {
    MUNET_WARNING << "Skipping tensor step with no grad" << std::endl;
    return;
  }
  impl_->backend().update(*impl_->storage, *impl_->grad->storage, lr, size());
  impl_->bump_version();
}

Tensor Tensor::batch_norm(Tensor &running_mean, Tensor &running_var,
                          const Tensor &weight, const Tensor &bias,
                          bool training, float momentum, float eps) const {
  return ops::batch_norm(*this, running_mean, running_var, weight, bias,
                         training, momentum, eps);
}

Tensor Tensor::layer_norm(const Tensor &weight, const Tensor &bias,
                          float eps) const {
  return ops::layer_norm(*this, weight, bias, eps);
}

Tensor Tensor::mse_loss(const Tensor &target) const {
  return ops::mse_loss(*this, target);
}

Tensor Tensor::cross_entropy(const Tensor &target) const {
  return ops::cross_entropy(*this, target);
}

void Tensor::uniform_(float low, float high) {
  if (size() == 0)
    return;
  if (!is_floating(dtype())) {
    throw std::runtime_error("uniform_ only supports floating-point tensors");
  }
  impl_->backend().fill_uniform(*impl_->storage, low, high, size());
  impl_->bump_version();
}

void Tensor::fill_(const ScalarValue &value) {
  if (size() == 0)
    return;

  Device cpu{DeviceType::CPU, 0};
  Tensor cpu_out(shape(), cpu, dtype(), requires_grad());
  char *cpu_bytes = static_cast<char *>(cpu_out.data());
  const size_t element_size = dtype_size(dtype());
  for (size_t i = 0; i < size(); ++i) {
    write_scalar_to_buffer(cpu_bytes + i * element_size, dtype(), value.value);
  }

  if (device().type == DeviceType::CPU) {
    impl_->backend().copy(cpu_out.data(), data(), bytes(), cpu, device());
    impl_->bump_version();
    return;
  }

  BackendManager::get(device())->copy(cpu_out.data(), data(), bytes(), cpu,
                                      device());
  impl_->bump_version();
}

Tensor Tensor::transpose(int dim0, int dim1) const {
  Tensor out = *this;
  out.impl_ = std::make_shared<TensorImpl>(*impl_);
  std::swap(out.impl_->shape[dim0], out.impl_->shape[dim1]);
  std::swap(out.impl_->strides[dim0], out.impl_->strides[dim1]);
  return out;
}

Tensor Tensor::permute(const std::vector<int> &dims) const {
  if (dims.size() != impl_->shape.size())
    throw std::runtime_error("permute: dims size must match rank");

  std::vector<int> seen(dims.size(), 0);
  Tensor out = *this;
  out.impl_ = std::make_shared<TensorImpl>(*impl_);

  Shape new_shape(dims.size());
  Strides new_strides(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    int d = dims[i];
    if (d < 0 || d >= (int)dims.size())
      throw std::runtime_error("permute: dim out of range");
    if (seen[d])
      throw std::runtime_error("permute: dims must be unique");
    seen[d] = 1;
    new_shape[i] = impl_->shape[d];
    new_strides[i] = impl_->strides[d];
  }

  out.impl_->shape = new_shape;
  out.impl_->strides = new_strides;
  return out;
}

Tensor Tensor::contiguous() const {
  if (is_contiguous())
    return *this;

  if (device().type != DeviceType::CPU) {
    Tensor cpu_copy = this->to(Device{DeviceType::CPU, 0});
    Tensor cpu_contig = cpu_copy.contiguous();
    return cpu_contig.to(device());
  }

  Tensor out(shape(), options());

  const char *src = static_cast<const char *>(data());
  char *dst = static_cast<char *>(out.data());
  size_t elem_size = dtype_size(dtype());

  Shape idx(shape().size(), 0);
  for (size_t linear = 0; linear < size(); ++linear) {
    size_t rem = linear;
    for (int d = static_cast<int>(shape().size()) - 1; d >= 0; --d) {
      idx[d] = static_cast<int>(rem % static_cast<size_t>(shape()[d]));
      rem /= static_cast<size_t>(shape()[d]);
    }

    size_t src_offset_elems = storage_offset();
    for (size_t d = 0; d < idx.size(); ++d) {
      src_offset_elems += static_cast<size_t>(idx[d]) *
                          static_cast<size_t>(strides()[d]);
    }

    std::memcpy(dst + linear * elem_size, src + src_offset_elems * elem_size,
                elem_size);
  }

  return out;
}

} // namespace munet
