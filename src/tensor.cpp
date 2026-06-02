#include "tensor.hpp"
#include "autograd/engine.hpp"
#include "core/util.hpp"
#include "ops.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace munet {

// --- Autograd ---
void Tensor::backward(const Tensor &grad) { backward(grad, false); }

void Tensor::backward(bool retain_graph) {
  if (!impl_->grad_fn)
    return;

  if (size() != 1)
    throw std::runtime_error("backward() requires scalar tensor");
  if (!is_floating(dtype())) {
    throw std::runtime_error("backward() requires a floating-point tensor");
  }

  // Create the root gradient without tracking history.
  Tensor root_grad;
  {
    NoGradGuard guard;
    root_grad = Tensor(shape(), device(), dtype());
    root_grad.fill_(1.0f);
  }

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
Tensor Tensor::exp() const { return ops::exp(*this); }
Tensor Tensor::log() const { return ops::log(*this); }
Tensor Tensor::sqrt() const { return ops::sqrt(*this); }
Tensor Tensor::rsqrt() const { return ops::rsqrt(*this); }
Tensor Tensor::sin() const { return ops::sin(*this); }
Tensor Tensor::cos() const { return ops::cos(*this); }
Tensor Tensor::softmax(int dim) const { return ops::softmax(*this, dim); }
Tensor Tensor::log_softmax(int dim) const {
  return ops::log_softmax(*this, dim);
}

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

  if (!is_contiguous()) {
    return this->contiguous().to(dev);
  }

  Tensor out(shape(), dev, dtype(), requires_grad());
  size_t byte_count = bytes();

  if (device().type != DeviceType::VULKAN || dev.type != DeviceType::VULKAN) {
    throw std::runtime_error(
        "Tensor::to only supports Vulkan tensors; external transfers must use "
        "from_numpy/copy_from_numpy/numpy explicitly.");
  }

  impl_->backend().copy(data(), out.data(), byte_count, device(), dev);

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

void Tensor::to_(Device dev) {
  // In-place device transfer: preserve TensorImpl identity for optimizer
  // references
  Tensor moved = this->to(dev);
  // Swap storage within the existing TensorImpl
  impl_->storage = std::move(moved.impl_->storage);
}

Tensor Tensor::to(DataType target_dtype) const {
  if (dtype() == target_dtype)
    return *this;

  Tensor out(shape(), device(), target_dtype, requires_grad());
  const size_t src_bytes = bytes();
  const size_t dst_bytes = out.bytes();
  std::vector<char> src_external(src_bytes);
  std::vector<char> dst_external(dst_bytes);

  impl_->backend().copy(data(), src_external.data(), src_bytes, device(),
                        Device{DeviceType::EXTERNAL, 0});
  convert_buffer_dtype(src_external.data(), dtype(), dst_external.data(),
                       target_dtype, size());
  out.impl_->backend().copy(dst_external.data(), out.data(), dst_bytes,
                            Device{DeviceType::EXTERNAL, 0}, out.device());

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

Tensor Tensor::mean(int dim, bool keepdim) const {
  return ops::mean(*this, dim, keepdim);
}

Tensor Tensor::reshape(Shape new_shape) const {
  return ops::reshape(*this, new_shape);
}

Tensor Tensor::narrow(int dim, int start, int length) const {
  return ops::narrow(*this, dim, start, length);
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

  if (device().type == DeviceType::VULKAN) {
    impl_->backend().synchronize();
    return read_scalar_from_buffer(data(), dtype());
  }

  return to(Device{DeviceType::VULKAN, 0}).item_value();
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

  if (device().type != DeviceType::VULKAN) {
    throw std::runtime_error("fill_ only supports Vulkan tensors.");
  }

  std::vector<char> external_bytes(bytes());
  const size_t element_size = dtype_size(dtype());
  for (size_t i = 0; i < size(); ++i) {
    write_scalar_to_buffer(external_bytes.data() + i * element_size, dtype(),
                           value.value);
  }

  impl_->backend().copy(external_bytes.data(), data(), bytes(),
                        Device{DeviceType::EXTERNAL, 0}, device());
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

  Tensor out(shape(), options());
  impl_->backend().to_contiguous(*impl_->storage, *out.impl_->storage, shape(),
                                 strides(), storage_offset());
  return out;
}

} // namespace munet
