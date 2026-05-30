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
namespace {

Tensor convert_tensor_dtype_host(const Tensor &input, DataType target_dtype) {
  NoGradGuard guard;
  Tensor out(input.shape(), Device{DeviceType::VULKAN, 0}, target_dtype,
             input.requires_grad());

  convert_buffer_dtype(input.data(), input.dtype(), out.data(), target_dtype,
                       input.size());
  return out;
}

} // namespace

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

  // Create the root gradient without tracking history
  Tensor root_grad;
  {
    NoGradGuard guard;
    root_grad = Tensor(shape(), device(), dtype());
    Tensor root_host(shape(), Device{DeviceType::VULKAN, 0}, dtype());
    write_scalar_to_buffer(root_host.data(), dtype(), 1.0);
    impl_->backend().copy(root_host.data(), root_grad.data(), root_grad.bytes(),
                          root_host.device(), device());
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
    // To avoid recursion, we perform the Host-based packs/moves manually
    // or ensure contiguous() doesn't call back into this check.
    // If we are moving to another device, pack it on Host first.
    if (device().type != DeviceType::VULKAN) {
      // Create a temporary Host tensor and manually pack into it
      Tensor host_contig(shape(), Device{DeviceType::VULKAN, 0}, dtype());
      Tensor host_view = this->to(Device{
          DeviceType::VULKAN, 0}); // This calls back, but 'this' is non-contiguous
      // We need a way to move the raw bytes to Host regardless of contiguity
    }
    return this->contiguous().to(dev);
  }

  Tensor out(shape(), dev, dtype(), requires_grad());
  size_t byte_count = bytes();

  const bool src_non_host = device().type != DeviceType::VULKAN;
  const bool dst_non_host = dev.type != DeviceType::VULKAN;
  const bool cross_backend_non_host =
      src_non_host && dst_non_host && device().type != dev.type;
  const bool cross_vulkan_device_non_host =
      src_non_host && dst_non_host && device().type == DeviceType::VULKAN &&
      dev.type == DeviceType::VULKAN && device().index != dev.index;

  if (cross_backend_non_host || cross_vulkan_device_non_host) {
    // Route heterogeneous/non-peer Vulkan transfers through Host staging to avoid
    // backend-specific direct-copy assumptions and multi-device Vulkan buffer
    // ownership constraints.
    Tensor host_stage(shape(), Device{DeviceType::VULKAN, 0}, dtype(), false);
    BackendManager::get(device())->copy(data(), host_stage.data(), byte_count,
                                        device(), host_stage.device());
    BackendManager::get(dev)->copy(host_stage.data(), out.data(), byte_count,
                                   host_stage.device(), dev);
  } else if (device().type == DeviceType::VULKAN ||
             dev.type == DeviceType::VULKAN) {
    Device vk_dev = (device().type == DeviceType::VULKAN) ? device() : dev;
    BackendManager::get(vk_dev)->copy(data(), out.data(), byte_count, device(),
                                      dev);
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

void Tensor::to_(Device dev) {
  // In-place device transfer: preserve TensorImpl identity for optimizer references
  Tensor moved = this->to(dev);
  // Swap storage within the existing TensorImpl
  impl_->storage = std::move(moved.impl_->storage);
}

Tensor Tensor::to(DataType target_dtype) const {
  if (dtype() == target_dtype)
    return *this;

  Device host{DeviceType::VULKAN, 0};
  Tensor host_src = (device().type == DeviceType::VULKAN) ? *this : to(host);
  Tensor host_out;
  if (is_profile_enabled()) {
    Timer timer;
    host_out = convert_tensor_dtype_host(host_src, target_dtype);
    Profiler::get().record("transfer.dtype_convert", timer.elapsed_us(), 0.0,
                           host_src.bytes(), to_string(host_src.shape()));
  } else {
    host_out = convert_tensor_dtype_host(host_src, target_dtype);
  }
  Tensor out =
      (device().type == DeviceType::VULKAN) ? host_out : host_out.to(device());

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

  Device host{DeviceType::VULKAN, 0};
  Tensor host_out(shape(), host, dtype(), requires_grad());
  char *host_bytes = static_cast<char *>(host_out.data());
  const size_t element_size = dtype_size(dtype());
  for (size_t i = 0; i < size(); ++i) {
    write_scalar_to_buffer(host_bytes + i * element_size, dtype(), value.value);
  }

  if (device().type == DeviceType::VULKAN) {
    impl_->backend().copy(host_out.data(), data(), bytes(), host, device());
    impl_->bump_version();
    return;
  }

  BackendManager::get(device())->copy(host_out.data(), data(), bytes(), host,
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

  Tensor out(shape(), options());
  impl_->backend().to_contiguous(*impl_->storage, *out.impl_->storage, shape(),
                                 strides(), storage_offset());
  return out;
}

} // namespace munet
