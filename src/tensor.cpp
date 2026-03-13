#include "tensor.hpp"
#include "amp.hpp"
#include "autograd/autograd.hpp"
#include "ops.hpp"
#include "util.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace munet {

namespace {
inline Tensor maybe_autocast_tensor(const Tensor &t, amp::AutocastOp op) {
  if (!amp::AutocastMode::is_enabled() || !amp::should_autocast(op))
    return t;
  DataType target = amp::AutocastMode::dtype();
  if (!is_float_dtype(t.dtype()) || t.dtype() == target)
    return t;
  return t.to_dtype(target);
}
} // namespace

// --- Autograd ---
void Tensor::backward(const Tensor &grad) {
  if (!impl_->grad_fn)
    return;
  Engine::execute(impl_->grad_fn.get(), grad);
}

void Tensor::backward() {
  if (!impl_->grad_fn)
    return;

  if (size() != 1)
    throw std::runtime_error("backward() requires scalar tensor");

  Tensor root_grad(shape(), device(), dtype());

  float one = 1.0f;

  impl_->backend().copy(&one, root_grad.data(), sizeof(float),
                        Device{DeviceType::CPU, 0}, device());

  Engine::execute(impl_->grad_fn.get(), root_grad);
}

Tensor Tensor::detach() const {
  // Create a new tensor with the same shape, device, and dtype, but NO grad
  Tensor out(shape(), device(), dtype(), false);

  // Share the underlying storage (no physical memory copy needed yet)
  out.impl_->storage = impl_->storage;

  return out;
}

// --- Ops ---
Tensor Tensor::operator+(const Tensor &other) const {
  Tensor lhs = maybe_autocast_tensor(*this, amp::AutocastOp::Add);
  Tensor rhs = maybe_autocast_tensor(other, amp::AutocastOp::Add);
  return ops::add(lhs, rhs);
}

Tensor Tensor::matmul(const Tensor &other) const {
  Tensor lhs = maybe_autocast_tensor(*this, amp::AutocastOp::Matmul);
  Tensor rhs = maybe_autocast_tensor(other, amp::AutocastOp::Matmul);
  return ops::matmul(lhs, rhs);
}

Tensor Tensor::relu() const {
  Tensor x = maybe_autocast_tensor(*this, amp::AutocastOp::Relu);
  return ops::relu(x);
}
Tensor Tensor::sigmoid() const {
  Tensor x = maybe_autocast_tensor(*this, amp::AutocastOp::Sigmoid);
  return ops::sigmoid(x);
}
Tensor Tensor::softmax(int dim) const {
  Tensor x = maybe_autocast_tensor(*this, amp::AutocastOp::Softmax);
  return ops::softmax(x, dim);
}
Tensor Tensor::log_softmax(int dim) const {
  Tensor x = maybe_autocast_tensor(*this, amp::AutocastOp::LogSoftmax);
  return ops::log_softmax(x, dim);
}

Tensor Tensor::conv2d(const Tensor &weight, const Tensor &bias, int stride,
                      int padding) const {
  Tensor in = maybe_autocast_tensor(*this, amp::AutocastOp::Conv2D);
  Tensor w = maybe_autocast_tensor(weight, amp::AutocastOp::Conv2D);
  Tensor b = maybe_autocast_tensor(bias, amp::AutocastOp::Conv2D);
  return ops::conv2d(in, w, b, stride, padding);
}
Tensor Tensor::max_pool2d(int kernel_size, int stride, int padding) const {
  Tensor in = maybe_autocast_tensor(*this, amp::AutocastOp::MaxPool2D);
  return ops::max_pool2d(in, kernel_size, stride, padding);
}
Tensor Tensor::upsample2d(int scale_factor) const {
  Tensor in = maybe_autocast_tensor(*this, amp::AutocastOp::Upsample2D);
  return ops::upsample2d(in, scale_factor);
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


struct ToDTypeBackward : public Node {
  DataType src_dtype;
  explicit ToDTypeBackward(DataType dt) : src_dtype(dt) {}
  std::string name() const override { return "ToDTypeBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    return {grads[0].to_dtype(src_dtype)};
  }
};

Tensor Tensor::to(Device dev) const {
  if (device() == dev)
    return *this;

  // 1. Setup Profiling only if enabled (enqueue-time, non-blocking)
  bool profiling = is_profile_enabled();
  std::unique_ptr<Timer> timer;

  Tensor out(shape(), dev, dtype(), requires_grad());
  size_t byte_count = bytes();

  if (profiling) {
    timer = std::make_unique<Timer>();
  }

  // 2. Perform the actual transfer
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

  // 3. Finalize Profiling (avoid forced sync in profile-only mode)
  if (profiling) {
    double us = timer->elapsed_us();
    std::string name = "to(" + dev.to_string() + ")";
    Profiler::get().record(name, us, 0.0, byte_count, to_string(shape()));
  }

  // 4. Autograd and Tracing
  if (GradMode::is_enabled() && requires_grad()) {
    // Only link ToBackward if the current tensor is already part of a graph
    // (non-leaf). If it's a leaf (no grad_fn), moving it to a new device
    // creates a new leaf on that device.
    if (impl_->grad_fn) {
      auto fn = std::make_shared<ToBackward>(device());
      ops::link_backward_edges(fn.get(), {*this});
      out.set_requires_grad(true);
      out.impl_->grad_fn = fn;
    } else {
      out.set_requires_grad(true);
      // out.impl_->grad_fn remains nullptr, making 'out' a proper leaf tensor.
    }
  }

  if (impl_->grad_fn) {
    ops::record_trace(out, "To", {*this});
  }

  return out;
}


Tensor Tensor::to_dtype(DataType target_dtype) const {
  if (dtype() == target_dtype)
    return *this;

  Device cpu{DeviceType::CPU, 0};
  Tensor src_cpu = (device().type == DeviceType::CPU) ? *this : to(cpu);
  Tensor out_cpu(shape(), cpu, target_dtype, requires_grad());

  for (size_t i = 0; i < src_cpu.size(); ++i) {
    double v = load_scalar_as_double(src_cpu.data(), src_cpu.dtype(), i);
    store_scalar_from_double(out_cpu.data(), target_dtype, i, v);
  }

  Tensor out = (device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(device());

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
    ops::record_trace(out, "ToDType", {*this});
  }

  return out;
}

Tensor Tensor::operator-(const Tensor &other) const {
  Tensor lhs = maybe_autocast_tensor(*this, amp::AutocastOp::Sub);
  Tensor rhs = maybe_autocast_tensor(other, amp::AutocastOp::Sub);
  return ops::sub(lhs, rhs);
}
Tensor Tensor::operator*(const Tensor &other) const {
  Tensor lhs = maybe_autocast_tensor(*this, amp::AutocastOp::Mul);
  Tensor rhs = maybe_autocast_tensor(other, amp::AutocastOp::Mul);
  return ops::mul(lhs, rhs);
}

Tensor Tensor::operator/(const Tensor &other) const {
  Tensor lhs = maybe_autocast_tensor(*this, amp::AutocastOp::Div);
  Tensor rhs = maybe_autocast_tensor(other, amp::AutocastOp::Div);
  return ops::div(lhs, rhs);
}

Tensor Tensor::cat(const std::vector<Tensor> &inputs, int dim) {
  return ops::cat(inputs, dim);
}

Tensor Tensor::sum() const { return ops::sum(*this); }

Tensor Tensor::reshape(Shape new_shape) const {
  return ops::reshape(*this, new_shape);
}

Tensor Tensor::masked_fill(const Tensor &mask, float value) const {
  return ops::masked_fill(*this, mask, value);
}

float Tensor::item() const {
  if (size() != 1) {
    throw std::runtime_error(
        "item() can only be called on tensors with 1 element");
  }

  if (device().type == DeviceType::CPU) {
    impl_->backend().synchronize();
    return static_cast<const float *>(data())[0];
  } else {
    // Recursively call item() on a CPU copy
    return to(Device{DeviceType::CPU, 0}).item();
  }
}

void Tensor::step(float lr) {
  if (!impl_ || !impl_->grad) {
    MUNET_WARNING << "Skipping tensor step with no grad" << std::endl;
    return;
  }
  impl_->backend().update(*impl_->storage, *impl_->grad->storage, lr, size());
}

Tensor Tensor::batch_norm(Tensor &running_mean, Tensor &running_var,
                          const Tensor &weight, const Tensor &bias,
                          bool training, float momentum, float eps) const {
  Tensor in = maybe_autocast_tensor(*this, amp::AutocastOp::BatchNorm);
  Tensor w = maybe_autocast_tensor(weight, amp::AutocastOp::BatchNorm);
  Tensor b = maybe_autocast_tensor(bias, amp::AutocastOp::BatchNorm);
  return ops::batch_norm(in, running_mean, running_var, w, b, training,
                         momentum, eps);
}

Tensor Tensor::layer_norm(const Tensor &weight, const Tensor &bias,
                          float eps) const {
  Tensor in = maybe_autocast_tensor(*this, amp::AutocastOp::LayerNorm);
  Tensor w = maybe_autocast_tensor(weight, amp::AutocastOp::LayerNorm);
  Tensor b = maybe_autocast_tensor(bias, amp::AutocastOp::LayerNorm);
  return ops::layer_norm(in, w, b, eps);
}

Tensor Tensor::mse_loss(const Tensor &target) const {
  Tensor pred = maybe_autocast_tensor(*this, amp::AutocastOp::MSELoss);
  Tensor tgt = maybe_autocast_tensor(target, amp::AutocastOp::MSELoss);
  return ops::mse_loss(pred, tgt);
}

Tensor Tensor::cross_entropy(const Tensor &target) const {
  Tensor logits = maybe_autocast_tensor(*this, amp::AutocastOp::CrossEntropy);
  Tensor tgt = maybe_autocast_tensor(target, amp::AutocastOp::CrossEntropy);
  return ops::cross_entropy(logits, tgt);
}

void Tensor::uniform_(float low, float high) {
  if (size() == 0)
    return;
  // Delegate directly to backend. No more CPU roundtrip.
  impl_->backend().fill_uniform(*impl_->storage, low, high, size());
}

Tensor Tensor::transpose(int dim0, int dim1) const {
  Tensor out = *this; // Shallow copy of TensorImpl
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

  Tensor out(shape(), device(), dtype(), requires_grad());

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
