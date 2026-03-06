#include "tensor.hpp"
#include "autograd/autograd.hpp"
#include "backend/cpu_backend.hpp"
#include "ops.hpp"
#include "util.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef MUNET_USE_CUDA
#include "backend/cuda_backend.hpp"
#endif

#ifdef MUNET_USE_VULKAN
#include "backend/vulkan_backend.hpp"
#endif

namespace munet {

namespace {

// Intercepts all backend calls and enforces a synchronized sanity check
class DebugBackend : public Backend {
  std::shared_ptr<Backend> base_;

  void check(const char *name, double cpu_us,
             const Storage *out_storage = nullptr) {
    try {
      base_->synchronize();
      double gpu_us = base_->get_last_kernel_time_us();

      if (is_profile_enabled()) {
        size_t bytes = out_storage ? out_storage->size_bytes() : 0;
        Profiler::get().record(name, cpu_us, gpu_us, bytes,
                               out_storage ? to_string(out_storage->shape())
                                           : "");
      }

      if (out_storage && out_storage->device().type == DeviceType::CPU) {
        float *data = (float *)out_storage->data();
        for (size_t i = 0; i < out_storage->size_bytes() / 4; ++i) {
          if (!std::isfinite(data[i])) {
            MUNET_ERROR << "Non-finite value detected in output of " << name
                        << " at index " << i << std::endl;
            break;
          }
        }
      }
    } catch (const std::exception &e) {
      MUNET_ERROR << "CRASH in/after " << name << ": " << e.what() << std::endl;
      throw;
    }
  }

public:
  DebugBackend(std::shared_ptr<Backend> base) : base_(base) {}

  double get_last_kernel_time_us() override {
    return base_->get_last_kernel_time_us();
  }

  void *allocate(size_t bytes) override {
    Profiler::get().record_alloc(bytes);
    return base_->allocate(bytes);
  }
  void deallocate(void *ptr) override {
    // Note: In a real implementation, you'd need to store the size
    // to subtract it from Profiler::get().record_free()
    base_->deallocate(ptr);
  }

  void memset(void *ptr, int value, size_t bytes) override {
    MUNET_DEBUG << "memset | " << bytes << " bytes" << std::endl;
    Timer t;
    base_->memset(ptr, value, bytes);
    check("memset", t.elapsed_us());
  }
  void copy(const void *src, void *dst, size_t bytes, Device src_dev,
            Device dst_dev) override {
    MUNET_DEBUG << "copy | " << bytes << " bytes" << std::endl;
    Timer t;
    base_->copy(src, dst, bytes, src_dev, dst_dev);
    check("copy", t.elapsed_us());
  }
  void synchronize() override { base_->synchronize(); }
  void all_reduce(Storage &buffer, size_t num_elements) override {
    MUNET_DEBUG << "all_reduce | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->all_reduce(buffer, num_elements);
    check("all_reduce", t.elapsed_us(), &buffer);
  }

  void add(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override {
    MUNET_DEBUG << "add | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->add(a, b, out, num_elements);
    check("add", t.elapsed_us(), &out);
  }
  void sub(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override {
    MUNET_DEBUG << "sub | " << num_elements << " elements" << std::endl;
    base_->sub(a, b, out, num_elements);
    Timer t;
    check("sub", t.elapsed_us(), &out);
  }
  void mul(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override {
    MUNET_DEBUG << "mul | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->mul(a, b, out, num_elements);
    check("mul", t.elapsed_us(), &out);
  }
  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) override {
    MUNET_DEBUG << "matmul | " << M << "x" << K << "x" << N << " matrix"
                << (transA ? " (transposed)" : "")
                << (transB ? " (transposed)" : "") << std::endl;
    Timer t;
    base_->matmul(a, b, out, M, K, N, transA, transB);
    check("matmul", t.elapsed_us(), &out);
  }
  void concat(const std::vector<Storage *> &inputs, Storage &out, int dim,
              const std::vector<Shape> &shapes) override {
    MUNET_DEBUG << "concat | " << dim << " dimension" << std::endl;
    Timer t;
    base_->concat(inputs, out, dim, shapes);
    check("concat", t.elapsed_us(), &out);
  }
  void concat_backward(const Storage &grad_out,
                       std::vector<Storage *> &grad_inputs, int dim,
                       const std::vector<Shape> &shapes) override {
    MUNET_DEBUG << "concat_backward | " << dim << " dimension" << std::endl;
    Timer t;
    base_->concat_backward(grad_out, grad_inputs, dim, shapes);
    check("concat_backward", t.elapsed_us(), &grad_out);
  }
  void relu(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "relu | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->relu(in, out, num_elements);
    check("relu", t.elapsed_us(), &out);
  }
  void relu_backward(const Storage &grad_out, const Storage &input,
                     Storage &grad_in, size_t num_elements) override {
    MUNET_DEBUG << "relu_backward | " << num_elements << " elements"
                << std::endl;
    Timer t;
    base_->relu_backward(grad_out, input, grad_in, num_elements);
    check("relu_backward", t.elapsed_us(), &grad_in);
  }
  void sigmoid(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "sigmoid | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->sigmoid(in, out, num_elements);
    check("sigmoid", t.elapsed_us(), &out);
  }
  void sigmoid_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, size_t num_elements) override {
    MUNET_DEBUG << "sigmoid_backward | " << num_elements << " elements"
                << std::endl;
    Timer t;
    base_->sigmoid_backward(grad_out, out, grad_in, num_elements);
    check("sigmoid_backward", t.elapsed_us(), &grad_in);
  }
  void softmax(const Storage &in, Storage &out, int batch_size,
               int num_classes) override {
    MUNET_DEBUG << "softmax | " << batch_size << " batches, " << num_classes
                << " classes" << std::endl;
    Timer t;
    base_->softmax(in, out, batch_size, num_classes);
    check("softmax", t.elapsed_us(), &out);
  }
  void softmax_backward(const Storage &grad_out, const Storage &out,
                        Storage &grad_in, int batch_size,
                        int num_classes) override {
    MUNET_DEBUG << "softmax_backward | " << batch_size << " batches, "
                << num_classes << " classes" << std::endl;
    Timer t;
    base_->softmax_backward(grad_out, out, grad_in, batch_size, num_classes);
    check("softmax_backward", t.elapsed_us(), &grad_in);
  }
  void cross_entropy(const Storage &logits, const Storage &targets,
                     Storage &out_loss, int batch_size, int num_classes,
                     int spatial) override {
    MUNET_DEBUG << "cross_entropy | " << batch_size << " batches, "
                << num_classes << " classes" << std::endl;
    Timer t;
    base_->cross_entropy(logits, targets, out_loss, batch_size, num_classes,
                         spatial);
    check("cross_entropy", t.elapsed_us(), &out_loss);
  }
  void cross_entropy_backward(const Storage &grad_out, const Storage &logits,
                              const Storage &targets, Storage &grad_in,
                              int batch_size, int num_classes,
                              int spatial) override {
    MUNET_DEBUG << "cross_entropy_backward | " << batch_size << " batches, "
                << num_classes << " classes" << std::endl;
    Timer t;
    base_->cross_entropy_backward(grad_out, logits, targets, grad_in,
                                  batch_size, num_classes, spatial);
    check("cross_entropy_backward", t.elapsed_us(), &grad_in);
  }
  void mse_loss(const Storage &pred, const Storage &target, Storage &out_loss,
                size_t num_elements) override {
    MUNET_DEBUG << "mse_loss | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->mse_loss(pred, target, out_loss, num_elements);
    check("mse_loss", t.elapsed_us(), &out_loss);
  }
  void mse_loss_backward(const Storage &grad_out, const Storage &pred,
                         const Storage &target, Storage &grad_in,
                         size_t num_elements) override {
    MUNET_DEBUG << "mse_loss_backward | " << num_elements << " elements"
                << std::endl;
    Timer t;
    base_->mse_loss_backward(grad_out, pred, target, grad_in, num_elements);
    check("mse_loss_backward", t.elapsed_us(), &grad_in);
  }
  void conv2d(const Storage &in, const Storage &weight, const Storage *bias,
              Storage &out, int B, int iC, int iH, int iW, int oC, int kH,
              int kW, int s, int p) override {
    MUNET_DEBUG << "conv2d | " << B << " batches, " << iC << " input channels, "
                << iH << " input height, " << iW << " input width, " << oC
                << " output channels, " << kH << " kernel height, " << kW
                << " kernel width, " << s << " stride, " << p << " padding"
                << std::endl;
    Timer t;
    base_->conv2d(in, weight, bias, out, B, iC, iH, iW, oC, kH, kW, s, p);
    check("conv2d", t.elapsed_us(), &out);
  }
  void conv2d_backward(const Storage &grad_out, const Storage &in,
                       const Storage &weight, Storage &grad_in, Storage &grad_w,
                       Storage *grad_b, int B, int iC, int iH, int iW, int oC,
                       int kH, int kW, int s, int p) override {
    MUNET_DEBUG << "conv2d_backward | " << B << " batches, " << iC
                << " input channels, " << iH << " input height, " << iW
                << " input width, " << oC << " output channels, " << kH
                << " kernel height, " << kW << " kernel width, " << s
                << " stride, " << p << " padding" << std::endl;
    Timer t;
    base_->conv2d_backward(grad_out, in, weight, grad_in, grad_w, grad_b, B, iC,
                           iH, iW, oC, kH, kW, s, p);
    check("conv2d_backward", t.elapsed_us(), &grad_in);
  }
  void max_pool2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int k, int s, int p) override {
    MUNET_DEBUG << "max_pool2d | " << B << " batches, " << C << " channels, "
                << iH << " input height, " << iW << " input width, " << k
                << " kernel size, " << s << " stride, " << p << " padding"
                << std::endl;
    Timer t;
    base_->max_pool2d(in, out, B, C, iH, iW, k, s, p);
    check("max_pool2d", t.elapsed_us(), &out);
  }
  void max_pool2d_backward(const Storage &grad_out, const Storage &in,
                           Storage &grad_in, int B, int C, int iH, int iW,
                           int k, int s, int p) override {
    MUNET_DEBUG << "max_pool2d_backward | " << B << " batches, " << C
                << " channels, " << iH << " input height, " << iW
                << " input width, " << k << " kernel size, " << s << " stride, "
                << p << " padding" << std::endl;
    Timer t;
    base_->max_pool2d_backward(grad_out, in, grad_in, B, C, iH, iW, k, s, p);
    check("max_pool2d_backward", t.elapsed_us(), &grad_in);
  }
  void upsample2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int scale) override {
    MUNET_DEBUG << "upsample2d | " << B << " batches, " << C << " channels, "
                << iH << " input height, " << iW << " input width, " << scale
                << " scale factor" << std::endl;
    Timer t;
    base_->upsample2d(in, out, B, C, iH, iW, scale);
    check("upsample2d", t.elapsed_us(), &out);
  }
  void upsample2d_backward(const Storage &grad_out, Storage &grad_in, int B,
                           int C, int iH, int iW, int scale) override {
    MUNET_DEBUG << "upsample2d_backward | " << B << " batches, " << C
                << " channels, " << iH << " input height, " << iW
                << " input width, " << scale << " scale factor" << std::endl;
    Timer t;
    base_->upsample2d_backward(grad_out, grad_in, B, C, iH, iW, scale);
    check("upsample2d_backward", t.elapsed_us(), &grad_in);
  }
  void batch_norm(const Storage &in, const Storage &scale, const Storage &bias,
                  Storage &running_mean, Storage &running_var,
                  Storage &save_mean, Storage &save_var, Storage &out, int B,
                  int C, int H, int W, float momentum, float eps,
                  bool training) override {
    MUNET_DEBUG << "batch_norm | " << B << " batches, " << C << " channels, "
                << H << " height, " << W << " width" << std::endl;
    Timer t;
    base_->batch_norm(in, scale, bias, running_mean, running_var, save_mean,
                      save_var, out, B, C, H, W, momentum, eps, training);
    check("batch_norm", t.elapsed_us(), &out);
  }
  void batch_norm_backward(const Storage &grad_out, const Storage &in,
                           const Storage &scale, const Storage &save_mean,
                           const Storage &save_var, Storage &grad_in,
                           Storage &grad_scale, Storage &grad_bias, int B,
                           int C, int H, int W, float eps) override {
    MUNET_DEBUG << "batch_norm_backward | " << B << " batches, " << C
                << " channels, " << H << " height, " << W << " width"
                << std::endl;
    Timer t;
    base_->batch_norm_backward(grad_out, in, scale, save_mean, save_var,
                               grad_in, grad_scale, grad_bias, B, C, H, W, eps);
    check("batch_norm_backward", t.elapsed_us(), &grad_in);
  }
  void update(Storage &weight, const Storage &grad, float lr,
              size_t num_elements) override {
    MUNET_DEBUG << "update | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->update(weight, grad, lr, num_elements);
    check("update", t.elapsed_us(), &weight);
  }
  void fill_uniform(Storage &out, float low, float high,
                    size_t num_elements) override {
    MUNET_DEBUG << "fill_uniform | " << num_elements << " elements"
                << std::endl;
    Timer t;
    base_->fill_uniform(out, low, high, num_elements);
    check("fill_uniform", t.elapsed_us(), &out);
  }
  void sum(const Storage &in, Storage &out, size_t num_elements) override {
    MUNET_DEBUG << "sum | " << num_elements << " elements" << std::endl;
    Timer t;
    base_->sum(in, out, num_elements);
    check("sum", t.elapsed_us(), &out);
  }
};

} // anonymous namespace

// --- Backend Management ---
std::shared_ptr<Backend> BackendManager::get(Device device) {
  static std::unordered_map<int, std::shared_ptr<Backend>> cache;
  int key = (int)device.type * 1000 + device.index;
  if (cache.count(key))
    return cache.at(key);

  std::shared_ptr<Backend> base;
  if (device.type == DeviceType::CPU) {
    base = std::make_shared<CPUBackend>();
  }
#ifdef MUNET_USE_CUDA
  else if (device.type == DeviceType::CUDA) {
    base = std::make_shared<CUDABackend>(device.index);
  }
#endif
#ifdef MUNET_USE_VULKAN
  else if (device.type == DeviceType::VULKAN) {
    base = std::make_shared<VulkanBackend>();
  }
#endif
  else {
    throw std::runtime_error("Requested backend not compiled or implemented.");
  }

  if (munet::is_debug_enabled() || munet::is_profile_enabled()) {
    base = std::make_shared<DebugBackend>(base);
  }

  cache[key] = base;
  return base;
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

  // 1. Setup Profiling only if enabled
  bool profiling = is_profile_enabled();
  std::unique_ptr<Timer> timer;
  if (profiling) {
    impl_->backend().synchronize(); // Ensure source data is ready
    timer = std::make_unique<Timer>();
  }

  Tensor out(shape(), dev, dtype(), requires_grad());
  size_t byte_count = bytes();

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

  // 3. Finalize Profiling
  if (profiling) {
    out.impl_->backend().synchronize(); // Wait for transfer to complete
    double ms = timer->elapsed_us();
    std::string name = "to(" + dev.to_string() + ")";
    Profiler::get().record(name, ms, ms, byte_count, to_string(shape()));
  }

  // 4. Autograd and Tracing
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

Tensor Tensor::cat(const std::vector<Tensor> &inputs, int dim) {
  return ops::cat(inputs, dim);
}

Tensor Tensor::sum() const { return ops::sum(*this); }

Tensor Tensor::reshape(Shape new_shape) const {
  return ops::reshape(*this, new_shape);
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

void Tensor::uniform_(float low, float high) {
  if (size() == 0)
    return;
  // Delegate directly to backend. No more CPU roundtrip.
  impl_->backend().fill_uniform(*impl_->storage, low, high, size());
}

} // namespace munet
