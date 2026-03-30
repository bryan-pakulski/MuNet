#pragma once
#include "storage.hpp"
#include "types.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace munet {

struct Node;
class Tensor;
struct TensorImpl;
struct AutogradExtension;

struct ForwardNode {
  std::string op_name;
  std::vector<Shape> input_shapes;
  std::vector<std::string> input_names;
  std::unordered_map<std::string, float> attributes;
  std::unordered_map<std::string, std::vector<int>> int_attributes;
};

struct TensorImpl {
  Shape shape;
  Strides strides;
  size_t storage_offset = 0;
  std::shared_ptr<Storage> storage;
  std::string name = "";

  bool requires_grad = false;
  std::shared_ptr<TensorImpl> grad = nullptr;
  std::shared_ptr<Node> grad_fn = nullptr;
  uint64_t version_counter = 0;

  std::shared_ptr<ForwardNode> trace_node = nullptr;

  TensorImpl(Shape s, Device d, DataType dt, bool req_grad)
      : shape(std::move(s)), strides(default_strides(shape)),
        requires_grad(req_grad) {
    size_t bytes = numel(shape) * dtype_size(dt);
    storage = std::make_shared<Storage>(bytes, d, dt, shape);
  }

  TensorImpl(std::shared_ptr<Storage> s, Shape sh, Strides st, size_t off)
      : shape(std::move(sh)), strides(std::move(st)), storage_offset(off),
        storage(std::move(s)), requires_grad(false) {}

  bool is_contiguous() const {
    int expected = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      if (strides[i] != expected)
        return false;
      expected *= shape[i];
    }
    return true;
  }

  Backend &backend() { return storage->backend(); }
  void bump_version() { ++version_counter; }
};

class Tensor {
public:
  using GradientHook = std::function<Tensor(const Tensor &)>;

  Tensor() = default;

  Tensor(Shape shape, const TensorOptions &options) {
    impl_ = std::make_shared<TensorImpl>(std::move(shape), options.device,
                                         options.dtype, options.requires_grad);
  }

  Tensor(Shape shape, Device dev = Device{DeviceType::CPU, 0},
         DataType dtype = DataType::Float32, bool requires_grad = false)
      : Tensor(std::move(shape), TensorOptions{}
                                     .with_device(dev)
                                     .with_dtype(dtype)
                                     .with_requires_grad(requires_grad)) {}

  const Shape &shape() const { return impl_->shape; }
  const Strides &strides() const { return impl_->strides; }
  size_t storage_offset() const { return impl_->storage_offset; }
  bool is_contiguous() const { return impl_->is_contiguous(); }
  Device device() const { return impl_->storage->device(); }
  DataType dtype() const { return impl_->storage->dtype(); }
  TensorOptions options() const {
    return TensorOptions{}
        .with_device(device())
        .with_dtype(dtype())
        .with_requires_grad(requires_grad());
  }
  void *data() {
    return static_cast<char *>(impl_->storage->data()) +
           storage_offset() * dtype_size(dtype());
  }
  const void *data() const {
    return static_cast<const char *>(impl_->storage->data()) +
           storage_offset() * dtype_size(dtype());
  }
  size_t size() const { return numel(impl_->shape); }
  size_t bytes() const { return size() * dtype_size(dtype()); }
  const std::string &name() const { return impl_->name; }
  void set_name(const std::string &name) { impl_->name = name; }

  bool requires_grad() const { return impl_->requires_grad; }
  void set_requires_grad(bool r) { impl_->requires_grad = r; }
  uint64_t version() const { return impl_->version_counter; }
  void bump_version() {
    if (impl_) {
      impl_->bump_version();
    }
  }

  Tensor transpose(int dim0, int dim1) const;
  Tensor permute(const std::vector<int> &dims) const;
  Tensor contiguous() const;

  Tensor grad() const {
    if (!impl_->grad)
      return Tensor();
    Tensor t;
    t.impl_ = impl_->grad;
    return t;
  }
  bool has_grad() const { return impl_->grad != nullptr; }

  void zero_grad() {
    if (impl_->grad) {
      impl_->grad->storage->zero_();
      impl_->grad->bump_version();
    }
  }

  void backward();
  void backward(const Tensor &grad);
  void backward(bool retain_graph);
  void backward(const Tensor &grad, bool retain_graph);
  Tensor detach() const;

  void register_gradient_hook(GradientHook hook) const;

  Tensor clone() const {
    Tensor out(shape(), options());
    impl_->backend().copy(data(), out.data(), bytes(), device(), out.device());
    return out;
  }

  Tensor to(Device dev) const;
  Tensor to(DataType dtype) const;
  Tensor to(const TensorOptions &options) const;
  // In-place device transfer - updates storage within existing TensorImpl
  void to_(Device dev);

  static Tensor cat(const std::vector<Tensor> &inputs, int dim = 1);
  Tensor operator+(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;
  Tensor operator/(const Tensor &other) const;
  Tensor matmul(const Tensor &other) const;
  Tensor relu() const;
  Tensor sigmoid() const;
  Tensor exp() const;
  Tensor log() const;
  Tensor sqrt() const;
  Tensor rsqrt() const;
  Tensor sin() const;
  Tensor cos() const;
  Tensor softmax(int dim = -1) const;
  Tensor log_softmax(int dim = -1) const;
  Tensor sum() const;
  Tensor mean(int dim = -1, bool keepdim = false) const;
  Tensor reshape(Shape new_shape) const;
  Tensor narrow(int dim, int start, int length) const;
  Tensor masked_fill(const Tensor &mask, const ScalarValue &value) const;
  Tensor masked_fill(const Tensor &mask, float value) const;
  ScalarValue item_value() const;
  float item() const;

  Tensor conv2d(const Tensor &weight, const Tensor &bias, int stride = 1,
                int padding = 0) const;
  Tensor max_pool2d(int kernel_size, int stride, int padding = 0) const;
  Tensor upsample2d(int scale_factor) const;

  Tensor batch_norm(Tensor &running_mean, Tensor &running_var,
                    const Tensor &weight, const Tensor &bias, bool training,
                    float momentum, float eps) const;
  Tensor layer_norm(const Tensor &weight, const Tensor &bias,
                    float eps = 1e-5f) const;

  Tensor mse_loss(const Tensor &target) const;
  Tensor cross_entropy(const Tensor &target) const;

  void uniform_(float low = -1.0f, float high = 1.0f);
  void fill_(const ScalarValue &value);
  void fill_(float value) { fill_(make_scalar(value)); }
  void fill_(int32_t value) { fill_(make_scalar(value)); }
  void step(float lr);

  std::shared_ptr<TensorImpl> impl_ = nullptr;
};

} // namespace munet
