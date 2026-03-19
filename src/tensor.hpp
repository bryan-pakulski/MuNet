#pragma once
#include "storage.hpp"
#include "types.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace munet {

struct Node;
class Tensor;

struct ForwardNode {
  std::string op_name;
  std::vector<Tensor> inputs;
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

  std::shared_ptr<ForwardNode> trace_node = nullptr;

  TensorImpl(Shape s, Device d, DataType dt, bool req_grad)
      : shape(std::move(s)), strides(default_strides(shape)),
        requires_grad(req_grad) {
    size_t bytes = numel(shape) * dtype_size(dt);
    storage = std::make_shared<Storage>(bytes, d, dt, shape);
  }

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
};

class Tensor {
public:
  Tensor() = default;

  Tensor(Shape shape, const TensorOptions &options) {
    impl_ = std::make_shared<TensorImpl>(std::move(shape), options.device,
                                         options.dtype,
                                         options.requires_grad);
  }

  Tensor(Shape shape, Device dev = Device{DeviceType::CPU, 0},
         DataType dtype = DataType::Float32, bool requires_grad = false)
      : Tensor(std::move(shape),
               TensorOptions{}.with_device(dev).with_dtype(dtype).with_requires_grad(
                   requires_grad)) {}

  const Shape &shape() const { return impl_->shape; }
  const Strides &strides() const { return impl_->strides; }
  size_t storage_offset() const { return impl_->storage_offset; }
  bool is_contiguous() const { return impl_->is_contiguous(); }
  Device device() const { return impl_->storage->device(); }
  DataType dtype() const { return impl_->storage->dtype(); }
  TensorOptions options() const {
    return TensorOptions{}.with_device(device()).with_dtype(dtype()).with_requires_grad(
        requires_grad());
  }
  void *data() { return impl_->storage->data(); }
  const void *data() const { return impl_->storage->data(); }
  size_t size() const { return numel(impl_->shape); }
  size_t bytes() const { return impl_->storage->size_bytes(); }

  const std::string &name() const { return impl_->name; }
  void set_name(const std::string &name) { impl_->name = name; }

  bool requires_grad() const { return impl_->requires_grad; }
  void set_requires_grad(bool r) { impl_->requires_grad = r; }

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
    if (impl_->grad)
      impl_->grad->storage->zero_();
  }

  void backward();
  void backward(const Tensor &grad);
  Tensor detach() const;

  Tensor clone() const {
    Tensor out(shape(), options());
    impl_->backend().copy(data(), out.data(), bytes(), device(), out.device());
    return out;
  }

  Tensor to(Device dev) const;
  Tensor to(DataType dtype) const;
  Tensor to(const TensorOptions &options) const;

  static Tensor cat(const std::vector<Tensor> &inputs, int dim = 1);
  Tensor operator+(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;
  Tensor operator/(const Tensor &other) const;
  Tensor matmul(const Tensor &other) const;
  Tensor relu() const;
  Tensor sigmoid() const;
  Tensor softmax(int dim = -1) const;
  Tensor log_softmax(int dim = -1) const;
  Tensor sum() const;
  Tensor reshape(Shape new_shape) const;
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
  void step(float lr);

  std::shared_ptr<TensorImpl> impl_ = nullptr;
};

} // namespace munet
