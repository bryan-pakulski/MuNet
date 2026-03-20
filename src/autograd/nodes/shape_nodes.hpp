#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct ConcatBackward : public Node {
  int dim;
  std::vector<Shape> input_shapes;
  ConcatBackward(int d, std::vector<Shape> s)
      : dim(d), input_shapes(std::move(s)) {}

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    std::vector<Tensor> grad_ins;
    std::vector<Storage *> grad_stors;
    for (const auto &s : input_shapes) {
      grad_ins.emplace_back(s, grad_out.device(), grad_out.dtype());
      grad_stors.push_back(grad_ins.back().impl_->storage.get());
    }
    grad_out.impl_->backend().concat_backward(*grad_out.impl_->storage,
                                              grad_stors, dim, input_shapes);
    return grad_ins;
  }
};

struct SumBackward : public Node {
  Shape shape;
  Device dev;
  SumBackward(Shape s, Device d) : shape(std::move(s)), dev(d) {}
  std::string name() const override { return "SumBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out_cpu = grads[0].to(Device{DeviceType::CPU, 0});
    const ScalarValue g =
        read_scalar_from_buffer(grad_out_cpu.data(), grad_out_cpu.dtype());
    Tensor cpu_grad_in(shape, Device{DeviceType::CPU, 0}, grads[0].dtype());
    cpu_grad_in.fill_(make_scalar(g.value, cpu_grad_in.dtype()));
    return {cpu_grad_in.to(dev)};
  }
};

struct MeanBackward : public Node {
  Shape input_shape;
  int dim;
  bool keepdim;
  Device dev;
  DataType dtype;
  MeanBackward(Shape s, int d, bool k, Device device, DataType dt)
      : input_shape(std::move(s)), dim(d), keepdim(k), dev(device), dtype(dt) {}
  std::string name() const override { return "MeanBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Device cpu{DeviceType::CPU, 0};
    Tensor grad_out = grads[0].to(cpu).contiguous();
    Tensor grad_in(input_shape, cpu, dtype);
    grad_in.fill_(0.0f);

    const int rank = static_cast<int>(input_shape.size());
    const int resolved = (dim < 0) ? (rank + dim) : dim;
    const int reduce_size = input_shape[resolved];
    Shape grad_shape = grad_out.shape();
    Shape out_shape = keepdim ? grad_shape : grad_shape;
    if (!keepdim) {
      out_shape = input_shape;
      out_shape.erase(out_shape.begin() + resolved);
    }
    Strides out_strides = default_strides(out_shape);
    Strides in_strides = default_strides(input_shape);
    const size_t elem_size = dtype_size(dtype);
    const char *go = static_cast<const char *>(grad_out.data());
    char *gi = static_cast<char *>(grad_in.data());
    for (size_t linear = 0; linear < numel(input_shape); ++linear) {
      size_t curr = linear;
      size_t out_off = 0;
      int out_dim = 0;
      for (int d = 0; d < rank; ++d) {
        const int coord = static_cast<int>(curr / in_strides[d]);
        curr %= static_cast<size_t>(in_strides[d]);
        if (d == resolved) {
          if (keepdim) {
            out_off += 0;
            ++out_dim;
          }
          continue;
        }
        out_off += static_cast<size_t>(coord) * out_strides[out_dim++];
      }
      if (keepdim) {
        // Skip the reduced coordinate in grad_out, which is always 0.
      }
      const double value =
          read_scalar_from_buffer(go + out_off * elem_size, grad_out.dtype())
              .value /
          static_cast<double>(reduce_size);
      write_scalar_to_buffer(gi + linear * elem_size, dtype, value);
    }
    return {grad_in.to(dev)};
  }
};

struct ReshapeBackward : public Node {
  Shape input_shape;
  explicit ReshapeBackward(Shape s) : input_shape(std::move(s)) {}
  std::string name() const override { return "ReshapeBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    return {grads[0].reshape(input_shape)};
  }
};

struct TransposeBackward : public Node {
  int d0, d1;
  TransposeBackward(int dim0, int dim1) : d0(dim0), d1(dim1) {}
  std::string name() const override { return "TransposeBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    return {grads[0].transpose(d0, d1)};
  }
};

struct NarrowBackward : public Node {
  Shape input_shape;
  int dim;
  int start;
  int length;
  Device dev;
  DataType dtype;
  NarrowBackward(Shape shape, int dim_, int start_, int length_, Device device,
                 DataType dt)
      : input_shape(std::move(shape)), dim(dim_), start(start_),
        length(length_), dev(device), dtype(dt) {}
  std::string name() const override { return "NarrowBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Device cpu{DeviceType::CPU, 0};
    Tensor grad_out = grads[0].to(cpu).contiguous();
    Tensor grad_in(input_shape, cpu, dtype);
    grad_in.fill_(0.0f);

    Strides in_strides = default_strides(input_shape);
    Shape out_shape = input_shape;
    out_shape[dim] = length;
    Strides out_strides = default_strides(out_shape);
    const size_t elem_size = dtype_size(dtype);
    const char *go = static_cast<const char *>(grad_out.data());
    char *gi = static_cast<char *>(grad_in.data());

    for (size_t linear = 0; linear < numel(out_shape); ++linear) {
      size_t curr = linear;
      size_t dst_off = 0;
      for (size_t d = 0; d < out_shape.size(); ++d) {
        const int coord = static_cast<int>(curr / out_strides[d]);
        curr %= static_cast<size_t>(out_strides[d]);
        const int input_coord =
            (static_cast<int>(d) == dim) ? (start + coord) : coord;
        dst_off += static_cast<size_t>(input_coord) * in_strides[d];
      }
      const ScalarValue value =
          read_scalar_from_buffer(go + linear * elem_size, grad_out.dtype());
      write_scalar_to_buffer(gi + dst_off * elem_size, dtype, value.value);
    }
    return {grad_in.to(dev)};
  }
};

} // namespace autograd_nodes
} // namespace munet
