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

} // namespace autograd_nodes
} // namespace munet
