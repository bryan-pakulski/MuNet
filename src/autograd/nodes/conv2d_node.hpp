#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct Conv2DBackward : public Node {
  Tensor in, weight, bias;
  int stride, padding;
  Conv2DBackward(Tensor i, Tensor w, Tensor b, int s, int p)
      : in(std::move(i)), weight(std::move(w)), bias(std::move(b)), stride(s),
        padding(p) {}
  std::string name() const override { return "Conv2DBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    Tensor grad_w(weight.shape(), weight.device(), weight.dtype());
    Tensor grad_b;
    if (bias.impl_) {
      grad_b = Tensor(bias.shape(), bias.device(), bias.dtype());
    }
    grad_in.impl_->storage->zero_();
    grad_w.impl_->storage->zero_();
    if (bias.impl_) {
      grad_b.impl_->storage->zero_();
    }

    const int B = in.shape()[0];
    const int iC = in.shape()[1];
    const int iH = in.shape()[2];
    const int iW = in.shape()[3];
    const int oC = weight.shape()[0];
    const int kH = weight.shape()[2];
    const int kW = weight.shape()[3];
    in.impl_->backend().conv2d_backward(
        *grad_out.impl_->storage, *in.impl_->storage, *weight.impl_->storage,
        *grad_in.impl_->storage, *grad_w.impl_->storage,
        bias.impl_ ? grad_b.impl_->storage.get() : nullptr, B, iC, iH, iW, oC,
        kH, kW, stride, padding);
    return {grad_in, grad_w, grad_b};
  }
};

} // namespace autograd_nodes
} // namespace munet
