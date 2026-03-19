#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct MSELossBackward : public Node {
  Tensor pred, target;
  MSELossBackward(Tensor p, Tensor t) : pred(std::move(p)), target(std::move(t)) {}
  std::string name() const override { return "MSELossBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(pred.shape(), pred.device(), pred.dtype());
    pred.impl_->backend().mse_loss_backward(
        *grad_out.impl_->storage, *pred.impl_->storage, *target.impl_->storage,
        *grad_in.impl_->storage, pred.size());
    return {grad_in, Tensor()};
  }
};

struct CrossEntropyBackward : public Node {
  Tensor logits, targets;
  int batch_size, num_classes, spatial;
  CrossEntropyBackward(Tensor l, Tensor t, int b, int c, int s)
      : logits(std::move(l)), targets(std::move(t)), batch_size(b),
        num_classes(c), spatial(s) {}
  std::string name() const override { return "CrossEntropyBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(logits.shape(), logits.device(), logits.dtype());
    logits.impl_->backend().cross_entropy_backward(
        *grad_out.impl_->storage, *logits.impl_->storage,
        *targets.impl_->storage, *grad_in.impl_->storage, batch_size,
        num_classes, spatial);
    return {grad_in, Tensor()};
  }
};

} // namespace autograd_nodes
} // namespace munet
