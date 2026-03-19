#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct MaskedFillBackward : public Node {
  Shape input_shape;

  explicit MaskedFillBackward(Tensor m) : input_shape(m.shape()) {
    save_tensor(m, "masked_fill_mask");
  }

  std::string name() const override { return "MaskedFillBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor mask = saved_tensor(0);
    const Tensor &grad_out = grads[0];
    Device cpu{DeviceType::CPU, 0};
    Tensor go_cpu = grad_out.to(cpu);
    Tensor mask_cpu = mask.to(cpu);
    Tensor gi_cpu(input_shape, cpu, grad_out.dtype());

    for (size_t i = 0; i < gi_cpu.size(); ++i) {
      const ScalarValue mask_value = read_scalar_from_buffer(
          static_cast<const char *>(mask_cpu.data()) + i * dtype_size(mask.dtype()),
          mask.dtype());
      const ScalarValue grad_value = read_scalar_from_buffer(
          static_cast<const char *>(go_cpu.data()) + i * dtype_size(grad_out.dtype()),
          grad_out.dtype());
      write_scalar_to_buffer(
          static_cast<char *>(gi_cpu.data()) + i * dtype_size(gi_cpu.dtype()),
          gi_cpu.dtype(), mask_value.is_nonzero() ? 0.0 : grad_value.value);
    }

    Tensor gi_dev = (grad_out.device().type == DeviceType::CPU)
                        ? gi_cpu
                        : gi_cpu.to(grad_out.device());
    return {gi_dev, Tensor()};
  }
};

struct SubBackward : public Node {
  Shape shape_a, shape_b;
  SubBackward(Shape a, Shape b) : shape_a(std::move(a)), shape_b(std::move(b)) {}
  std::string name() const override { return "SubBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor zeros(grad_out.shape(), grad_out.device(), grad_out.dtype());
    zeros.impl_->storage->zero_();
    Tensor neg_grad_out = ops::sub(zeros, grad_out);
    return {ops::sum_to_shape(grad_out, shape_a),
            ops::sum_to_shape(neg_grad_out, shape_b)};
  }
};

struct MulBackward : public Node {
  Shape shape_a, shape_b;
  MulBackward(Tensor a, Tensor b) : shape_a(a.shape()), shape_b(b.shape()) {
    save_tensor(a, "mul_lhs");
    save_tensor(b, "mul_rhs");
  }
  std::string name() const override { return "MulBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor A = saved_tensor(0);
    Tensor B = saved_tensor(1);
    Tensor grad_out = grads[0];
    Tensor da = ops::mul(B, grad_out);
    Tensor db = ops::mul(A, grad_out);
    return {ops::sum_to_shape(da, shape_a), ops::sum_to_shape(db, shape_b)};
  }
};

struct DivBackward : public Node {
  Shape shape_a, shape_b;
  DivBackward(Tensor a, Tensor b) : shape_a(a.shape()), shape_b(b.shape()) {
    save_tensor(a, "div_lhs");
    save_tensor(b, "div_rhs");
  }
  std::string name() const override { return "DivBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor A = saved_tensor(0);
    Tensor B = saved_tensor(1);
    Tensor grad_out = grads[0];
    Tensor da = ops::div(grad_out, B);
    Tensor b2 = ops::mul(B, B);
    Tensor num = ops::mul(grad_out, A);
    Tensor db_pos = ops::div(num, b2);
    Tensor zeros(db_pos.shape(), db_pos.device(), db_pos.dtype());
    zeros.impl_->storage->zero_();
    Tensor db = ops::sub(zeros, db_pos);
    return {ops::sum_to_shape(da, shape_a), ops::sum_to_shape(db, shape_b)};
  }
};

} // namespace autograd_nodes
} // namespace munet
