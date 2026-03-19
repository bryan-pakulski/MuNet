#pragma once

#include "../../core/ops/common.hpp"

namespace munet {
namespace autograd_nodes {

struct BatchNormBackward : public Node {
  Tensor in, weight, save_mean, save_var;
  float eps;
  BatchNormBackward(Tensor i, Tensor w, Tensor sm, Tensor sv, float e)
      : in(std::move(i)), weight(std::move(w)), save_mean(std::move(sm)),
        save_var(std::move(sv)), eps(e) {}
  std::string name() const override { return "BatchNormBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    Tensor grad_scale(weight.shape(), weight.device(), weight.dtype());
    Tensor grad_bias(weight.shape(), weight.device(), weight.dtype());

    const int B = in.shape()[0];
    const int C = in.shape()[1];
    const int H = in.shape()[2];
    const int W = in.shape()[3];
    in.impl_->backend().batch_norm_backward(
        *grad_out.impl_->storage, *in.impl_->storage, *weight.impl_->storage,
        *save_mean.impl_->storage, *save_var.impl_->storage,
        *grad_in.impl_->storage, *grad_scale.impl_->storage,
        *grad_bias.impl_->storage, B, C, H, W, eps);

    return {grad_in, grad_scale, grad_bias};
  }
};

struct LayerNormBackward : public Node {
  Tensor x, weight, bias;
  Tensor mean_cpu, inv_std_cpu;
  int rows, cols;
  LayerNormBackward(Tensor x_, Tensor w_, Tensor b_, Tensor m_, Tensor is_,
                    int r, int c)
      : x(std::move(x_)), weight(std::move(w_)), bias(std::move(b_)),
        mean_cpu(std::move(m_)), inv_std_cpu(std::move(is_)), rows(r), cols(c) {}

  std::string name() const override { return "LayerNormBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Device cpu{DeviceType::CPU, 0};
    Tensor go_cpu = grads[0].to(cpu);
    Tensor x_cpu = x.to(cpu);
    Tensor w_cpu = weight.to(cpu);

    Tensor dx_cpu(x.shape(), cpu, x.dtype());
    Tensor dw_cpu(weight.shape(), cpu, weight.dtype());
    Tensor db_cpu(bias.shape(), cpu, bias.dtype());
    dw_cpu.fill_(make_scalar(0.0, dw_cpu.dtype()));
    db_cpu.fill_(make_scalar(0.0, db_cpu.dtype()));

    const char *go = static_cast<const char *>(go_cpu.data());
    const char *xv = static_cast<const char *>(x_cpu.data());
    const char *wv = static_cast<const char *>(w_cpu.data());
    const char *mv = static_cast<const char *>(mean_cpu.data());
    const char *iv = static_cast<const char *>(inv_std_cpu.data());

    char *dx = static_cast<char *>(dx_cpu.data());
    char *dw = static_cast<char *>(dw_cpu.data());
    char *db = static_cast<char *>(db_cpu.data());

    const size_t go_stride = dtype_size(go_cpu.dtype());
    const size_t x_stride = dtype_size(x_cpu.dtype());
    const size_t w_stride = dtype_size(w_cpu.dtype());
    const size_t acc_stride = dtype_size(mean_cpu.dtype());
    const size_t dx_stride = dtype_size(dx_cpu.dtype());
    const size_t dw_stride = dtype_size(dw_cpu.dtype());
    const size_t db_stride = dtype_size(db_cpu.dtype());

    for (int r = 0; r < rows; ++r) {
      const double mean =
          read_scalar_from_buffer(mv + r * acc_stride, mean_cpu.dtype()).value;
      const double inv =
          read_scalar_from_buffer(iv + r * acc_stride, inv_std_cpu.dtype())
              .value;
      double sum_gy = 0.0;
      double sum_gy_xhat = 0.0;

      for (int c = 0; c < cols; ++c) {
        const int idx = r * cols + c;
        const double x_value =
            read_scalar_from_buffer(xv + idx * x_stride, x_cpu.dtype()).value;
        const double go_value =
            read_scalar_from_buffer(go + idx * go_stride, go_cpu.dtype()).value;
        const double w_value =
            read_scalar_from_buffer(wv + c * w_stride, w_cpu.dtype()).value;
        const double xhat = (x_value - mean) * inv;
        const double gy = go_value * w_value;
        sum_gy += gy;
        sum_gy_xhat += gy * xhat;

        const double dw_prev =
            read_scalar_from_buffer(dw + c * dw_stride, dw_cpu.dtype()).value;
        const double db_prev =
            read_scalar_from_buffer(db + c * db_stride, db_cpu.dtype()).value;
        write_scalar_to_buffer(dw + c * dw_stride, dw_cpu.dtype(),
                               dw_prev + go_value * xhat);
        write_scalar_to_buffer(db + c * db_stride, db_cpu.dtype(),
                               db_prev + go_value);
      }

      for (int c = 0; c < cols; ++c) {
        const int idx = r * cols + c;
        const double x_value =
            read_scalar_from_buffer(xv + idx * x_stride, x_cpu.dtype()).value;
        const double go_value =
            read_scalar_from_buffer(go + idx * go_stride, go_cpu.dtype()).value;
        const double w_value =
            read_scalar_from_buffer(wv + c * w_stride, w_cpu.dtype()).value;
        const double xhat = (x_value - mean) * inv;
        const double gy = go_value * w_value;
        write_scalar_to_buffer(dx + idx * dx_stride, dx_cpu.dtype(),
                               (inv / cols) *
                                   (cols * gy - sum_gy - xhat * sum_gy_xhat));
      }
    }

    Tensor dx_dev =
        (x.device().type == DeviceType::CPU) ? dx_cpu : dx_cpu.to(x.device());
    Tensor dw_dev = (weight.device().type == DeviceType::CPU)
                        ? dw_cpu
                        : dw_cpu.to(weight.device());
    Tensor db_dev = (bias.device().type == DeviceType::CPU)
                        ? db_cpu
                        : db_cpu.to(bias.device());
    return {dx_dev, dw_dev, db_dev};
  }
};

} // namespace autograd_nodes
} // namespace munet
