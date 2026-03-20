#pragma once

#include "../../autograd/engine.hpp"
#include "../../core/op_dispatch.hpp"
#include "../../core/util.hpp"
#include "../../types.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace munet {
namespace ops {

namespace detail {

inline void require_same_dtype(const std::string &op, const Tensor &a,
                               const Tensor &b) {
  if (a.dtype() != b.dtype()) {
    throw std::runtime_error(op + ": dtype mismatch " + dtype_name(a.dtype()) +
                             " vs " + dtype_name(b.dtype()));
  }
}

inline void require_floating_dtype(const std::string &op, const Tensor &a) {
  if (!is_floating(a.dtype())) {
    throw std::runtime_error(op + " requires a floating-point tensor, got " +
                             dtype_name(a.dtype()));
  }
}

inline bool backend_supports(const Tensor &tensor, BackendFeature feature) {
  return tensor.impl_->backend().supports(feature, tensor.dtype());
}

inline void require_backend_support(const std::string &op, const Tensor &tensor,
                                    BackendFeature feature) {
  if (!backend_supports(tensor, feature)) {
    throw std::runtime_error(
        op + ": backend '" + std::string(tensor.impl_->backend().name()) +
        "' does not support feature '" + backend_feature_name(feature) +
        "' for dtype " + dtype_name(tensor.dtype()));
  }
}

template <typename Fn>
inline Tensor binary_broadcast_cpu_fallback(const Tensor &a, const Tensor &b,
                                            const BroadcastInfo &info,
                                            Fn &&fn) {
  Device cpu{DeviceType::CPU, 0};
  Tensor a_cpu = a.to(cpu);
  Tensor b_cpu = b.to(cpu);
  Tensor out_cpu(info.out_shape, cpu, a.dtype());

  const char *ap = static_cast<const char *>(a_cpu.data());
  const char *bp = static_cast<const char *>(b_cpu.data());
  char *op = static_cast<char *>(out_cpu.data());
  const size_t a_stride = dtype_size(a.dtype());
  const size_t b_stride = dtype_size(b.dtype());
  const size_t out_stride = dtype_size(out_cpu.dtype());
  const size_t total = numel(info.out_shape);
  const int ndim = static_cast<int>(info.out_shape.size());

  for (size_t i = 0; i < total; ++i) {
    size_t off_a = 0;
    size_t off_b = 0;
    size_t curr = i;
    for (int d = ndim - 1; d >= 0; --d) {
      const size_t coord = curr % info.out_shape[d];
      off_a += coord * info.strides_a[d];
      off_b += coord * info.strides_b[d];
      curr /= info.out_shape[d];
    }

    const ScalarValue lhs =
        read_scalar_from_buffer(ap + off_a * a_stride, a.dtype());
    const ScalarValue rhs =
        read_scalar_from_buffer(bp + off_b * b_stride, b.dtype());
    write_scalar_to_buffer(op + i * out_stride, out_cpu.dtype(),
                           fn(lhs.value, rhs.value));
  }

  return (a.device().type == DeviceType::CPU) ? out_cpu
                                              : out_cpu.to(a.device());
}

inline Tensor sum_to_shape_cpu_fallback(const Tensor &t,
                                        const Shape &target_shape) {
  Device cpu{DeviceType::CPU, 0};
  Tensor t_cpu = t.to(cpu);
  Tensor out_cpu(target_shape, cpu, t.dtype());
  out_cpu.fill_(make_scalar(0.0, out_cpu.dtype()));

  const char *ip = static_cast<const char *>(t_cpu.data());
  char *op = static_cast<char *>(out_cpu.data());
  const size_t in_stride = dtype_size(t.dtype());
  const size_t out_stride = dtype_size(out_cpu.dtype());

  const int ndim = static_cast<int>(t.shape().size());
  const int out_ndim = static_cast<int>(target_shape.size());
  const Strides out_strides = default_strides(target_shape);

  for (size_t i = 0; i < t_cpu.size(); ++i) {
    size_t out_off = 0;
    size_t curr = i;
    for (int d = ndim - 1; d >= 0; --d) {
      const size_t coord = curr % t.shape()[d];
      curr /= t.shape()[d];

      const int out_d_idx = d - (ndim - out_ndim);
      if (out_d_idx >= 0 && target_shape[out_d_idx] != 1) {
        out_off += coord * out_strides[out_d_idx];
      }
    }

    const ScalarValue accum =
        read_scalar_from_buffer(op + out_off * out_stride, out_cpu.dtype());
    const ScalarValue val =
        read_scalar_from_buffer(ip + i * in_stride, t.dtype());
    write_scalar_to_buffer(op + out_off * out_stride, out_cpu.dtype(),
                           accum.value + val.value);
  }

  return (t.device().type == DeviceType::CPU) ? out_cpu
                                              : out_cpu.to(t.device());
}

inline Tensor matmul_cpu_fallback(const Tensor &a, const Tensor &b, bool transA,
                                  bool transB) {
  Device cpu{DeviceType::CPU, 0};
  Tensor a_cpu = a.to(cpu);
  Tensor b_cpu = b.to(cpu);

  const int M = transA ? a.shape()[1] : a.shape()[0];
  const int K = transA ? a.shape()[0] : a.shape()[1];
  const int N = transB ? b.shape()[0] : b.shape()[1];

  Tensor out_cpu({M, N}, cpu, a.dtype());
  const char *ap = static_cast<const char *>(a_cpu.data());
  const char *bp = static_cast<const char *>(b_cpu.data());
  char *cp = static_cast<char *>(out_cpu.data());
  const size_t a_stride = dtype_size(a.dtype());
  const size_t b_stride = dtype_size(b.dtype());
  const size_t out_stride = dtype_size(out_cpu.dtype());

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        const int a_index = transA ? (k * M + m) : (m * K + k);
        const int b_index = transB ? (n * K + k) : (k * N + n);
        const ScalarValue a_val =
            read_scalar_from_buffer(ap + a_index * a_stride, a.dtype());
        const ScalarValue b_val =
            read_scalar_from_buffer(bp + b_index * b_stride, b.dtype());
        sum += a_val.value * b_val.value;
      }
      write_scalar_to_buffer(cp + (m * N + n) * out_stride, out_cpu.dtype(),
                             sum);
    }
  }

  return (a.device().type == DeviceType::CPU) ? out_cpu
                                              : out_cpu.to(a.device());
}

template <typename Fn>
inline Tensor unary_cpu_fallback(const Tensor &input, Fn &&fn) {
  Device cpu{DeviceType::CPU, 0};
  Tensor in_cpu = input.to(cpu);
  Tensor out_cpu(input.shape(), cpu, input.dtype());
  const char *ip = static_cast<const char *>(in_cpu.data());
  char *op = static_cast<char *>(out_cpu.data());
  const size_t stride = dtype_size(input.dtype());
  for (size_t i = 0; i < in_cpu.size(); ++i) {
    const ScalarValue value =
        read_scalar_from_buffer(ip + i * stride, input.dtype());
    write_scalar_to_buffer(op + i * stride, out_cpu.dtype(), fn(value.value));
  }
  return (input.device().type == DeviceType::CPU) ? out_cpu
                                                  : out_cpu.to(input.device());
}

} // namespace detail

Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, const Tensor &b);
Tensor div(const Tensor &a, const Tensor &b);
Tensor masked_fill(const Tensor &a, const Tensor &mask,
                   const ScalarValue &value);
Tensor matmul_internal(const Tensor &a, const Tensor &b, bool transA,
                       bool transB);
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor relu(const Tensor &a);
Tensor sigmoid(const Tensor &a);
Tensor softmax(const Tensor &a, int dim);
Tensor log_softmax(const Tensor &a, int dim);
Tensor cat(const std::vector<Tensor> &inputs, int dim);
Tensor sum(const Tensor &a);
Tensor reshape(const Tensor &in, Shape new_shape);
Tensor conv2d(const Tensor &in, const Tensor &weight, const Tensor &bias,
              int stride, int padding);
Tensor max_pool2d(const Tensor &in, int kernel_size, int stride, int padding);
Tensor upsample2d(const Tensor &in, int scale_factor);
Tensor batch_norm(const Tensor &in, Tensor &running_mean, Tensor &running_var,
                  const Tensor &weight, const Tensor &bias, bool training,
                  float momentum, float eps);
Tensor layer_norm(const Tensor &x, const Tensor &weight, const Tensor &bias,
                  float eps);
Tensor mse_loss(const Tensor &pred, const Tensor &target);
Tensor cross_entropy(const Tensor &logits, const Tensor &targets);
Tensor transpose(const Tensor &in, int dim0, int dim1);
Tensor zeros(Shape shape, Device device, bool requires_grad, DataType dtype);

inline void link_backward_edges(Node *node, const std::vector<Tensor> &inputs) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &t = inputs[i];
    if (t.impl_->grad_fn) {
      node->next_edges.push_back({t.impl_->grad_fn, static_cast<int>(i),
                                  "input_" + std::to_string(i)});
    } else if (t.requires_grad()) {
      auto acc_node = std::make_shared<AccumulateGrad>(t.impl_);
      node->next_edges.push_back(
          {acc_node, static_cast<int>(i), "input_" + std::to_string(i)});
    } else {
      node->next_edges.push_back({nullptr, 0, "input_" + std::to_string(i)});
    }
  }
}

inline void record_trace(
    Tensor &out, const std::string &op_name, const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, std::vector<int>> &int_attrs = {},
    const std::unordered_map<std::string, float> &float_attrs = {}) {
  auto fn = std::make_shared<ForwardNode>();
  fn->op_name = op_name;
  for (const auto &t : inputs) {
    if (t.name().empty()) {
      t.impl_->name =
          "tensor_" +
          std::to_string(reinterpret_cast<uintptr_t>(t.impl_.get()));
    }
    fn->input_names.push_back(t.name());
    fn->inputs.push_back(t);
  }
  fn->int_attributes = int_attrs;
  fn->attributes = float_attrs;
  out.impl_->trace_node = fn;
}

inline Tensor sum_to_shape(const Tensor &t, const Shape &target_shape) {
  if (t.shape() == target_shape) {
    return t;
  }

  if (!detail::backend_supports(t, BackendFeature::Reduction)) {
    return detail::sum_to_shape_cpu_fallback(t, target_shape);
  }

  Tensor out(target_shape, t.device(), t.dtype());
  t.impl_->backend().sum_to_shape(*t.impl_->storage, *out.impl_->storage,
                                  t.shape(), target_shape);
  return out;
}

} // namespace ops
} // namespace munet
