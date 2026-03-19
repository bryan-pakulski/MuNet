#pragma once
#include "autograd/engine.hpp"
#include "types.hpp"
#include "core/util.hpp"
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
    throw std::runtime_error(op + ": backend '" +
                             std::string(tensor.impl_->backend().name()) +
                             "' does not support feature '" +
                             backend_feature_name(feature) + "' for dtype " +
                             dtype_name(tensor.dtype()));
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
    size_t off_a = 0, off_b = 0, curr = i;
    for (int d = ndim - 1; d >= 0; --d) {
      size_t coord = curr % info.out_shape[d];
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

  return (a.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(a.device());
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

  int ndim = static_cast<int>(t.shape().size());
  int out_ndim = static_cast<int>(target_shape.size());
  Strides out_strides = default_strides(target_shape);

  for (size_t i = 0; i < t_cpu.size(); ++i) {
    size_t out_off = 0;
    size_t curr = i;
    for (int d = ndim - 1; d >= 0; --d) {
      size_t coord = curr % t.shape()[d];
      curr /= t.shape()[d];

      int out_d_idx = d - (ndim - out_ndim);
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

  return (t.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(t.device());
}

inline Tensor matmul_cpu_fallback(const Tensor &a, const Tensor &b, bool transA,
                                  bool transB) {
  Device cpu{DeviceType::CPU, 0};
  Tensor a_cpu = a.to(cpu);
  Tensor b_cpu = b.to(cpu);

  int M = transA ? a.shape()[1] : a.shape()[0];
  int K = transA ? a.shape()[0] : a.shape()[1];
  int N = transB ? b.shape()[0] : b.shape()[1];

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

  return (a.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(a.device());
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

// 1. Forward Declarations (so structs can see the functions)
inline Tensor add(const Tensor &a, const Tensor &b);
inline Tensor sub(const Tensor &a, const Tensor &b);
inline Tensor mul(const Tensor &a, const Tensor &b);
inline Tensor div(const Tensor &a, const Tensor &b);
inline Tensor layer_norm(const Tensor &x, const Tensor &weight,
                         const Tensor &bias, float eps = 1e-5f);
inline Tensor masked_fill(const Tensor &a, const Tensor &mask,
                          const ScalarValue &value);
inline Tensor log_softmax(const Tensor &a, int dim = -1);

inline void link_backward_edges(Node *node, const std::vector<Tensor> &inputs) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &t = inputs[i];
    if (t.impl_->grad_fn) {
      node->next_edges.push_back({t.impl_->grad_fn, (int)i});
    } else if (t.requires_grad()) {
      auto acc_node = std::make_shared<AccumulateGrad>(t.impl_);
      node->next_edges.push_back({acc_node, (int)i});
    } else {
      node->next_edges.push_back({nullptr, 0});
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
    if (t.name().empty())
      t.impl_->name =
          "tensor_" +
          std::to_string(reinterpret_cast<uintptr_t>(t.impl_.get()));
    fn->input_names.push_back(t.name());
    fn->inputs.push_back(t);
  }
  fn->int_attributes = int_attrs;
  fn->attributes = float_attrs;
  out.impl_->trace_node = fn;
}

struct MaskedFillBackward : public Node {
  Tensor mask;
  Shape input_shape;

  explicit MaskedFillBackward(Tensor m)
      : mask(std::move(m)), input_shape(mask.shape()) {}

  std::string name() const override { return "MaskedFillBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
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
          static_cast<const char *>(go_cpu.data()) +
              i * dtype_size(grad_out.dtype()),
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

inline Tensor masked_fill(const Tensor &a, const Tensor &mask,
                          const ScalarValue &value) {
  if (a.shape() != mask.shape())
    throw std::runtime_error("masked_fill: input/mask shape mismatch");
  if (a.device() != mask.device())
    throw std::runtime_error("masked_fill: input/mask device mismatch");

  Device cpu{DeviceType::CPU, 0};
  Tensor a_cpu = a.to(cpu);
  Tensor m_cpu = mask.to(cpu);
  Tensor out_cpu(a.shape(), cpu, a.dtype());

  const char *av = static_cast<const char *>(a_cpu.data());
  const char *mv = static_cast<const char *>(m_cpu.data());
  char *ov = static_cast<char *>(out_cpu.data());
  const size_t a_stride = dtype_size(a.dtype());
  const size_t mask_stride = dtype_size(mask.dtype());
  const size_t out_stride = dtype_size(out_cpu.dtype());

  for (size_t i = 0; i < out_cpu.size(); ++i) {
    const ScalarValue mask_value =
        read_scalar_from_buffer(mv + i * mask_stride, mask.dtype());
    const ScalarValue input_value =
        read_scalar_from_buffer(av + i * a_stride, a.dtype());
    write_scalar_to_buffer(ov + i * out_stride, out_cpu.dtype(),
                           mask_value.is_nonzero() ? value.value
                                                   : input_value.value);
  }

  Tensor out = (a.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(a.device());

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<MaskedFillBackward>(mask);
    link_backward_edges(fn.get(), {a, mask});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }

  if (value.dtype == DataType::Int32) {
    record_trace(out, "MaskedFill", {a, mask},
                 {{"value", {value.as_int32()}}});
  } else {
    record_trace(out, "MaskedFill", {a, mask}, {}, {{"value", value.as_float()}});
  }
  return out;
}

inline Tensor sum_to_shape(const Tensor &t, const Shape &target_shape) {
  if (t.shape() == target_shape)
    return t;

  if (!detail::backend_supports(t, BackendFeature::Reduction)) {
    return detail::sum_to_shape_cpu_fallback(t, target_shape);
  }

  Tensor out(target_shape, t.device(), t.dtype());
  t.impl_->backend().sum_to_shape(*t.impl_->storage, *out.impl_->storage,
                                  t.shape(), target_shape);
  return out;
}

// --- ADD ---
struct AddBackward : public Node {
  Shape shape_a, shape_b;
  AddBackward(Shape a, Shape b) : shape_a(a), shape_b(b) {}
  std::string name() const override { return "AddBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    return {sum_to_shape(grads[0], shape_a), sum_to_shape(grads[0], shape_b)};
  }
};

inline Tensor broadcast_expand(const Tensor &src, int N) {
  Tensor out({N, (int)src.size()}, src.device(), src.dtype());
  src.impl_->backend().broadcast_row(*src.impl_->storage, *out.impl_->storage,
                                     N, src.size());
  return out;
}

inline Tensor expand_scalar(const Tensor &scalar, const Shape &target_shape) {
  Tensor out(target_shape, scalar.device(), scalar.dtype());
  out.fill_(scalar.item_value());
  return out;
}

inline Tensor add(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Add: device mismatch");
  detail::require_same_dtype("Add", a, b);

  auto info = compute_broadcast(a.shape(), a.strides(), b.shape(), b.strides());
  if (!info.can_broadcast) {
    throw std::runtime_error("Add: shape mismatch " + to_string(a.shape()) +
                             " vs " + to_string(b.shape()));
  }

  Tensor out = detail::backend_supports(a, BackendFeature::ElementwiseBinary)
                   ? Tensor(info.out_shape, a.device(), a.dtype())
                   : detail::binary_broadcast_cpu_fallback(
                         a, b, info, [](double lhs, double rhs) {
                           return lhs + rhs;
                         });
  if (detail::backend_supports(a, BackendFeature::ElementwiseBinary)) {
    a.impl_->backend().add(*a.impl_->storage, *b.impl_->storage,
                           *out.impl_->storage, info);
  }

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<AddBackward>(a.shape(), b.shape());
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Add", {a, b});
  return out;
}

// --- CONCAT ---
struct ConcatBackward : public Node {
  int dim;
  std::vector<Shape> input_shapes;
  ConcatBackward(int d, const std::vector<Shape> &s)
      : dim(d), input_shapes(s) {}

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

inline Tensor cat(const std::vector<Tensor> &inputs, int dim = 1) {
  if (inputs.empty())
    return Tensor();

  Shape out_shape = inputs[0].shape();
  std::vector<Shape> shapes;
  std::vector<Storage *> storages;
  int concat_size = 0;

  // Collect shapes and storages
  for (const auto &t : inputs) {
    const auto &s = t.shape();
    if (s.size() != out_shape.size())
      throw std::runtime_error(
          "All tensors must have the same number of dimensions for cat");

    // Check other dims match
    for (size_t d = 0; d < s.size(); ++d) {
      if (d != dim && s[d] != out_shape[d])
        throw std::runtime_error(
            "Tensor shapes must match except along concat dim");
    }

    concat_size += s[dim]; // Add size along the concat axis
    shapes.push_back(s);
    storages.push_back(t.impl_->storage.get());
  }

  // Set concatenated dimension in output shape
  out_shape[dim] = concat_size;

  Tensor out(out_shape, inputs[0].device(), inputs[0].dtype());

  // Call backend to perform actual concatenation
  inputs[0].impl_->backend().concat(storages, *out.impl_->storage, dim, shapes);

  // Setup backward graph if any input requires grad
  bool req = false;
  for (auto &t : inputs)
    if (t.requires_grad())
      req = true;

  if (GradMode::is_enabled() && req) {
    auto fn = std::make_shared<ConcatBackward>(dim, shapes);
    link_backward_edges(fn.get(), inputs);
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }

  record_trace(out, "Concat", inputs, {{"axis", {dim}}});
  return out;
}

// --- MSE LOSS ---
struct MSELossBackward : public Node {
  Tensor pred, target;
  MSELossBackward(Tensor p, Tensor t) : pred(p), target(t) {}
  std::string name() const override { return "MSELossBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(pred.shape(), pred.device(), pred.dtype());
    pred.impl_->backend().mse_loss_backward(
        *grad_out.impl_->storage, *pred.impl_->storage, *target.impl_->storage,
        *grad_in.impl_->storage, pred.size());
    // Only return gradient for predictions (targets don't need gradients)
    return {grad_in, Tensor()};
  }
};

inline Tensor mse_loss(const Tensor &pred, const Tensor &target) {
  detail::require_same_dtype("MSELoss", pred, target);
  detail::require_floating_dtype("MSELoss", pred);
  detail::require_backend_support("MSELoss", pred, BackendFeature::Loss);
  if (pred.shape() == target.shape()) {
    Tensor out({1}, pred.device(), pred.dtype());
    pred.impl_->backend().mse_loss(*pred.impl_->storage, *target.impl_->storage,
                                   *out.impl_->storage, pred.size());

    if (GradMode::is_enabled() && pred.requires_grad()) {
      auto fn = std::make_shared<MSELossBackward>(pred, target);
      link_backward_edges(fn.get(), {pred, target});
      out.set_requires_grad(true);
      out.impl_->grad_fn = fn;
    }
    record_trace(out, "MSELoss", {pred, target});
    return out;
  }

  // Auto-broadcast for NCHW if channels differ (1 vs C)
  if (pred.shape().size() == 4 && target.shape().size() == 4 &&
      pred.shape()[0] == target.shape()[0] &&
      pred.shape()[2] == target.shape()[2] &&
      pred.shape()[3] == target.shape()[3]) {

    int pC = pred.shape()[1];
    int tC = target.shape()[1];

    if (pC == 1 && tC > 1) {
      std::vector<Tensor> expanded(tC, pred);
      return mse_loss(cat(expanded, 1), target);
    }
    if (tC == 1 && pC > 1) {
      std::vector<Tensor> expanded(pC, target);
      return mse_loss(pred, cat(expanded, 1));
    }
  }

  throw std::runtime_error(
      "MSELoss: shape mismatch. Pred=" + to_string(pred.shape()) +
      " Target=" + to_string(target.shape()));
}

// --- CROSS ENTROPY LOSS ---
struct CrossEntropyBackward : public Node {
  Tensor logits, targets;
  int batch_size, num_classes, spatial;
  CrossEntropyBackward(Tensor l, Tensor t, int b, int c, int s)
      : logits(l), targets(t), batch_size(b), num_classes(c), spatial(s) {}
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

inline Tensor cross_entropy(const Tensor &logits, const Tensor &targets) {
  detail::require_same_dtype("CrossEntropy", logits, targets);
  detail::require_floating_dtype("CrossEntropy", logits);
  detail::require_backend_support("CrossEntropy", logits, BackendFeature::Loss);
  if (logits.shape() != targets.shape()) {
    throw std::runtime_error(
        "CrossEntropy: shape mismatch. Logits=" + to_string(logits.shape()) +
        " Targets=" + to_string(targets.shape()));
  }

  int batch_size = logits.shape().size() > 0 ? logits.shape()[0] : 1;
  int num_classes, spatial;

  if (logits.shape().size() == 4) {
    // NCHW
    num_classes = logits.shape()[1];
    spatial = logits.shape()[2] * logits.shape()[3];
  } else {
    // NC or flattened
    num_classes = logits.size() / batch_size;
    spatial = 1;
  }

  Tensor out({1}, logits.device(), logits.dtype());
  logits.impl_->backend().cross_entropy(
      *logits.impl_->storage, *targets.impl_->storage, *out.impl_->storage,
      batch_size, num_classes, spatial);

  if (GradMode::is_enabled() && logits.requires_grad()) {
    auto fn = std::make_shared<CrossEntropyBackward>(
        logits, targets, batch_size, num_classes, spatial);
    link_backward_edges(fn.get(), {logits, targets});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "CrossEntropy", {logits, targets});
  return out;
}



// --- LAYER NORM (CPU fallback with autograd) ---
struct LayerNormBackward : public Node {
  Tensor x, weight, bias;
  Tensor mean_cpu, inv_std_cpu;
  int rows, cols;
  LayerNormBackward(Tensor x_, Tensor w_, Tensor b_, Tensor m_, Tensor is_,
                    int r, int c)
      : x(x_), weight(w_), bias(b_), mean_cpu(m_), inv_std_cpu(is_), rows(r),
        cols(c) {}

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
          read_scalar_from_buffer(iv + r * acc_stride, inv_std_cpu.dtype()).value;
      double sum_gy = 0.0;
      double sum_gy_xhat = 0.0;

      for (int c = 0; c < cols; ++c) {
        int idx = r * cols + c;
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
        int idx = r * cols + c;
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

    Tensor dx_dev = (x.device().type == DeviceType::CPU) ? dx_cpu
                                                          : dx_cpu.to(x.device());
    Tensor dw_dev = (weight.device().type == DeviceType::CPU)
                        ? dw_cpu
                        : dw_cpu.to(weight.device());
    Tensor db_dev = (bias.device().type == DeviceType::CPU) ? db_cpu
                                                             : db_cpu.to(bias.device());
    return {dx_dev, dw_dev, db_dev};
  }
};

inline Tensor layer_norm(const Tensor &x, const Tensor &weight,
                         const Tensor &bias, float eps) {
  detail::require_same_dtype("LayerNorm", x, weight);
  detail::require_same_dtype("LayerNorm", x, bias);
  detail::require_floating_dtype("LayerNorm", x);
  if (x.shape().empty())
    throw std::runtime_error("LayerNorm: input must have at least 1 dim");

  int cols = x.shape().back();
  if ((int)weight.size() != cols || (int)bias.size() != cols)
    throw std::runtime_error("LayerNorm: weight/bias size must equal last dim");

  int rows = (int)(x.size() / cols);
  Device cpu{DeviceType::CPU, 0};

  Tensor x_cpu = x.to(cpu);
  Tensor w_cpu = weight.to(cpu);
  Tensor b_cpu = bias.to(cpu);

  Tensor out_cpu(x.shape(), cpu, x.dtype());
  const DataType acc_dtype =
      accumulation_type(AccumulationOp::Normalization, x.dtype());
  Tensor mean_cpu({rows}, cpu, acc_dtype, false);
  Tensor inv_std_cpu({rows}, cpu, acc_dtype, false);

  const char *xv = static_cast<const char *>(x_cpu.data());
  const char *wv = static_cast<const char *>(w_cpu.data());
  const char *bv = static_cast<const char *>(b_cpu.data());
  char *ov = static_cast<char *>(out_cpu.data());
  char *mv = static_cast<char *>(mean_cpu.data());
  char *iv = static_cast<char *>(inv_std_cpu.data());
  const size_t x_stride = dtype_size(x_cpu.dtype());
  const size_t w_stride = dtype_size(w_cpu.dtype());
  const size_t b_stride = dtype_size(b_cpu.dtype());
  const size_t out_stride = dtype_size(out_cpu.dtype());
  const size_t acc_stride = dtype_size(mean_cpu.dtype());

  for (int r = 0; r < rows; ++r) {
    double mean = 0.0;
    for (int c = 0; c < cols; ++c) {
      mean += read_scalar_from_buffer(xv + (r * cols + c) * x_stride,
                                      x_cpu.dtype())
                  .value;
    }
    mean /= cols;

    double var = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double x_value =
          read_scalar_from_buffer(xv + (r * cols + c) * x_stride, x_cpu.dtype())
              .value;
      const double d = x_value - mean;
      var += d * d;
    }
    var /= cols;

    const double inv = 1.0 / std::sqrt(var + eps);
    write_scalar_to_buffer(mv + r * acc_stride, mean_cpu.dtype(), mean);
    write_scalar_to_buffer(iv + r * acc_stride, inv_std_cpu.dtype(), inv);

    for (int c = 0; c < cols; ++c) {
      const double x_value =
          read_scalar_from_buffer(xv + (r * cols + c) * x_stride, x_cpu.dtype())
              .value;
      const double w_value =
          read_scalar_from_buffer(wv + c * w_stride, w_cpu.dtype()).value;
      const double b_value =
          read_scalar_from_buffer(bv + c * b_stride, b_cpu.dtype()).value;
      const double xhat = (x_value - mean) * inv;
      write_scalar_to_buffer(ov + (r * cols + c) * out_stride, out_cpu.dtype(),
                             xhat * w_value + b_value);
    }
  }

  Tensor out = (x.device().type == DeviceType::CPU) ? out_cpu
                                                     : out_cpu.to(x.device());

  if (GradMode::is_enabled() &&
      (x.requires_grad() || weight.requires_grad() || bias.requires_grad())) {
    auto fn = std::make_shared<LayerNormBackward>(x, weight, bias, mean_cpu,
                                                  inv_std_cpu, rows, cols);
    link_backward_edges(fn.get(), {x, weight, bias});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }

  record_trace(out, "LayerNorm", {x, weight, bias}, {}, {{"eps", eps}});
  return out;
}

// --- RELU ---
struct ReluBackward : public Node {
  Tensor saved_input;
  ReluBackward(Tensor a) : saved_input(a) {}
  std::string name() const override { return "ReluBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(saved_input.shape(), saved_input.device(),
                   saved_input.dtype());
    saved_input.impl_->backend().relu_backward(
        *grad_out.impl_->storage, *saved_input.impl_->storage,
        *grad_in.impl_->storage, saved_input.size());
    return {grad_in};
  }
};

inline Tensor relu(const Tensor &a) {
  Tensor out = detail::backend_supports(a, BackendFeature::UnaryActivation)
                   ? Tensor(a.shape(), a.device(), a.dtype())
                   : detail::unary_cpu_fallback(a, [](double v) {
                       return std::max(v, 0.0);
                     });
  if (detail::backend_supports(a, BackendFeature::UnaryActivation)) {
    a.impl_->backend().relu(*a.impl_->storage, *out.impl_->storage, a.size());
  } else if (!is_floating(a.dtype()) && a.dtype() != DataType::Int32) {
    detail::require_backend_support("Relu", a, BackendFeature::UnaryActivation);
  }

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<ReluBackward>(a);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Relu", {a});
  return out;
}

// --- SIGMOID ---
struct SigmoidBackward : public Node {
  Tensor saved_out;
  SigmoidBackward(Tensor o) : saved_out(o) {}
  std::string name() const override { return "SigmoidBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(saved_out.shape(), saved_out.device(), saved_out.dtype());
    saved_out.impl_->backend().sigmoid_backward(
        *grad_out.impl_->storage, *saved_out.impl_->storage,
        *grad_in.impl_->storage, saved_out.size());
    return {grad_in};
  }
};

inline Tensor sigmoid(const Tensor &a) {
  detail::require_floating_dtype("Sigmoid", a);
  Tensor out = detail::backend_supports(a, BackendFeature::UnaryActivation)
                   ? Tensor(a.shape(), a.device(), a.dtype())
                   : detail::unary_cpu_fallback(a, [](double v) {
                       return 1.0 / (1.0 + std::exp(-v));
                     });
  if (detail::backend_supports(a, BackendFeature::UnaryActivation)) {
    a.impl_->backend().sigmoid(*a.impl_->storage, *out.impl_->storage, a.size());
  }

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<SigmoidBackward>(out);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Sigmoid", {a});
  return out;
}

// --- SOFTMAX ---
struct SoftmaxBackward : public Node {
  Tensor saved_out;
  int dim;
  Shape shape;
  SoftmaxBackward(Tensor o, int d) : saved_out(o), dim(d), shape(o.shape()) {}
  std::string name() const override { return "SoftmaxBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Device cpu{DeviceType::CPU, 0};
    Tensor go_cpu = grads[0].to(cpu);
    Tensor out_cpu = saved_out.to(cpu);
    Tensor gi_cpu(shape, cpu, saved_out.dtype());

    int rank = static_cast<int>(shape.size());
    int resolved = (dim < 0) ? (rank + dim) : dim;
    int outer = 1, inner = 1, dim_size = shape[resolved];
    for (int i = 0; i < resolved; ++i)
      outer *= shape[i];
    for (int i = resolved + 1; i < rank; ++i)
      inner *= shape[i];

    const char *go = static_cast<const char *>(go_cpu.data());
    const char *out = static_cast<const char *>(out_cpu.data());
    char *gi = static_cast<char *>(gi_cpu.data());
    const size_t go_stride = dtype_size(go_cpu.dtype());
    const size_t out_stride = dtype_size(out_cpu.dtype());
    const size_t gi_stride = dtype_size(gi_cpu.dtype());

    for (int o = 0; o < outer; ++o) {
      for (int in = 0; in < inner; ++in) {
        double dot = 0.0;
        for (int d = 0; d < dim_size; ++d) {
          int idx = (o * dim_size + d) * inner + in;
          dot += read_scalar_from_buffer(go + idx * go_stride, go_cpu.dtype())
                     .value *
                 read_scalar_from_buffer(out + idx * out_stride, out_cpu.dtype())
                     .value;
        }
        for (int d = 0; d < dim_size; ++d) {
          int idx = (o * dim_size + d) * inner + in;
          const double out_value =
              read_scalar_from_buffer(out + idx * out_stride, out_cpu.dtype())
                  .value;
          const double go_value =
              read_scalar_from_buffer(go + idx * go_stride, go_cpu.dtype()).value;
          write_scalar_to_buffer(gi + idx * gi_stride, gi_cpu.dtype(),
                                 out_value * (go_value - dot));
        }
      }
    }

    Tensor gi_dev = (saved_out.device().type == DeviceType::CPU)
                        ? gi_cpu
                        : gi_cpu.to(saved_out.device());
    return {gi_dev};
  }
};

inline Tensor softmax(const Tensor &a, int dim = -1) {
  detail::require_floating_dtype("Softmax", a);
  if (a.shape().empty())
    throw std::runtime_error("Softmax expects non-empty shape");

  int rank = static_cast<int>(a.shape().size());
  int resolved = (dim < 0) ? (rank + dim) : dim;
  if (resolved < 0 || resolved >= rank)
    throw std::runtime_error("Softmax: dim out of range");
  if (resolved != rank - 1)
    throw std::runtime_error("Softmax currently supports only the last dimension");

  int num_classes = a.shape().back();
  int batch_size = a.size() / num_classes;
  Tensor out(a.shape(), a.device(), a.dtype());
  if (detail::backend_supports(a, BackendFeature::Softmax)) {
    a.impl_->backend().softmax(*a.impl_->storage, *out.impl_->storage,
                               batch_size, num_classes);
  } else {
    Device cpu{DeviceType::CPU, 0};
    Tensor a_cpu = a.to(cpu);
    Tensor out_cpu(a.shape(), cpu, a.dtype());
    const char *ip = static_cast<const char *>(a_cpu.data());
    char *op = static_cast<char *>(out_cpu.data());
    const size_t stride = dtype_size(a.dtype());
    for (int b = 0; b < batch_size; ++b) {
      double max_val = read_scalar_from_buffer(ip + b * num_classes * stride,
                                               a.dtype())
                           .value;
      for (int i = 1; i < num_classes; ++i) {
        max_val = std::max(
            max_val,
            read_scalar_from_buffer(ip + (b * num_classes + i) * stride,
                                    a.dtype())
                .value);
      }
      double sum_exp = 0.0;
      for (int i = 0; i < num_classes; ++i) {
        sum_exp += std::exp(read_scalar_from_buffer(
                                ip + (b * num_classes + i) * stride, a.dtype())
                                .value -
                            max_val);
      }
      for (int i = 0; i < num_classes; ++i) {
        const double prob =
            std::exp(read_scalar_from_buffer(
                         ip + (b * num_classes + i) * stride, a.dtype())
                         .value -
                     max_val) /
            sum_exp;
        write_scalar_to_buffer(op + (b * num_classes + i) * stride,
                               out_cpu.dtype(), prob);
      }
    }
    out = (a.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(a.device());
  }

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<SoftmaxBackward>(out, resolved);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Softmax", {a}, {{"dim", {resolved}}});
  return out;
}

struct LogSoftmaxBackward : public Node {
  Tensor saved_log_probs;
  int dim;
  Shape shape;
  LogSoftmaxBackward(Tensor lp, int d)
      : saved_log_probs(lp), dim(d), shape(lp.shape()) {}

  std::string name() const override { return "LogSoftmaxBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Device cpu{DeviceType::CPU, 0};
    Tensor go_cpu = grads[0].to(cpu);
    Tensor lp_cpu = saved_log_probs.to(cpu);
    Tensor gi_cpu(shape, cpu, saved_log_probs.dtype());

    int rank = static_cast<int>(shape.size());
    int resolved = (dim < 0) ? (rank + dim) : dim;
    int outer = 1, inner = 1, dim_size = shape[resolved];
    for (int i = 0; i < resolved; ++i)
      outer *= shape[i];
    for (int i = resolved + 1; i < rank; ++i)
      inner *= shape[i];

    const char *go = static_cast<const char *>(go_cpu.data());
    const char *lp = static_cast<const char *>(lp_cpu.data());
    char *gi = static_cast<char *>(gi_cpu.data());
    const size_t go_stride = dtype_size(go_cpu.dtype());
    const size_t lp_stride = dtype_size(lp_cpu.dtype());
    const size_t gi_stride = dtype_size(gi_cpu.dtype());

    for (int o = 0; o < outer; ++o) {
      for (int in = 0; in < inner; ++in) {
        double sum_go = 0.0;
        for (int d = 0; d < dim_size; ++d) {
          int idx = (o * dim_size + d) * inner + in;
          sum_go +=
              read_scalar_from_buffer(go + idx * go_stride, go_cpu.dtype()).value;
        }

        for (int d = 0; d < dim_size; ++d) {
          int idx = (o * dim_size + d) * inner + in;
          const double p =
              std::exp(read_scalar_from_buffer(lp + idx * lp_stride,
                                               lp_cpu.dtype())
                           .value);
          const double go_value =
              read_scalar_from_buffer(go + idx * go_stride, go_cpu.dtype()).value;
          write_scalar_to_buffer(gi + idx * gi_stride, gi_cpu.dtype(),
                                 go_value - p * sum_go);
        }
      }
    }

    Tensor gi_dev = (saved_log_probs.device().type == DeviceType::CPU)
                        ? gi_cpu
                        : gi_cpu.to(saved_log_probs.device());
    return {gi_dev};
  }
};

inline Tensor log_softmax(const Tensor &a, int dim) {
  detail::require_floating_dtype("LogSoftmax", a);
  Tensor p = softmax(a, dim);

  Device cpu{DeviceType::CPU, 0};
  Tensor p_cpu = p.to(cpu);
  Tensor out_cpu(a.shape(), cpu, a.dtype());
  const char *pv = static_cast<const char *>(p_cpu.data());
  char *ov = static_cast<char *>(out_cpu.data());
  const size_t stride = dtype_size(a.dtype());
  for (size_t i = 0; i < p_cpu.size(); ++i) {
    const double prob =
        read_scalar_from_buffer(pv + i * stride, p_cpu.dtype()).value;
    write_scalar_to_buffer(ov + i * stride, out_cpu.dtype(),
                           std::log(std::max(prob, 1e-20)));
  }

  Tensor out = (a.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(a.device());

  int rank = static_cast<int>(a.shape().size());
  int resolved = (dim < 0) ? (rank + dim) : dim;
  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<LogSoftmaxBackward>(out, resolved);
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "LogSoftmax", {a}, {{"dim", {resolved}}});
  return out;
}

// --- MATMUL ---
inline Tensor matmul_internal(const Tensor &a, const Tensor &b, bool transA,
                              bool transB) {
  detail::require_same_dtype("Matmul", a, b);
  if (a.shape().size() != 2 || b.shape().size() != 2)
    throw std::runtime_error("Matmul currently requires 2D tensors");

  int M = transA ? a.shape()[1] : a.shape()[0];
  int K = transA ? a.shape()[0] : a.shape()[1];
  int N = transB ? b.shape()[0] : b.shape()[1];

  if (!detail::backend_supports(a, BackendFeature::Matmul)) {
    return detail::matmul_cpu_fallback(a, b, transA, transB);
  }

  Tensor out({M, N}, a.device(), a.dtype());
  a.impl_->backend().matmul(*a.impl_->storage, *b.impl_->storage,
                            *out.impl_->storage, M, K, N, transA, transB);
  return out;
}

struct MatmulBackward : public Node {
  Tensor A, B;
  MatmulBackward(Tensor a, Tensor b) : A(a), B(b) {}
  std::string name() const override { return "MatmulBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_a, grad_b;

    // dA = dC @ B^T
    if (next_edges.size() > 0 && next_edges[0].node) {
      grad_a = matmul_internal(grad_out, B, false, true);
    } else {
      grad_a = Tensor();
    }

    // dB = A^T @ dC
    if (next_edges.size() > 1 && next_edges[1].node) {
      grad_b = matmul_internal(A, grad_out, true, false);
    } else {
      grad_b = Tensor();
    }

    return {grad_a, grad_b};
  }
};

inline Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Matmul: device mismatch");

  Tensor out = matmul_internal(a, b, false, false);

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<MatmulBackward>(a, b);
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "MatMul", {a, b});
  return out;
}

// --- SUB ---
struct SubBackward : public Node {
  Shape shape_a, shape_b;
  SubBackward(Shape a, Shape b) : shape_a(a), shape_b(b) {}
  std::string name() const override { return "SubBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    // db = -grad_out.
    Tensor zeros(grad_out.shape(), grad_out.device(), grad_out.dtype());
    zeros.impl_->storage->zero_();
    Tensor neg_grad_out = sub(zeros, grad_out); // Recursive call is safe here

    return {sum_to_shape(grad_out, shape_a),
            sum_to_shape(neg_grad_out, shape_b)};
  }
};

inline Tensor sub(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Sub: device mismatch");
  detail::require_same_dtype("Sub", a, b);

  auto info = compute_broadcast(a.shape(), a.strides(), b.shape(), b.strides());
  if (!info.can_broadcast) {
    throw std::runtime_error("Sub: shape mismatch " + to_string(a.shape()) +
                             " vs " + to_string(b.shape()));
  }

  Tensor out = detail::backend_supports(a, BackendFeature::ElementwiseBinary)
                   ? Tensor(info.out_shape, a.device(), a.dtype())
                   : detail::binary_broadcast_cpu_fallback(
                         a, b, info, [](double lhs, double rhs) {
                           return lhs - rhs;
                         });
  if (detail::backend_supports(a, BackendFeature::ElementwiseBinary)) {
    a.impl_->backend().sub(*a.impl_->storage, *b.impl_->storage,
                           *out.impl_->storage, info);
  }

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<SubBackward>(a.shape(), b.shape());
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Sub", {a, b});
  return out;
}

// --- MUL ---
struct MulBackward : public Node {
  Tensor A, B;
  MulBackward(Tensor a, Tensor b) : A(a), B(b) {}
  std::string name() const override { return "MulBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    // dC/dA = B * grad_out, then reduce to A's shape
    Tensor da = mul(B, grad_out);
    // dC/dB = A * grad_out, then reduce to B's shape
    Tensor db = mul(A, grad_out);

    return {sum_to_shape(da, A.shape()), sum_to_shape(db, B.shape())};
  }
};

inline Tensor mul(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Mul: device mismatch");
  detail::require_same_dtype("Mul", a, b);

  auto info = compute_broadcast(a.shape(), a.strides(), b.shape(), b.strides());
  if (!info.can_broadcast) {
    throw std::runtime_error("Mul: shape mismatch");
  }

  Tensor out = detail::backend_supports(a, BackendFeature::ElementwiseBinary)
                   ? Tensor(info.out_shape, a.device(), a.dtype())
                   : detail::binary_broadcast_cpu_fallback(
                         a, b, info, [](double lhs, double rhs) {
                           return lhs * rhs;
                         });
  if (detail::backend_supports(a, BackendFeature::ElementwiseBinary)) {
    a.impl_->backend().mul(*a.impl_->storage, *b.impl_->storage,
                           *out.impl_->storage, info);
  }

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<MulBackward>(a, b);
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Mul", {a, b});
  return out;
}

// --- DIV ---
struct DivBackward : public Node {
  Tensor A, B;
  DivBackward(Tensor a, Tensor b) : A(a), B(b) {}
  std::string name() const override { return "DivBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor da = div(grad_out, B);

    Tensor b2 = mul(B, B);
    Tensor num = mul(grad_out, A);
    Tensor db_pos = div(num, b2);
    Tensor zeros(db_pos.shape(), db_pos.device(), db_pos.dtype());
    zeros.impl_->storage->zero_();
    Tensor db = sub(zeros, db_pos);

    return {sum_to_shape(da, A.shape()), sum_to_shape(db, B.shape())};
  }
};

inline Tensor div(const Tensor &a, const Tensor &b) {
  if (a.device() != b.device())
    throw std::runtime_error("Div: device mismatch");
  detail::require_same_dtype("Div", a, b);
  detail::require_floating_dtype("Div", a);

  auto info = compute_broadcast(a.shape(), a.strides(), b.shape(), b.strides());
  if (!info.can_broadcast) {
    throw std::runtime_error("Div: shape mismatch");
  }

  Tensor out = detail::backend_supports(a, BackendFeature::ElementwiseBinary)
                   ? Tensor(info.out_shape, a.device(), a.dtype())
                   : detail::binary_broadcast_cpu_fallback(
                         a, b, info, [](double lhs, double rhs) {
                           return lhs / rhs;
                         });
  if (detail::backend_supports(a, BackendFeature::ElementwiseBinary)) {
    a.impl_->backend().div(*a.impl_->storage, *b.impl_->storage,
                           *out.impl_->storage, info);
  }

  if (GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())) {
    auto fn = std::make_shared<DivBackward>(a, b);
    link_backward_edges(fn.get(), {a, b});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Div", {a, b});
  return out;
}

// --- SUM (Global) ---
struct SumBackward : public Node {
  Shape shape;
  Device dev;
  SumBackward(Shape s, Device d) : shape(s), dev(d) {}
  std::string name() const override { return "SumBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    // grad_out is a scalar on the device.
    // We need to broadcast it.
    // For simplicity, we can just copy that 1 value to a full tensor.
    // Since we don't have a "Broadcast" kernel yet, we can fallback to CPU for
    // the BACKWARD pass only or implement a fill kernel. For now, let's keep
    // the backward pass simple, but ensure FORWARD is fast.

    Tensor grad_out_cpu = grads[0].to(Device{DeviceType::CPU, 0});
    const ScalarValue g =
        read_scalar_from_buffer(grad_out_cpu.data(), grad_out_cpu.dtype());

    Tensor cpu_grad_in(shape, Device{DeviceType::CPU, 0}, grads[0].dtype());
    cpu_grad_in.fill_(make_scalar(g.value, cpu_grad_in.dtype()));

    return {cpu_grad_in.to(dev)};
  }
};

inline Tensor sum(const Tensor &a) {
  if (!detail::backend_supports(a, BackendFeature::Reduction)) {
    Device cpu{DeviceType::CPU, 0};
    Tensor a_cpu = a.to(cpu);
    Tensor out_cpu({1}, cpu, a.dtype());
    double total = 0.0;
    const char *ip = static_cast<const char *>(a_cpu.data());
    const size_t stride = dtype_size(a.dtype());
    for (size_t i = 0; i < a_cpu.size(); ++i) {
      total += read_scalar_from_buffer(ip + i * stride, a.dtype()).value;
    }
    write_scalar_to_buffer(out_cpu.data(), out_cpu.dtype(), total);
    Tensor out = (a.device().type == DeviceType::CPU) ? out_cpu : out_cpu.to(a.device());

    if (GradMode::is_enabled() && a.requires_grad()) {
      auto fn = std::make_shared<SumBackward>(a.shape(), a.device());
      link_backward_edges(fn.get(), {a});
      out.set_requires_grad(true);
      out.impl_->grad_fn = fn;
    }
    record_trace(out, "ReduceSum", {a}, {{"keepdims", {0}}});
    return out;
  }

  Tensor out({1}, a.device(), a.dtype());
  a.impl_->backend().sum(*a.impl_->storage, *out.impl_->storage, a.size());

  if (GradMode::is_enabled() && a.requires_grad()) {
    auto fn = std::make_shared<SumBackward>(a.shape(), a.device());
    link_backward_edges(fn.get(), {a});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "ReduceSum", {a}, {{"keepdims", {0}}});
  return out;
}

struct ReshapeBackward : public Node {
  Shape input_shape;
  ReshapeBackward(Shape s) : input_shape(s) {}
  std::string name() const override { return "ReshapeBackward"; }

  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    // Gradient just needs to be reshaped back to original input layout
    return {grads[0].reshape(input_shape)};
  }
};

inline Tensor reshape(const Tensor &in, Shape new_shape) {
  size_t src_el = numel(in.shape());
  size_t dst_el = numel(new_shape);
  if (src_el != dst_el)
    throw std::runtime_error("Reshape: element count mismatch");

  Tensor out;
  // Create new TensorImpl pointing to the SAME storage (View)
  out.impl_ = std::make_shared<TensorImpl>(new_shape, in.device(), in.dtype(),
                                           in.requires_grad());
  out.impl_->storage = in.impl_->storage;

  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn = std::make_shared<ReshapeBackward>(in.shape());
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Reshape", {in}, {{"shape", new_shape}});
  return out;
}

// --- CONV2D ---
struct Conv2DBackward : public Node {
  Tensor in, weight, bias;
  int stride, padding;
  Conv2DBackward(Tensor i, Tensor w, Tensor b, int s, int p)
      : in(i), weight(w), bias(b), stride(s), padding(p) {}
  std::string name() const override { return "Conv2DBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    Tensor grad_w(weight.shape(), weight.device(), weight.dtype());
    Tensor grad_b;
    if (bias.impl_)
      grad_b = Tensor(bias.shape(), bias.device(), bias.dtype());
    grad_in.impl_->storage->zero_();
    grad_w.impl_->storage->zero_();
    if (bias.impl_)
      grad_b.impl_->storage->zero_();
    int B = in.shape()[0], iC = in.shape()[1], iH = in.shape()[2],
        iW = in.shape()[3];
    int oC = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
    in.impl_->backend().conv2d_backward(
        *grad_out.impl_->storage, *in.impl_->storage, *weight.impl_->storage,
        *grad_in.impl_->storage, *grad_w.impl_->storage,
        bias.impl_ ? grad_b.impl_->storage.get() : nullptr, B, iC, iH, iW, oC,
        kH, kW, stride, padding);
    return {grad_in, grad_w, grad_b};
  }
};
inline Tensor conv2d(const Tensor &in, const Tensor &weight, const Tensor &bias,
                     int stride, int padding) {
  detail::require_same_dtype("Conv2d", in, weight);
  if (bias.impl_)
    detail::require_same_dtype("Conv2d", in, bias);
  detail::require_floating_dtype("Conv2d", in);
  detail::require_backend_support("Conv2d", in, BackendFeature::Convolution);
  if (in.device() != weight.device() ||
      (bias.impl_ && bias.device() != in.device())) {
    MUNET_ERROR << "conv2d: inputs not on same device: "
                << in.device().to_string()
                << " != " << weight.device().to_string() << std::endl;
    throw std::runtime_error("Conv2d: inputs must be on same device");
  }

  if (in.shape().size() != 4 || weight.shape().size() != 4) {
    MUNET_ERROR << "conv2d: inputs must be 4D, in.shape: "
                << to_string(in.shape())
                << " weight shape: " << to_string(weight.shape()) << std::endl;
    throw std::runtime_error("Conv2d: inputs must be 4D (NCHW)");
  }

  if (in.shape()[1] != weight.shape()[1]) {
    MUNET_ERROR << "conv2d: channel mismatch. input=" << in.shape()[1]
                << ", weight=" << weight.shape()[1] << std::endl;
    throw std::runtime_error("Conv2d: input channels mismatch. Input=" +
                             std::to_string(in.shape()[1]) +
                             ", Weight=" + std::to_string(weight.shape()[1]));
  }

  int B = in.shape()[0], iC = in.shape()[1], iH = in.shape()[2],
      iW = in.shape()[3];
  int oC = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
  int oH = (iH + 2 * padding - kH) / stride + 1;
  int oW = (iW + 2 * padding - kW) / stride + 1;
  Tensor out({B, oC, oH, oW}, in.device(), in.dtype());
  in.impl_->backend().conv2d(*in.impl_->storage, *weight.impl_->storage,
                             bias.impl_ ? bias.impl_->storage.get() : nullptr,
                             *out.impl_->storage, B, iC, iH, iW, oC, kH, kW,
                             stride, padding);
  if (GradMode::is_enabled() && (in.requires_grad() || weight.requires_grad() ||
                                 (bias.impl_ && bias.requires_grad()))) {
    auto fn =
        std::make_shared<Conv2DBackward>(in, weight, bias, stride, padding);
    std::vector<Tensor> inputs = {in, weight};
    if (bias.impl_)
      inputs.push_back(bias);
    link_backward_edges(fn.get(), inputs);
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  std::vector<Tensor> inputs = {in, weight};
  if (bias.impl_)
    inputs.push_back(bias);
  record_trace(out, "Conv", inputs,
               {{"strides", {stride, stride}},
                {"pads", {padding, padding, padding, padding}}});
  return out;
}
// --- MAXPOOL2D ---
struct MaxPool2DBackward : public Node {
  Tensor in;
  int k, s, p;
  MaxPool2DBackward(Tensor i, int k_, int s_, int p_)
      : in(i), k(k_), s(s_), p(p_) {}
  std::string name() const override { return "MaxPool2DBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    grad_in.impl_->storage->zero_();
    int B = in.shape()[0], C = in.shape()[1], iH = in.shape()[2],
        iW = in.shape()[3];
    in.impl_->backend().max_pool2d_backward(
        *grads[0].impl_->storage, *in.impl_->storage, *grad_in.impl_->storage,
        B, C, iH, iW, k, s, p);
    return {grad_in};
  }
};
inline Tensor max_pool2d(const Tensor &in, int kernel_size, int stride,
                         int padding) {
  detail::require_floating_dtype("MaxPool2d", in);
  detail::require_backend_support("MaxPool2d", in, BackendFeature::Pooling);
  int B = in.shape()[0], C = in.shape()[1], iH = in.shape()[2],
      iW = in.shape()[3];
  int oH = (iH + 2 * padding - kernel_size) / stride + 1;
  int oW = (iW + 2 * padding - kernel_size) / stride + 1;
  Tensor out({B, C, oH, oW}, in.device(), in.dtype());
  in.impl_->backend().max_pool2d(*in.impl_->storage, *out.impl_->storage, B, C,
                                 iH, iW, kernel_size, stride, padding);

  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn =
        std::make_shared<MaxPool2DBackward>(in, kernel_size, stride, padding);
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "MaxPool", {in},
               {{"kernel_shape", {kernel_size, kernel_size}},
                {"strides", {stride, stride}},
                {"pads", {padding, padding, padding, padding}}});
  return out;
}
// --- UPSAMPLE2D ---
struct Upsample2DBackward : public Node {
  Tensor in;
  int scale;
  Upsample2DBackward(Tensor i, int sc) : in(i), scale(sc) {}
  std::string name() const override { return "Upsample2DBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    grad_in.impl_->storage->zero_();
    int B = in.shape()[0], C = in.shape()[1], iH = in.shape()[2],
        iW = in.shape()[3];
    in.impl_->backend().upsample2d_backward(
        *grads[0].impl_->storage, *grad_in.impl_->storage, B, C, iH, iW, scale);
    return {grad_in};
  }
};
inline Tensor upsample2d(const Tensor &in, int scale_factor) {
  detail::require_floating_dtype("Upsample2d", in);
  detail::require_backend_support("Upsample2d", in, BackendFeature::Pooling);
  int B = in.shape()[0], C = in.shape()[1], iH = in.shape()[2],
      iW = in.shape()[3];
  Tensor out({B, C, iH * scale_factor, iW * scale_factor}, in.device(),
             in.dtype());
  in.impl_->backend().upsample2d(*in.impl_->storage, *out.impl_->storage, B, C,
                                 iH, iW, scale_factor);

  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn = std::make_shared<Upsample2DBackward>(in, scale_factor);
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Upsample2d", {in}, {{"scale", {scale_factor}}});
  return out;
}

// --- BATCH NORM ---
struct BatchNormBackward : public Node {
  Tensor in, weight, save_mean, save_var;
  float eps;
  BatchNormBackward(Tensor i, Tensor w, Tensor sm, Tensor sv, float e)
      : in(i), weight(w), save_mean(sm), save_var(sv), eps(e) {}
  std::string name() const override { return "BatchNormBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    Tensor grad_out = grads[0];
    Tensor grad_in(in.shape(), in.device(), in.dtype());
    Tensor grad_scale(weight.shape(), weight.device(), weight.dtype());
    Tensor grad_bias(weight.shape(), weight.device(), weight.dtype());

    int B = in.shape()[0], C = in.shape()[1], H = in.shape()[2],
        W = in.shape()[3];
    in.impl_->backend().batch_norm_backward(
        *grad_out.impl_->storage, *in.impl_->storage, *weight.impl_->storage,
        *save_mean.impl_->storage, *save_var.impl_->storage,
        *grad_in.impl_->storage, *grad_scale.impl_->storage,
        *grad_bias.impl_->storage, B, C, H, W, eps);

    // Inputs: in (0), weight (1), bias (2)
    return {grad_in, grad_scale, grad_bias};
  }
};

inline Tensor batch_norm(const Tensor &in, Tensor &running_mean,
                         Tensor &running_var, const Tensor &weight,
                         const Tensor &bias, bool training, float momentum,
                         float eps) {
  detail::require_same_dtype("BatchNorm", in, weight);
  detail::require_same_dtype("BatchNorm", in, bias);
  detail::require_floating_dtype("BatchNorm", in);
  detail::require_backend_support("BatchNorm", in, BackendFeature::BatchNorm);
  int B = in.shape()[0], C = in.shape()[1], H = in.shape()[2],
      W = in.shape()[3];
  Tensor out(in.shape(), in.device(), in.dtype());

  // We need to save mean/var for backward if training
  const DataType stats_dtype =
      accumulation_type(AccumulationOp::Normalization, in.dtype());
  Tensor save_mean({C}, in.device(), stats_dtype);
  Tensor save_var({C}, in.device(), stats_dtype);

  in.impl_->backend().batch_norm(
      *in.impl_->storage, *weight.impl_->storage, *bias.impl_->storage,
      *running_mean.impl_->storage, *running_var.impl_->storage,
      *save_mean.impl_->storage, *save_var.impl_->storage, *out.impl_->storage,
      B, C, H, W, momentum, eps, training);

  if (GradMode::is_enabled() && training &&
      (in.requires_grad() || weight.requires_grad() || bias.requires_grad())) {
    auto fn = std::make_shared<BatchNormBackward>(in, weight, save_mean,
                                                  save_var, eps);
    // Link inputs: input, weight, bias. Running stats do not participate in
    // gradient graph for backprop.
    link_backward_edges(fn.get(), {in, weight, bias});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "BatchNormalization",
               {in, weight, bias, running_mean, running_var}, {},
               {{"epsilon", eps}, {"momentum", momentum}});
  return out;
}

struct TransposeBackward : public Node {
  int d0, d1;
  TransposeBackward(int dim0, int dim1) : d0(dim0), d1(dim1) {}
  std::string name() const override { return "TransposeBackward"; }
  std::vector<Tensor> apply(const std::vector<Tensor> &grads) override {
    // Transpose is its own inverse
    return {grads[0].transpose(d0, d1)};
  }
};

inline Tensor transpose(const Tensor &in, int dim0, int dim1) {
  Tensor out = in.transpose(dim0, dim1);
  if (GradMode::is_enabled() && in.requires_grad()) {
    auto fn = std::make_shared<TransposeBackward>(dim0, dim1);
    link_backward_edges(fn.get(), {in});
    out.set_requires_grad(true);
    out.impl_->grad_fn = fn;
  }
  record_trace(out, "Transpose", {in}, {{"dims", {dim0, dim1}}});
  return out;
}

inline Tensor zeros(Shape shape, Device device = Device{DeviceType::CPU, 0},
                    bool requires_grad = false,
                    DataType dtype = DataType::Float32) {
  Tensor t(shape, device, dtype, requires_grad);
  t.impl_->storage->zero_();
  return t;
}

} // namespace ops
} // namespace munet
