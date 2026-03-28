#pragma once

#include "backend.hpp"
#include "tensor.hpp"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace munet {
namespace ops {

enum class OpId {
  Add,
  Sub,
  Mul,
  Div,
  MaskedFill,
  Matmul,
  Relu,
  Sigmoid,
  Exp,
  Log,
  Sqrt,
  Rsqrt,
  Sin,
  Cos,
  Softmax,
  LogSoftmax,
  Cat,
  Sum,
  SumToShape,
  Mean,
  Reshape,
  Conv2D,
  MaxPool2D,
  Upsample2D,
  BatchNorm,
  LayerNorm,
  MSELoss,
  CrossEntropy,
  Transpose,
  Narrow,
  Zeros,
};

struct OpMetadata {
  OpId id;
  const char *name;
  const char *trace_name;
  std::optional<BackendFeature> feature;
  bool requires_floating = false;
  BackendFallbackPolicy fallback_policy =
      BackendFallbackPolicy::ExplicitUnsupported;
};

struct DispatchDecision {
  const OpMetadata &metadata;
  bool use_backend = false;
  bool use_cpu_fallback = false;
  BackendSupport backend_support;
};

const OpMetadata &op_metadata(OpId id);
DispatchDecision resolve_dispatch(OpId id, const Tensor &tensor);
std::string dispatch_policy_snapshot();
std::string dispatch_decision_debug_dump(OpId id, const Tensor &tensor);
void record_registered_trace(
    OpId id, Tensor &out, const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, std::vector<int>> &int_attrs = {},
    const std::unordered_map<std::string, float> &float_attrs = {});

} // namespace ops
} // namespace munet
