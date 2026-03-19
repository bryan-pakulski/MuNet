#include "op_dispatch.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace munet {
namespace ops {
namespace {

const std::unordered_map<OpId, OpMetadata> &registry() {
  static const std::unordered_map<OpId, OpMetadata> kRegistry = {
      {OpId::Add,
       {OpId::Add, "Add", "Add", BackendFeature::ElementwiseBinary, false,
        true}},
      {OpId::Sub,
       {OpId::Sub, "Sub", "Sub", BackendFeature::ElementwiseBinary, false,
        true}},
      {OpId::Mul,
       {OpId::Mul, "Mul", "Mul", BackendFeature::ElementwiseBinary, false,
        true}},
      {OpId::Div,
       {OpId::Div, "Div", "Div", BackendFeature::ElementwiseBinary, true,
        true}},
      {OpId::MaskedFill,
       {OpId::MaskedFill, "MaskedFill", "MaskedFill", std::nullopt, false,
        true}},
      {OpId::Matmul,
       {OpId::Matmul, "Matmul", "MatMul", BackendFeature::Matmul, false,
        true}},
      {OpId::Relu,
       {OpId::Relu, "Relu", "Relu", BackendFeature::UnaryActivation, false,
        true}},
      {OpId::Sigmoid,
       {OpId::Sigmoid, "Sigmoid", "Sigmoid",
        BackendFeature::UnaryActivation, true, true}},
      {OpId::Softmax,
       {OpId::Softmax, "Softmax", "Softmax", BackendFeature::Softmax, true,
        true}},
      {OpId::LogSoftmax,
       {OpId::LogSoftmax, "LogSoftmax", "LogSoftmax", std::nullopt, true,
        true}},
      {OpId::Cat,
       {OpId::Cat, "Concat", "Concat", BackendFeature::Concat, false,
        false}},
      {OpId::Sum,
       {OpId::Sum, "Sum", "ReduceSum", BackendFeature::Reduction, false,
        true}},
      {OpId::Reshape,
       {OpId::Reshape, "Reshape", "Reshape", std::nullopt, false, true}},
      {OpId::Conv2D,
       {OpId::Conv2D, "Conv2d", "Conv", BackendFeature::Convolution, true,
        false}},
      {OpId::MaxPool2D,
       {OpId::MaxPool2D, "MaxPool2d", "MaxPool", BackendFeature::Pooling,
        true, false}},
      {OpId::Upsample2D,
       {OpId::Upsample2D, "Upsample2d", "Upsample2d",
        BackendFeature::Pooling, true, false}},
      {OpId::BatchNorm,
       {OpId::BatchNorm, "BatchNorm", "BatchNormalization",
        BackendFeature::BatchNorm, true, false}},
      {OpId::LayerNorm,
       {OpId::LayerNorm, "LayerNorm", "LayerNorm", std::nullopt, true,
        true}},
      {OpId::MSELoss,
       {OpId::MSELoss, "MSELoss", "MSELoss", BackendFeature::Loss, true,
        false}},
      {OpId::CrossEntropy,
       {OpId::CrossEntropy, "CrossEntropy", "CrossEntropy",
        BackendFeature::Loss, true, false}},
      {OpId::Transpose,
       {OpId::Transpose, "Transpose", "Transpose", std::nullopt, false,
        true}},
      {OpId::Zeros,
       {OpId::Zeros, "Zeros", "Zeros", std::nullopt, false, true}},
  };
  return kRegistry;
}

ForwardNode make_trace_node(const char *op_name,
                            const std::vector<Tensor> &inputs,
                            const std::unordered_map<std::string,
                                                     std::vector<int>> &int_attrs,
                            const std::unordered_map<std::string, float>
                                &float_attrs) {
  ForwardNode node;
  node.op_name = op_name;
  for (const auto &tensor : inputs) {
    if (tensor.name().empty()) {
      tensor.impl_->name =
          "tensor_" + std::to_string(reinterpret_cast<uintptr_t>(tensor.impl_.get()));
    }
    node.input_names.push_back(tensor.name());
    node.inputs.push_back(tensor);
  }
  node.int_attributes = int_attrs;
  node.attributes = float_attrs;
  return node;
}

} // namespace

const OpMetadata &op_metadata(OpId id) {
  const auto &map = registry();
  const auto it = map.find(id);
  if (it == map.end()) {
    throw std::runtime_error("Unknown op metadata request");
  }
  return it->second;
}

DispatchDecision resolve_dispatch(OpId id, const Tensor &tensor) {
  const auto &meta = op_metadata(id);

  if (meta.requires_floating && !is_floating(tensor.dtype())) {
    throw std::runtime_error(std::string(meta.name) +
                             " requires a floating-point tensor, got " +
                             dtype_name(tensor.dtype()));
  }

  if (!meta.feature.has_value()) {
    return {meta, false, meta.allow_cpu_fallback};
  }

  const auto feature = *meta.feature;
  const bool dtype_supported = supports_backend_feature_dtype(feature, tensor.dtype());
  const bool backend_supported =
      dtype_supported && tensor.impl_->backend().supports(feature, tensor.dtype());

  if (backend_supported) {
    return {meta, true, false};
  }

  if (meta.allow_cpu_fallback) {
    return {meta, false, true};
  }

  throw std::runtime_error(std::string(meta.name) + ": backend '" +
                           std::string(tensor.impl_->backend().name()) +
                           "' does not support feature '" +
                           backend_feature_name(feature) +
                           "' for dtype " + dtype_name(tensor.dtype()));
}

void record_registered_trace(
    OpId id, Tensor &out, const std::vector<Tensor> &inputs,
    const std::unordered_map<std::string, std::vector<int>> &int_attrs,
    const std::unordered_map<std::string, float> &float_attrs) {
  const auto &meta = op_metadata(id);
  out.impl_->trace_node = std::make_shared<ForwardNode>(
      make_trace_node(meta.trace_name, inputs, int_attrs, float_attrs));
}

} // namespace ops
} // namespace munet
