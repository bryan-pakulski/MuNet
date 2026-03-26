#include "core/op_dispatch.hpp"
#include "tensor.hpp"

#include <array>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace munet;
using namespace munet::ops;

namespace {

std::vector<Device> discover_devices() {
  std::vector<Device> devices;
  devices.push_back(Device{DeviceType::CPU, 0});

#ifdef MUNET_USE_CUDA
  try {
    Tensor probe({1}, Device{DeviceType::CUDA, 0}, DataType::Float32, false);
    (void)probe;
    devices.push_back(Device{DeviceType::CUDA, 0});
  } catch (...) {
  }
#endif

#ifdef MUNET_USE_VULKAN
  try {
    Tensor probe({1}, Device{DeviceType::VULKAN, 0}, DataType::Float32, false);
    (void)probe;
    devices.push_back(Device{DeviceType::VULKAN, 0});
  } catch (...) {
  }
#endif

  return devices;
}

Shape shape_for_op(OpId id) {
  switch (id) {
  case OpId::Conv2D:
  case OpId::MaxPool2D:
  case OpId::Upsample2D:
  case OpId::BatchNorm:
    return {1, 1, 4, 4};
  case OpId::Cat:
    return {2, 2};
  default:
    return {2, 2};
  }
}

std::string bool_text(bool v) { return v ? "yes" : "no"; }

} // namespace

int main() {
  const std::array<DataType, 5> dtypes = {DataType::Float32, DataType::Float16,
                                          DataType::BFloat16, DataType::Int32,
                                          DataType::Int8};
  const std::array<OpId, 29> ops = {
      OpId::Add,         OpId::Sub,       OpId::Mul,      OpId::Div,
      OpId::MaskedFill,  OpId::Matmul,    OpId::Relu,     OpId::Sigmoid,
      OpId::Exp,         OpId::Log,       OpId::Sqrt,     OpId::Rsqrt,
      OpId::Sin,         OpId::Cos,       OpId::Softmax,  OpId::LogSoftmax,
      OpId::Cat,         OpId::Sum,       OpId::Mean,     OpId::Reshape,
      OpId::Conv2D,      OpId::MaxPool2D, OpId::Upsample2D,
      OpId::BatchNorm,   OpId::LayerNorm, OpId::MSELoss,  OpId::CrossEntropy,
      OpId::Transpose,   OpId::Narrow};

  std::vector<Device> devices = discover_devices();

  std::cout << "backend,dtype,op,feature,backend_support,fallback_policy,dispatch_backend,dispatch_cpu_fallback,status,error\n";

  for (const auto &dev : devices) {
    for (DataType dtype : dtypes) {
      for (OpId id : ops) {
        const OpMetadata &meta = op_metadata(id);
        const Shape shape = shape_for_op(id);

        std::string feature_name = meta.feature.has_value()
                                       ? backend_feature_name(*meta.feature)
                                       : "none";
        std::string backend_support = "n/a";
        std::string fallback_policy = "n/a";
        std::string dispatch_backend = "no";
        std::string dispatch_cpu_fallback = "no";
        std::string status = "ok";
        std::string error;

        try {
          Tensor t(shape, dev, dtype, false);
          if (meta.feature.has_value()) {
            BackendSupport support =
                t.impl_->backend().query_support(*meta.feature, dtype, &shape);
            backend_support = bool_text(support.available);
            fallback_policy =
                backend_fallback_policy_name(support.fallback_policy);
          }

          DispatchDecision decision = resolve_dispatch(id, t);
          dispatch_backend = bool_text(decision.use_backend);
          dispatch_cpu_fallback = bool_text(decision.use_cpu_fallback);
        } catch (const std::exception &ex) {
          status = "error";
          error = ex.what();
        }

        std::cout << dev.to_string() << "," << dtype_name(dtype) << ","
                  << meta.name << "," << feature_name << ","
                  << backend_support << "," << fallback_policy << ","
                  << dispatch_backend << "," << dispatch_cpu_fallback << ","
                  << status << ",\"" << error << "\"\n";
      }
    }
  }

  return 0;
}
