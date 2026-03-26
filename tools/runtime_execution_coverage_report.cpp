#include "core/op_dispatch.hpp"
#include "nn.hpp"
#include "tensor.hpp"

#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace munet;
using namespace munet::ops;

namespace {

struct OperationCase {
  OpId id;
  std::string input_shape_case;
  std::function<void(const Device &, DataType)> run;
};

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

Shape dispatch_shape_for_op(OpId id) {
  switch (id) {
  case OpId::Conv2D:
  case OpId::MaxPool2D:
  case OpId::Upsample2D:
  case OpId::BatchNorm:
    return {1, 1, 4, 4};
  default:
    return {2, 2};
  }
}

std::vector<OperationCase> build_cases() {
  return {
      {OpId::Add, "2x2", [](const Device &dev, DataType dtype) {
         Tensor a({2, 2}, dev, dtype, false);
         Tensor b({2, 2}, dev, dtype, false);
         (void)(a + b);
       }},
      {OpId::Matmul, "2x2", [](const Device &dev, DataType dtype) {
         Tensor a({2, 2}, dev, dtype, false);
         Tensor b({2, 2}, dev, dtype, false);
         (void)a.matmul(b);
       }},
      {OpId::Relu, "2x2", [](const Device &dev, DataType dtype) {
         Tensor x({2, 2}, dev, dtype, false);
         (void)x.relu();
       }},
      {OpId::Softmax, "2x2", [](const Device &dev, DataType dtype) {
         Tensor x({2, 2}, dev, dtype, false);
         (void)x.softmax(-1);
       }},
      {OpId::Sum, "2x2", [](const Device &dev, DataType dtype) {
         Tensor x({2, 2}, dev, dtype, false);
         (void)x.sum();
       }},
      {OpId::Conv2D, "1x1x4x4", [](const Device &dev, DataType dtype) {
         Tensor in({1, 1, 4, 4}, dev, dtype, false);
         Tensor w({1, 1, 3, 3}, dev, dtype, false);
         (void)in.conv2d(w, Tensor(), 1, 1);
       }},
      {OpId::MaxPool2D, "1x1x4x4", [](const Device &dev, DataType dtype) {
         Tensor in({1, 1, 4, 4}, dev, dtype, false);
         (void)in.max_pool2d(2, 2, 0);
       }},
      {OpId::Upsample2D, "1x1x4x4", [](const Device &dev, DataType dtype) {
         Tensor in({1, 1, 4, 4}, dev, dtype, false);
         (void)in.upsample2d(2);
       }},
      {OpId::BatchNorm, "1x1x4x4", [](const Device &dev, DataType dtype) {
         TensorOptions options;
         options.device = dev;
         options.dtype = dtype;
         nn::BatchNorm2d bn(1, 1e-5f, 0.1f, options);
         bn.eval();
         Tensor in({1, 1, 4, 4}, dev, dtype, false);
         (void)bn.forward(in);
       }},
      {OpId::MSELoss, "2x2", [](const Device &dev, DataType dtype) {
         Tensor a({2, 2}, dev, dtype, false);
         Tensor b({2, 2}, dev, dtype, false);
         (void)a.mse_loss(b);
       }},
      {OpId::CrossEntropy, "2x3", [](const Device &dev, DataType dtype) {
         Tensor logits({2, 3}, dev, dtype, false);
         Tensor targets({2, 3}, dev, dtype, false);
         if (is_floating(dtype)) {
           targets.fill_(make_scalar(0.0, dtype));
         }
         (void)logits.cross_entropy(targets);
       }},
  };
}

} // namespace

int main() {
  const std::array<DataType, 5> dtypes = {DataType::Float32, DataType::Float16,
                                          DataType::BFloat16, DataType::Int32,
                                          DataType::Int8};

  std::vector<Device> devices = discover_devices();
  std::vector<OperationCase> cases = build_cases();

  std::cout << "backend,dtype,op,input_shape_case,dispatch_path,runtime_status,error\n";

  for (const auto &dev : devices) {
    for (DataType dtype : dtypes) {
      for (const auto &op_case : cases) {
        std::string dispatch_path = "unresolved";
        std::string runtime_status = "pass";
        std::string error;

        try {
          Tensor probe(dispatch_shape_for_op(op_case.id), dev, dtype, false);
          const auto decision = resolve_dispatch(op_case.id, probe);
          dispatch_path = decision.use_backend
                              ? "backend"
                              : (decision.use_cpu_fallback ? "fallback"
                                                           : "none");
        } catch (const std::exception &ex) {
          dispatch_path = "error";
          runtime_status = "fail";
          error = std::string("dispatch: ") + ex.what();
        }

        if (runtime_status == "pass") {
          try {
            op_case.run(dev, dtype);
          } catch (const std::exception &ex) {
            runtime_status = "fail";
            error = ex.what();
          }
        }

        std::cout << dev.to_string() << "," << dtype_name(dtype) << ","
                  << op_metadata(op_case.id).name << ","
                  << op_case.input_shape_case << "," << dispatch_path << ","
                  << runtime_status << ",\"" << error << "\"\n";
      }
    }
  }

  return 0;
}
