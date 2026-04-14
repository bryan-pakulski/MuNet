#pragma once
#include "tensor.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace munet {
namespace test {

inline bool accelerator_health_check(const Device &device,
                                     std::string *error_detail = nullptr) {
  try {
    Tensor a({1}, device, DataType::Float32);
    Tensor b({1}, device, DataType::Float32);
    a.fill_(2.0f);
    b.fill_(3.0f);
    Tensor out = a + b; // real backend op
    out.impl_->backend().synchronize();
    Tensor out_cpu = out.to({DeviceType::CPU, 0}); // copy back
    const float v = static_cast<const float *>(out_cpu.data())[0];
    return std::abs(v - 5.0f) <= 1e-4f;
  } catch (const std::exception &e) {
    if (error_detail) {
      *error_detail = e.what();
    }
    return false;
  } catch (...) {
    if (error_detail) {
      *error_detail = "unknown error";
    }
    return false;
  }
}

inline std::vector<Device> get_available_devices() {
  std::vector<Device> devices = {{DeviceType::CPU, 0}};
#ifdef MUNET_USE_CUDA
  {
    Device d{DeviceType::CUDA, 0};
    if (accelerator_health_check(d)) {
      devices.push_back(d);
    }
  }
#endif
#ifdef MUNET_USE_VULKAN
  {
    Device d{DeviceType::VULKAN, 0};
    if (accelerator_health_check(d)) {
      devices.push_back(d);
    }
  }
#endif
  return devices;
}

inline bool all_close(const Tensor &a, const Tensor &b, float atol = 1e-4f) {
  if (a.shape() != b.shape())
    return false;
  Tensor a_cpu = a.to({DeviceType::CPU, 0});
  Tensor b_cpu = b.to({DeviceType::CPU, 0});
  const float *ptr_a = static_cast<const float *>(a_cpu.data());
  const float *ptr_b = static_cast<const float *>(b_cpu.data());
  for (size_t i = 0; i < a_cpu.size(); ++i) {
    if (std::abs(ptr_a[i] - ptr_b[i]) > atol)
      return false;
  }
  return true;
}

} // namespace test
} // namespace munet
