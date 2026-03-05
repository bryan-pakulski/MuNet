#pragma once
#include "tensor.hpp"
#include <vector>

namespace munet {
namespace test {

inline std::vector<Device> get_available_devices() {
  std::vector<Device> devices = {{DeviceType::CPU, 0}};
#ifdef MUNET_USE_CUDA
  try {
    // Basic probe to check if a device exists
    Device d{DeviceType::CUDA, 0};
    Tensor t({1}, d);
    devices.push_back(d);
  } catch (...) {
  }
#endif
#ifdef MUNET_USE_VULKAN
  try {
    Device d{DeviceType::VULKAN, 0};
    Tensor t({1}, d);
    devices.push_back(d);
  } catch (...) {
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
