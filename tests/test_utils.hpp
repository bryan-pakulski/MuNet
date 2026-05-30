#pragma once
#include "tensor.hpp"
#include <cmath>
#include <vector>

namespace munet {
namespace test {

inline bool accelerator_health_check(const Device &device) {
  try {
    Tensor a({1}, device, DataType::Float32);
    Tensor b({1}, device, DataType::Float32);
    a.fill_(2.0f);
    b.fill_(3.0f);
    Tensor out = a + b; // real backend op
    out.impl_->backend().synchronize();
    Tensor out_host = out.to({DeviceType::VULKAN, 0}); // copy back
    const float v = static_cast<const float *>(out_host.data())[0];
    return std::abs(v - 5.0f) <= 1e-4f;
  } catch (...) {
    return false;
  }
}

inline std::vector<Device> get_available_devices() {
  return {{DeviceType::VULKAN, 0}};
}

inline bool all_close(const Tensor &a, const Tensor &b, float atol = 1e-4f) {
  if (a.shape() != b.shape())
    return false;
  Tensor a_host = a.to({DeviceType::VULKAN, 0});
  Tensor b_host = b.to({DeviceType::VULKAN, 0});
  const float *ptr_a = static_cast<const float *>(a_host.data());
  const float *ptr_b = static_cast<const float *>(b_host.data());
  for (size_t i = 0; i < a_host.size(); ++i) {
    if (std::abs(ptr_a[i] - ptr_b[i]) > atol)
      return false;
  }
  return true;
}

} // namespace test
} // namespace munet
