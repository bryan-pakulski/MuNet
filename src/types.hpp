#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace munet {

enum class DeviceType { CPU, CUDA, VULKAN, UNKNOWN };
enum class DataType { Float32, Float16, Int32 };

struct Device {
  DeviceType type;
  int index = 0;

  bool operator==(const Device &other) const {
    return type == other.type && index == other.index;
  }
  bool operator!=(const Device &other) const { return !(*this == other); }

  std::string to_string() const {
    std::string t = (type == DeviceType::CPU)      ? "cpu"
                    : (type == DeviceType::CUDA)   ? "cuda"
                    : (type == DeviceType::VULKAN) ? "vulkan"
                                                   : "unknown";
    return t + ":" + std::to_string(index);
  }
};

using Shape = std::vector<int>;

inline std::string to_string(const Shape &shape) {
  std::string shape_str = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_str += std::to_string(shape[i]);
    if (i < shape.size() - 1)
      shape_str += ", ";
  }
  shape_str += "]";
  return shape_str;
}

inline size_t numel(const Shape &shape) {
  if (shape.empty())
    return 0;
  size_t n = 1;
  for (int s : shape)
    n *= s;
  return n;
}

inline size_t dtype_size(DataType dt) {
  switch (dt) {
  case DataType::Float32:
    return 4;
  case DataType::Int32:
    return 4;
  case DataType::Float16:
    return 2;
  default:
    return 0;
  }
}

} // namespace munet
