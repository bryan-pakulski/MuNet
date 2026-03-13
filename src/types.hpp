#pragma once

#include <cstddef>
#include <string>
#include <vector>
namespace munet {

enum class DeviceType { CPU, CUDA, VULKAN, UNKNOWN };
enum class DataType {
  Float32,
  Float16,
  BFloat16,
  Float8E4M3FN,
  Float8E5M2,
  Int8,
  Int4,
  Int32,
  Float64
};

inline bool is_float_dtype(DataType dt) {
  switch (dt) {
  case DataType::Float8E4M3FN:
  case DataType::Float8E5M2:
  case DataType::Float16:
  case DataType::BFloat16:
  case DataType::Float32:
  case DataType::Float64:
    return true;
  case DataType::Int8:
  case DataType::Int4:
  case DataType::Int32:
    return false;
  default:
    return false;
  }
}

inline bool is_fp8(DataType dt) {
  return dt == DataType::Float8E4M3FN || dt == DataType::Float8E5M2;
}

inline bool is_low_precision(DataType dt) {
  return is_fp8(dt) || dt == DataType::Float16 || dt == DataType::BFloat16;
}

inline DataType accumulation_dtype(DataType dt) {
  if (is_low_precision(dt))
    return DataType::Float32;
  return dt;
}

inline const char *dtype_name(DataType dt) {
  switch (dt) {
  case DataType::Float32:
    return "float32";
  case DataType::Float16:
    return "float16";
  case DataType::BFloat16:
    return "bfloat16";
  case DataType::Float8E4M3FN:
    return "float8_e4m3fn";
  case DataType::Float8E5M2:
    return "float8_e5m2";
  case DataType::Int8:
    return "int8";
  case DataType::Int4:
    return "int4";
  case DataType::Int32:
    return "int32";
  case DataType::Float64:
    return "float64";
  default:
    return "unknown";
  }
}

using Shape = std::vector<int>;
using Strides = std::vector<int>;

inline Strides default_strides(const Shape &shape) {
  Strides strides(shape.size());
  int s = 1;
  for (int i = (int)shape.size() - 1; i >= 0; --i) {
    strides[i] = s;
    s *= shape[i];
  }
  return strides;
}

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

inline size_t numel(const Shape &shape) {
  if (shape.empty())
    return 0;
  size_t n = 1;
  for (int s : shape)
    n *= s;
  return n;
}

struct BroadcastInfo {
  Shape out_shape;
  Strides strides_a;
  Strides strides_b;
  bool can_broadcast = false;
};

struct GPUBroadcastInfo {
  int ndim;
  int shape[6];
  int out_strides[6];
  int strides_a[6];
  int strides_b[6];
	int total;
};

inline BroadcastInfo compute_broadcast(const Shape &a_shape,
                                       const Strides &a_strides,
                                       const Shape &b_shape,
                                       const Strides &b_strides) {
  int ndim = std::max((int)a_shape.size(), (int)b_shape.size());
  BroadcastInfo info;
  info.out_shape.resize(ndim);
  info.strides_a.resize(ndim);
  info.strides_b.resize(ndim);

  for (int i = 0; i < ndim; ++i) {
    // Alignment: work from right to left
    int idx_a = (int)a_shape.size() - 1 - i;
    int idx_b = (int)b_shape.size() - 1 - i;
    int out_idx = ndim - 1 - i;

    // Correctly access the dimension value at the index
    int dim_a = (idx_a >= 0) ? a_shape[idx_a] : 1;
    int dim_b = (idx_b >= 0) ? b_shape[idx_b] : 1;

    // Compatibility check
    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      info.can_broadcast = false;
      return info;
    }

    // Assign to the specific index of the output vector
    info.out_shape[out_idx] = std::max(dim_a, dim_b);

    // Calculate Virtual Strides for A
    if (idx_a >= 0) {
      info.strides_a[out_idx] =
          (dim_a == 1 && dim_b != 1) ? 0 : a_strides[idx_a];
    } else {
      info.strides_a[out_idx] = 0;
    }

    // Calculate Virtual Strides for B
    if (idx_b >= 0) {
      info.strides_b[out_idx] =
          (dim_b == 1 && dim_a != 1) ? 0 : b_strides[idx_b];
    } else {
      info.strides_b[out_idx] = 0;
    }
  }

  info.can_broadcast = true;
  return info;
}

inline GPUBroadcastInfo to_gpu_info(const BroadcastInfo &info) {
  GPUBroadcastInfo gpu;
  gpu.ndim = (int)info.out_shape.size();
  gpu.total = (int)numel(info.out_shape);
  Strides out_strides = default_strides(info.out_shape); // Calculate strides
  for (int i = 0; i < 6; ++i) {
    if (i < gpu.ndim) {
      gpu.shape[i] = info.out_shape[i];
      gpu.out_strides[i] = out_strides[i];
      gpu.strides_a[i] = info.strides_a[i];
      gpu.strides_b[i] = info.strides_b[i];
    } else {
      gpu.shape[i] = 1;
      gpu.out_strides[i] = 0;
      gpu.strides_a[i] = 0;
      gpu.strides_b[i] = 0;
    }
  }
  return gpu;
}

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

inline size_t dtype_size(DataType dt) {
  switch (dt) {
  case DataType::Float32:
    return 4;
  case DataType::Float64:
    return 8;
  case DataType::Int8:
    return 1;
  case DataType::Int4:
    return 1;
  case DataType::Int32:
    return 4;
  case DataType::Float16:
  case DataType::BFloat16:
    return 2;
  case DataType::Float8E4M3FN:
  case DataType::Float8E5M2:
    return 1;
  default:
    return 0;
  }
}

} // namespace munet
