#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace munet {

enum class DeviceType { CPU, CUDA, VULKAN, UNKNOWN };
enum class DataType { Float32, Float16, Int32 };
enum class AccumulationOp { Elementwise, Reduction, Matmul, Convolution, Normalization };

using Shape = std::vector<int>;
using Strides = std::vector<int>;

inline std::string dtype_name(DataType dt) {
  switch (dt) {
  case DataType::Float32:
    return "float32";
  case DataType::Float16:
    return "float16";
  case DataType::Int32:
    return "int32";
  default:
    return "unknown";
  }
}

inline bool is_floating(DataType dt) {
  return dt == DataType::Float32 || dt == DataType::Float16;
}

inline bool is_integral(DataType dt) { return dt == DataType::Int32; }

inline bool is_low_precision(DataType dt) { return dt == DataType::Float16; }

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

inline DataType promote_types(DataType a, DataType b) {
  if (a == b)
    return a;
  if (a == DataType::Float32 || b == DataType::Float32)
    return DataType::Float32;
  if (a == DataType::Float16 || b == DataType::Float16)
    return DataType::Float16;
  if (a == DataType::Int32 && b == DataType::Int32)
    return DataType::Int32;
  return DataType::Float32;
}

inline DataType accumulation_type(AccumulationOp op, DataType dtype) {
  switch (op) {
  case AccumulationOp::Elementwise:
  case AccumulationOp::Reduction:
  case AccumulationOp::Matmul:
  case AccumulationOp::Convolution:
  case AccumulationOp::Normalization:
    return (dtype == DataType::Float16) ? DataType::Float32 : dtype;
  default:
    return dtype;
  }
}

inline DataType optimizer_state_type(DataType parameter_dtype) {
  return accumulation_type(AccumulationOp::Elementwise, parameter_dtype);
}

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

struct TensorOptions {
  Device device{DeviceType::CPU, 0};
  DataType dtype{DataType::Float32};
  bool requires_grad = false;

  TensorOptions() = default;
  explicit TensorOptions(Device new_device,
                         DataType new_dtype = DataType::Float32,
                         bool new_requires_grad = false)
      : device(new_device), dtype(new_dtype),
        requires_grad(new_requires_grad) {}

  TensorOptions &with_device(Device new_device) {
    device = new_device;
    return *this;
  }

  TensorOptions &with_dtype(DataType new_dtype) {
    dtype = new_dtype;
    return *this;
  }

  TensorOptions &with_requires_grad(bool value) {
    requires_grad = value;
    return *this;
  }
};

struct DTypeInfo {
  DataType dtype;
  const char *name;
  size_t size_bytes;
  bool floating;
  bool integral;
  bool low_precision;
};

struct ScalarValue {
  DataType dtype = DataType::Float32;
  double value = 0.0;

  float as_float() const { return static_cast<float>(value); }
  int32_t as_int32() const { return static_cast<int32_t>(value); }
  bool is_nonzero() const { return value != 0.0; }
};

inline DTypeInfo dtype_info(DataType dt) {
  switch (dt) {
  case DataType::Float32:
    return {dt, "float32", 4, true, false, false};
  case DataType::Float16:
    return {dt, "float16", 2, true, false, true};
  case DataType::Int32:
    return {dt, "int32", 4, false, true, false};
  default:
    throw std::runtime_error("Unsupported dtype info query");
  }
}

inline ScalarValue make_scalar(float value) {
  return ScalarValue{DataType::Float32, static_cast<double>(value)};
}

inline ScalarValue make_scalar(double value, DataType dtype) {
  return ScalarValue{dtype, value};
}

inline ScalarValue make_scalar(int32_t value) {
  return ScalarValue{DataType::Int32, static_cast<double>(value)};
}

inline uint16_t float_to_half_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const uint32_t sign = (bits >> 16) & 0x8000u;
  uint32_t mantissa = bits & 0x007fffffu;
  int exp = ((bits >> 23) & 0xff) - 127 + 15;

  if (((bits >> 23) & 0xff) == 0xff) {
    if (mantissa != 0) {
      return static_cast<uint16_t>(sign | 0x7e00u);
    }
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign);

    mantissa |= 0x00800000u;
    const uint32_t shifted = mantissa >> static_cast<uint32_t>(1 - exp);
    const uint32_t rounded = (shifted + 0x00001000u) >> 13;
    return static_cast<uint16_t>(sign | rounded);
  }

  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  const uint32_t rounded_mantissa = (mantissa + 0x00001000u) >> 13;
  if (rounded_mantissa == 0x0400u) {
    ++exp;
    if (exp >= 31)
      return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10));
  }

  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               (rounded_mantissa & 0x03ffu));
}

inline float half_bits_to_float(uint16_t value) {
  const uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16;
  const uint32_t exp = (value >> 10) & 0x1fu;
  const uint32_t mantissa = value & 0x03ffu;

  uint32_t bits = 0;
  if (exp == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      uint32_t mant = mantissa;
      int shift = -1;
      do {
        ++shift;
        mant <<= 1;
      } while ((mant & 0x0400u) == 0);
      mant &= 0x03ffu;
      bits = sign | (static_cast<uint32_t>(127 - 15 - shift) << 23) |
             (mant << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline void write_scalar_to_buffer(void *dst, DataType dtype, double value) {
  switch (dtype) {
  case DataType::Float32: {
    float converted = static_cast<float>(value);
    std::memcpy(dst, &converted, sizeof(converted));
    return;
  }
  case DataType::Float16: {
    uint16_t converted = float_to_half_bits(static_cast<float>(value));
    std::memcpy(dst, &converted, sizeof(converted));
    return;
  }
  case DataType::Int32: {
    int32_t converted = static_cast<int32_t>(value);
    std::memcpy(dst, &converted, sizeof(converted));
    return;
  }
  default:
    throw std::runtime_error("Unsupported scalar dtype write");
  }
}

inline ScalarValue read_scalar_from_buffer(const void *src, DataType dtype) {
  ScalarValue out;
  out.dtype = dtype;
  switch (dtype) {
  case DataType::Float32: {
    float value = 0.0f;
    std::memcpy(&value, src, sizeof(value));
    out.value = value;
    return out;
  }
  case DataType::Float16: {
    uint16_t value = 0;
    std::memcpy(&value, src, sizeof(value));
    out.value = half_bits_to_float(value);
    return out;
  }
  case DataType::Int32: {
    int32_t value = 0;
    std::memcpy(&value, src, sizeof(value));
    out.value = static_cast<double>(value);
    return out;
  }
  default:
    throw std::runtime_error("Unsupported scalar dtype read");
  }
}

inline void convert_buffer_dtype(const void *src, DataType src_dtype, void *dst,
                                 DataType dst_dtype, size_t count) {
  const char *src_bytes = static_cast<const char *>(src);
  char *dst_bytes = static_cast<char *>(dst);
  const size_t src_size = dtype_size(src_dtype);
  const size_t dst_size = dtype_size(dst_dtype);

  for (size_t i = 0; i < count; ++i) {
    ScalarValue scalar =
        read_scalar_from_buffer(src_bytes + i * src_size, src_dtype);
    write_scalar_to_buffer(dst_bytes + i * dst_size, dst_dtype, scalar.value);
  }
}

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
    int idx_a = (int)a_shape.size() - 1 - i;
    int idx_b = (int)b_shape.size() - 1 - i;
    int out_idx = ndim - 1 - i;

    int dim_a = (idx_a >= 0) ? a_shape[idx_a] : 1;
    int dim_b = (idx_b >= 0) ? b_shape[idx_b] : 1;

    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      info.can_broadcast = false;
      return info;
    }

    info.out_shape[out_idx] = std::max(dim_a, dim_b);

    if (idx_a >= 0) {
      info.strides_a[out_idx] =
          (dim_a == 1 && dim_b != 1) ? 0 : a_strides[idx_a];
    } else {
      info.strides_a[out_idx] = 0;
    }

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
  Strides out_strides = default_strides(info.out_shape);
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

} // namespace munet
