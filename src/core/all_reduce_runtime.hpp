#pragma once

#include "../storage.hpp"
#include "../types.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace munet {

namespace detail {

inline bool all_reduce_host_accessible(DeviceType type) {
  return type == DeviceType::CPU || type == DeviceType::UNKNOWN;
}

struct AllReduceTarget {
  Storage *buffer = nullptr;
  Backend *backend = nullptr;
  Device device{};
};

struct AllReduceState {
  int generation = 0;
  int arrived = 0;
  std::vector<std::vector<uint8_t>> payloads;
  std::vector<AllReduceTarget> targets;
};

inline int configured_all_reduce_world_size() {
  const char *env = std::getenv("MUNET_ALLREDUCE_WORLD_SIZE");
  if (!env) {
    return 1;
  }
  int parsed = std::atoi(env);
  return parsed > 0 ? parsed : 1;
}

enum class AllReduceExecutionMode {
  DeviceNative,
  HostFallback,
};

inline AllReduceExecutionMode configured_all_reduce_mode() {
  const char *env = std::getenv("MUNET_ALLREDUCE_MODE");
  if (!env) {
    return AllReduceExecutionMode::DeviceNative;
  }
  const std::string mode(env);
  if (mode == "host_fallback") {
    return AllReduceExecutionMode::HostFallback;
  }
  return AllReduceExecutionMode::DeviceNative;
}

inline std::string all_reduce_key(const Device &device, DataType dtype,
                                  size_t num_elements) {
  (void)device;
  const char *group_env = std::getenv("MUNET_ALLREDUCE_GROUP");
  const std::string group = group_env ? std::string(group_env) : "default";
  return group + "|" + dtype_name(dtype) + "|" +
         std::to_string(num_elements);
}

inline void reduce_sum_bytes(const std::vector<std::vector<uint8_t>> &payloads,
                             DataType dtype, size_t num_elements,
                             std::vector<uint8_t> &out) {
  const size_t elem_size = dtype_size(dtype);
  out.assign(num_elements * elem_size, 0);
  uint8_t *out_bytes = out.data();

  for (size_t elem = 0; elem < num_elements; ++elem) {
    switch (dtype) {
    case DataType::Float32:
    case DataType::Float16:
    case DataType::BFloat16: {
      double sum = 0.0;
      for (const auto &payload : payloads) {
        const ScalarValue scalar =
            read_scalar_from_buffer(payload.data() + elem * elem_size, dtype);
        sum += scalar.value;
      }
      write_scalar_to_buffer(out_bytes + elem * elem_size, dtype, sum);
      break;
    }
    case DataType::Int32: {
      int64_t sum = 0;
      for (const auto &payload : payloads) {
        const ScalarValue scalar =
            read_scalar_from_buffer(payload.data() + elem * elem_size, dtype);
        sum += static_cast<int64_t>(scalar.as_int32());
      }
      write_scalar_to_buffer(out_bytes + elem * elem_size, dtype,
                             static_cast<double>(sum));
      break;
    }
    case DataType::Int8: {
      int64_t sum = 0;
      for (const auto &payload : payloads) {
        const ScalarValue scalar =
            read_scalar_from_buffer(payload.data() + elem * elem_size, dtype);
        sum += static_cast<int64_t>(scalar.as_int32());
      }
      sum = std::max<int64_t>(-128, std::min<int64_t>(127, sum));
      write_scalar_to_buffer(out_bytes + elem * elem_size, dtype,
                             static_cast<double>(sum));
      break;
    }
    default:
      throw std::runtime_error("all_reduce: unsupported dtype");
    }
  }
}

inline void all_reduce_via_host(Storage &buffer, size_t num_elements,
                                Backend &backend, Device device,
                                bool force_host_for_accelerators = false) {
  const int world_size = configured_all_reduce_world_size();
  if (world_size <= 1) {
    return;
  }

  const auto mode = configured_all_reduce_mode();
  const bool accelerator_device = device.type == DeviceType::CUDA ||
                                  device.type == DeviceType::VULKAN;
  if (!force_host_for_accelerators && accelerator_device &&
      mode == AllReduceExecutionMode::DeviceNative) {
    throw std::runtime_error(
        "all_reduce: device-native mode is default for CUDA/Vulkan in "
        "multi-GPU runs; native collective backend not implemented yet. "
        "Set MUNET_ALLREDUCE_MODE=host_fallback to use host staging.");
  }

  const DataType dtype = buffer.dtype();
  const size_t elem_size = dtype_size(dtype);
  if (elem_size == 0) {
    throw std::runtime_error("all_reduce: invalid dtype size");
  }
  const size_t bytes = num_elements * elem_size;
  if (bytes > buffer.size_bytes()) {
    throw std::runtime_error("all_reduce: num_elements exceeds storage size");
  }

  std::vector<uint8_t> payload(bytes);
  Device cpu{DeviceType::CPU, 0};
  if (all_reduce_host_accessible(device.type)) {
    std::memcpy(payload.data(), buffer.data(), bytes);
  } else {
    backend.copy(buffer.data(), payload.data(), bytes, device, cpu);
  }

  static std::mutex mutex;
  static std::condition_variable cv;
  static std::unordered_map<std::string, AllReduceState> states;

  const std::string key = all_reduce_key(device, dtype, num_elements);
  std::vector<uint8_t> reduced;
  std::vector<AllReduceTarget> targets;
  int generation = 0;
  bool last_arrival = false;

  {
    std::unique_lock<std::mutex> lock(mutex);
    auto &state = states[key];
    generation = state.generation;
    state.payloads.push_back(std::move(payload));
    state.targets.push_back(AllReduceTarget{&buffer, &backend, device});
    ++state.arrived;

    if (state.arrived == world_size) {
      reduce_sum_bytes(state.payloads, dtype, num_elements, reduced);
      targets = state.targets;
      state.payloads.clear();
      state.targets.clear();
      state.arrived = 0;
      ++state.generation;
      last_arrival = true;
      cv.notify_all();
    } else {
      cv.wait(lock, [&] { return state.generation != generation; });
      return;
    }
  }

  if (!last_arrival) {
    return;
  }

  for (const auto &target : targets) {
    if (all_reduce_host_accessible(target.device.type)) {
      std::memcpy(target.buffer->data(), reduced.data(), bytes);
    } else {
      target.backend->copy(reduced.data(), target.buffer->data(), bytes, cpu,
                           target.device);
    }
  }
}

} // namespace detail

} // namespace munet
