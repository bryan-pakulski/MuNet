#pragma once
#include "core.hpp"

namespace munet {

/// A contiguous, row-major FP32 host tensor. Scalars have shape {} and one value.
/// Storage belongs to the tensor and remains valid after later Model::run calls.
struct Tensor {
  Shape shape;
  std::vector<float> data;
};

/// Public model signature; names can be set with Python munet.export().
struct TensorInfo {
  std::string name;
  Shape shape;
};

struct ModelOptions {
  /// "cpu", "vulkan", or "vulkan:N". Vulkan never silently falls back to CPU.
  std::string device = "cpu";
  /// Bound both archive bytes and the planned device/CPU arena (separately).
  /// This is not a total process-memory limit; graph/tensor/driver storage is additional.
  uint64_t max_memory_bytes = uint64_t{2} * 1024 * 1024 * 1024;
};

/// Load an inference .mnet exported by Python, without a Python dependency.
/// CPU accepts ordinary saved inference graphs. Vulkan requires include_vulkan=True
/// at export time; the deployment host needs a driver, but no shader compiler.
/// Training programs/checkpoints are rejected. Calls on one instance serialize;
/// use separate instances for independent simultaneous inference.
class Model {
 public:
  explicit Model(const std::string& path, const ModelOptions& options = {});
  ~Model();
  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const std::vector<TensorInfo>& inputs() const;
  const std::vector<TensorInfo>& outputs() const;
  /// Positional inputs/outputs follow the signature order. Outputs are host copies.
  std::vector<Tensor> run(const std::vector<Tensor>& inputs);
  /// Requires exactly the exported input names; returns tensors by output name.
  std::map<std::string, Tensor> run_named(const std::map<std::string, Tensor>& inputs);
  std::map<std::string, uint64_t> stats() const;
  std::string device_name() const;
  void synchronize();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Impl& impl() const;
};
}  // namespace munet
