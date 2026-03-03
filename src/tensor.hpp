#pragma once

#include "device.hpp"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace munet {
class Tensor {
public:
  Tensor(const std::vector<int> &shape, Device dev = Device::CPU,
         DataType dtype = DataType::FP32);
  ~Tensor();

  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&) noexcept;
  Tensor &operator=(Tensor &&) noexcept;

  void *data() { return data_ptr_; }
  void *data() const { return data_ptr_; }
  size_t size() const;
  inline size_t bytes() const;
  const std::vector<int> &shape() const { return shape_; }

  std::shared_ptr<Tensor> grad_tensor_ = nullptr;

  inline Tensor *grad() { return grad_tensor_.get(); }

  inline void allocate_grad();

  void to_cpu();
  void to_gpu();
  void zero();
  Tensor clone() const;
  void copy_from(const void *host_data, size_t size);

  Device device_;
  DataType dtype_;

private:
  std::vector<int> shape_;
  void *data_ptr_ =
      nullptr; // Void pointer handles mixed precision (float vs half)

  void allocate();
  void deallocate();
};

inline Tensor::Tensor(const std::vector<int> &shape, Device dev, DataType dtype)
    : shape_(shape), device_(dev), dtype_(dtype) {
  allocate();
}

inline size_t Tensor::size() const {
  if (shape_.empty())
    return 0;
  return std::accumulate(shape_.begin(), shape_.end(), 1,
                         std::multiplies<int>());
}

inline size_t Tensor::bytes() const {
  return size() * (dtype_ == DataType::FP32 ? 4 : 2);
}

inline void Tensor::zero() {
  if (!data_ptr_)
    return;
  if (device_ == Device::CPU) {
    std::memset(data_ptr_, 0, bytes());
  }
#ifdef MUNET_USE_CUDA
  else if (device_ == Device::CUDA) {
    cudaMemset(data_ptr_, 0, bytes());
  }
#endif
  else {
    throw std::runtime_error("Unsupported device or missing CUDA support");
  }
}

inline Tensor Tensor::clone() const {
  Tensor copy(shape_, device_, dtype_);
  if (device_ == Device::CPU) {
    std::memcpy(copy.data(), data(), bytes());
  }
#ifdef MUNET_USE_CUDA
  else if (device_ == Device::CUDA) {
    cudaMemcpy(copy.data(), data(), bytes(), cudaMemcpyDeviceToDevice);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device or missing CUDA support");
  }
  return copy;
}

inline void Tensor::allocate() {
  if (size() == 0)
    return;

  if (device_ == Device::CPU) {
    data_ptr_ = std::malloc(bytes());
    if (!data_ptr_)
      throw std::bad_alloc();
  }
#ifdef MUNET_USE_CUDA
  else if (device_ == Device::CUDA) {
    if (cudaMalloc(&data_ptr_, bytes()) != cudaSuccess) {
      throw std::runtime_error("CUDA allocation failed");
    }
  }
#endif
  else {
    throw std::runtime_error("Unsupported device or missing CUDA support");
  }
}

inline void Tensor::deallocate() {
  if (!data_ptr_)
    return;

  if (device_ == Device::CPU) {
    std::free(data_ptr_);
  }
#ifdef MUNET_USE_CUDA
  else if (device_ == Device::CUDA) {
    cudaFree(data_ptr_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device or missing CUDA support");
  }
  data_ptr_ = nullptr;
}

inline Tensor::~Tensor() { deallocate(); }

inline Tensor::Tensor(Tensor &&other) noexcept
    : shape_(std::move(other.shape_)), device_(other.device_),
      dtype_(other.dtype_), data_ptr_(other.data_ptr_),
      grad_tensor_(std::move(other.grad_tensor_)) { // FIX: Move gradients
  other.data_ptr_ = nullptr;                        // Take ownership
}

inline Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    deallocate(); // Free existing memory
    shape_ = std::move(other.shape_);
    device_ = other.device_;
    dtype_ = other.dtype_;
    data_ptr_ = other.data_ptr_;
    grad_tensor_ = std::move(other.grad_tensor_); // FIX: Move gradients
    other.data_ptr_ = nullptr;
  }
  return *this;
}

inline void Tensor::to_gpu() {
#ifdef MUNET_USE_CUDA
  if (device_ == Device::CUDA)
    return;
  void *new_ptr;
  if (cudaMalloc(&new_ptr, bytes()) != cudaSuccess)
    throw std::runtime_error("CUDA alloc failed");
  if (cudaMemcpy(new_ptr, data_ptr_, bytes(), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    cudaFree(new_ptr);
    throw std::runtime_error("CUDA memcpy failed");
  }
  deallocate();
  data_ptr_ = new_ptr;
  device_ = Device::CUDA;
#else
  throw std::runtime_error("µNet compiled without CUDA support.");
#endif
}

inline void Tensor::to_cpu() {
  if (device_ == Device::CPU)
    return;
#ifdef MUNET_USE_CUDA
  void *new_ptr = std::malloc(bytes());
  if (!new_ptr)
    throw std::bad_alloc();
  if (cudaMemcpy(new_ptr, data_ptr_, bytes(), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    std::free(new_ptr);
    throw std::runtime_error("CUDA memcpy failed");
  }
  deallocate();
  data_ptr_ = new_ptr;
  device_ = Device::CPU;
#endif
}

inline void Tensor::allocate_grad() {
  if (!grad_tensor_) {
    grad_tensor_ = std::make_shared<Tensor>(shape_, device_, dtype_);
  }
  grad_tensor_->zero();
}

inline void Tensor::copy_from(const void *host_data, size_t size) {
  if (size != bytes())
    throw std::runtime_error("Size mismatch in copy_from");

  if (device_ == Device::CPU) {
    std::memcpy(data_ptr_, host_data, size);
  }
#ifdef MUNET_USE_CUDA
  else if (device_ == Device::CUDA) {
    if (cudaMemcpy(data_ptr_, host_data, size, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      throw std::runtime_error("CUDA memcpy failed");
    }
  }
#endif
  else {
    throw std::runtime_error("Unsupported device");
  }
}

} // namespace munet
