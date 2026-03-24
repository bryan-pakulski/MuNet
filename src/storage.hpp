#pragma once
#include "core/backend.hpp"
#include "types.hpp"
#include <memory>

namespace munet {

class Storage {
public:
  Storage(size_t size_bytes, Device device, DataType dtype, Shape shape = {})
      : size_bytes_(size_bytes), device_(device), dtype_(dtype), shape_(shape) {

    backend_ = BackendManager::get(device);
    data_ptr_ = backend_->allocate(size_bytes_);
  }

  ~Storage() {
    if (data_ptr_) {
      backend_->deallocate(data_ptr_);
    }
  }

  // Storage is strictly unique to a TensorImpl. No copying allowed.
  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;

  void *data() const { return data_ptr_; }
  size_t size_bytes() const { return size_bytes_; }
  Device device() const { return device_; }
  DataType dtype() const { return dtype_; }
  const Shape &shape() const { return shape_; }
    
  void zero_() { backend_->memset(data_ptr_, 0, size_bytes_); }

  Backend &backend() const { return *backend_; }

private:
  void *data_ptr_ = nullptr;
  size_t size_bytes_;
  Device device_;
  DataType dtype_;
  Shape shape_;
  std::shared_ptr<Backend> backend_;
};

} // namespace munet
