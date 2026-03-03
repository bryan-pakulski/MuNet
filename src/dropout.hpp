#pragma once

#include "layer.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace munet {

class Dropout : public Layer {
public:
  explicit Dropout(float p = 0.5f) : p_(p) {}

  void train() override { training_ = true; }
  void eval() override { training_ = false; }

  inline Tensor forward(const Tensor &input) override {
    if (!training_)
      return input.clone();

    mask_ = Tensor(input.shape(), input.device_, input.dtype_);
    Tensor output(input.shape(), input.device_, input.dtype_);
    float scale = 1.0f / (1.0f - p_);

    std::vector<float> host_mask(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      float r = static_cast<float>(std::rand()) / RAND_MAX;
      host_mask[i] = (r > p_) ? scale : 0.0f;
    }

    if (input.device_ == Device::CPU) {
      const float *in_ptr = static_cast<const float *>(input.data());
      float *out_ptr = static_cast<float *>(output.data());
      float *m_ptr = static_cast<float *>(mask_.data());

      std::memcpy(m_ptr, host_mask.data(), input.bytes());

      for (size_t i = 0; i < input.size(); ++i) {
        out_ptr[i] = in_ptr[i] * m_ptr[i];
      }
    }
#ifdef MUNET_USE_CUDA
    else if (input.device_ == Device::CUDA) {
      cudaMemcpy(mask_.data(), host_mask.data(), input.bytes(),
                 cudaMemcpyHostToDevice);
      cuda_kernels::elementwise_mul(static_cast<const float *>(input.data()),
                                    static_cast<const float *>(mask_.data()),
                                    static_cast<float *>(output.data()),
                                    input.size());
    }
#endif
    return output;
  }

  inline Tensor backward(const Tensor &grad_output) override {
    if (!training_)
      return grad_output.clone();

    Tensor grad_input(grad_output.shape(), grad_output.device_,
                      grad_output.dtype_);

    if (grad_output.device_ == Device::CPU) {
      const float *go_ptr = static_cast<const float *>(grad_output.data());
      const float *m_ptr = static_cast<const float *>(mask_.data());
      float *gi_ptr = static_cast<float *>(grad_input.data());
      for (size_t i = 0; i < grad_output.size(); ++i) {
        gi_ptr[i] = go_ptr[i] * m_ptr[i];
      }
    }
#ifdef MUNET_USE_CUDA
    else if (grad_output.device_ == Device::CUDA) {
      cuda_kernels::elementwise_mul(
          static_cast<const float *>(grad_output.data()),
          static_cast<const float *>(mask_.data()),
          static_cast<float *>(grad_input.data()), grad_output.size());
    }
#endif
    return grad_input;
  }

  inline std::string get_onnx_op_type() const override { return "Dropout"; }

private:
  float p_;
  bool training_ = true;
  Tensor mask_{std::vector<int>{}};
};
} // namespace munet
