#pragma once

#include "kernels.hpp"
#include "layer.hpp"
#include "tensor.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace munet {

class ReLU : public Layer {
public:
  inline Tensor forward(const Tensor &input) override;
  inline Tensor backward(const Tensor &grad_output) override;
  inline std::string get_onnx_op_type() const override { return "Relu"; }

private:
  Tensor input_cache_{std::vector<int>{}};
};

inline Tensor ReLU::forward(const Tensor &input) {
  input_cache_ = input.clone();
  Tensor output(input.shape(), input.device_, input.dtype_);

  const float *in_ptr = static_cast<const float *>(input.data());
  float *out_ptr = static_cast<float *>(output.data());

#ifdef MUNET_USE_CUDA
  if (input.device_ == Device::CUDA) {
    cuda_kernels::relu_forward(in_ptr, out_ptr, input.size());
    return output;
  }
#endif

  for (size_t i = 0; i < input.size(); ++i) {
    out_ptr[i] = in_ptr[i] > 0.0f ? in_ptr[i] : 0.0f;
  }
  return output;
}

inline Tensor ReLU::backward(const Tensor &grad_output) {
  Tensor grad_input(grad_output.shape(), grad_output.device_,
                    grad_output.dtype_);

  const float *go_ptr = static_cast<const float *>(grad_output.data());
  const float *in_ptr = static_cast<const float *>(input_cache_.data());
  float *gi_ptr = static_cast<float *>(grad_input.data());

#ifdef MUNET_USE_CUDA
  if (grad_output.device_ == Device::CUDA) {
    cuda_kernels::relu_backward(go_ptr, in_ptr, gi_ptr, grad_output.size());
    return grad_input;
  }
#endif

  for (size_t i = 0; i < grad_output.size(); ++i) {
    gi_ptr[i] = in_ptr[i] > 0.0f ? go_ptr[i] : 0.0f;
  }
  return grad_input;
}

class Softmax : public Layer {
public:
  inline Tensor forward(const Tensor &input) override;
  inline Tensor backward(const Tensor &grad_output) override;
  inline std::string get_onnx_op_type() const override { return "Softmax"; }

private:
  Tensor output_cache_{std::vector<int>{}};
};

inline Tensor Softmax::forward(const Tensor &input) {
  Tensor output(input.shape(), input.device_, input.dtype_);
  int batch_size = input.shape().size() > 1 ? input.shape()[0] : 1;
  int num_classes =
      input.shape().size() > 1 ? input.shape()[1] : input.shape()[0];

  const float *in_ptr = static_cast<const float *>(input.data());
  float *out_ptr = static_cast<float *>(output.data());

#ifdef MUNET_USE_CUDA
  if (input.device_ == Device::CUDA) {
    cuda_kernels::softmax_forward(in_ptr, out_ptr, batch_size, num_classes);
    output_cache_ = output.clone();
    return output;
  }
#endif

  for (int b = 0; b < batch_size; ++b) {
    const float *in_row = in_ptr + b * num_classes;
    float *out_row = out_ptr + b * num_classes;

    float max_val = in_row[0];
    for (int i = 1; i < num_classes; ++i) {
      if (in_row[i] > max_val)
        max_val = in_row[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
      out_row[i] = std::exp(in_row[i] - max_val);
      sum_exp += out_row[i];
    }

    for (int i = 0; i < num_classes; ++i) {
      out_row[i] /= sum_exp;
    }
  }

  output_cache_ = output.clone();
  return output;
}

inline Tensor Softmax::backward(const Tensor &grad_output) {
  Tensor grad_input(grad_output.shape(), grad_output.device_,
                    grad_output.dtype_);
  int batch_size = grad_output.shape().size() > 1 ? grad_output.shape()[0] : 1;
  int num_classes = grad_output.shape().size() > 1 ? grad_output.shape()[1]
                                                   : grad_output.shape()[0];

  const float *go_ptr = static_cast<const float *>(grad_output.data());
  const float *out_ptr = static_cast<const float *>(output_cache_.data());
  float *gi_ptr = static_cast<float *>(grad_input.data());

#ifdef MUNET_USE_CUDA
  if (grad_output.device_ == Device::CUDA) {
    cuda_kernels::softmax_backward(go_ptr, out_ptr, gi_ptr, batch_size,
                                   num_classes);
    return grad_input;
  }
#endif

  for (int b = 0; b < batch_size; ++b) {
    const float *go_row = go_ptr + b * num_classes;
    const float *out_row = out_ptr + b * num_classes;
    float *gi_row = gi_ptr + b * num_classes;

    float sum_out_go = 0.0f;
    for (int j = 0; j < num_classes; ++j) {
      sum_out_go += out_row[j] * go_row[j];
    }

    for (int i = 0; i < num_classes; ++i) {
      gi_row[i] = out_row[i] * (go_row[i] - sum_out_go);
    }
  }
  return grad_input;
}
} // namespace munet
