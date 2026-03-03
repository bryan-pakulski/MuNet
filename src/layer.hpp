#pragma once

#include "kernels.hpp"
#include "tensor.hpp"
#include "string"
#include "unordered_map"

#include <vector>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <cmath>
#include <cstring>


namespace munet {

	class Layer {
	public:
			virtual ~Layer() = default;
			virtual Tensor forward(const Tensor& input) = 0;
			virtual Tensor backward(const Tensor& grad_output) = 0;
			virtual std::unordered_map<std::string, Tensor*> get_parameters() { return {}; }
			virtual std::unordered_map<std::string, Tensor*> get_gradients() { return {}; }
			virtual std::string get_onnx_op_type() const = 0; 
			
			virtual void train() {}
			virtual void eval() {}
	};

	class Linear : public Layer {
	public:
			inline Linear(int in_features, int out_features);
			inline Tensor forward(const Tensor& input) override;
			inline Tensor backward(const Tensor& grad_output) override;
			
			inline std::unordered_map<std::string, Tensor*> get_parameters() override {
					return {{"weight", &weight_}, {"bias", &bias_}};
			}
			inline std::unordered_map<std::string, Tensor*> get_gradients() override {
					return {{"weight", weight_.grad()}, {"bias", bias_.grad()}};
			}
			inline std::string get_onnx_op_type() const override { return "Gemm"; }

	private:
			int in_features_, out_features_;
			Tensor weight_, bias_;
			Tensor input_cache_{std::vector<int>{}}; // Empty tensor for backprop caching
	};

	inline Linear::Linear(int in_features, int out_features) 
			: in_features_(in_features), out_features_(out_features),
				weight_({in_features, out_features}), bias_({out_features})
	{
			// Kaiming/He Initialization for weights, zero for bias (FP32 assumption)
			float* w_ptr = static_cast<float*>(weight_.data());
			float scale = std::sqrt(2.0f / in_features_);
			for (size_t i = 0; i < weight_.size(); ++i) {
					float r = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
					w_ptr[i] = r * scale;
			}
			
			bias_.zero();
			weight_.allocate_grad();
			bias_.allocate_grad();
	}

	inline Tensor Linear::forward(const Tensor& input) {
			input_cache_ = input.clone(); // Save for backward pass
			
			int batch_size = input.shape()[0];
			Tensor output({batch_size, out_features_}, input_cache_.device_, input_cache_.dtype_);
			
			const float* in_ptr = static_cast<const float*>(input.data());
			const float* w_ptr  = static_cast<const float*>(weight_.data());
			const float* b_ptr  = static_cast<const float*>(bias_.data());
			float* out_ptr      = static_cast<float*>(output.data());

#ifdef MUNET_USE_CUDA
			if (input.device_ == Device::CUDA) {
					float alpha = 1.0f;
					float beta = 0.0f;
					
					cublasSgemm(Context::instance().cublas_handle,
											CUBLAS_OP_N, CUBLAS_OP_N,
											out_features_, batch_size, in_features_,
											&alpha,
											w_ptr, out_features_, 
											in_ptr, in_features_, 
											&beta,
											out_ptr, out_features_);
					
					// Correctly call the now-declared kernel
					cuda_kernels::add_bias(out_ptr, b_ptr, batch_size, out_features_);
					
					return output;
			}
#endif

			for (int b = 0; b < batch_size; ++b) {
					for (int out_c = 0; out_c < out_features_; ++out_c) {
							float sum = b_ptr[out_c];
							for (int in_c = 0; in_c < in_features_; ++in_c) {
									sum += in_ptr[b * in_features_ + in_c] * w_ptr[in_c * out_features_ + out_c];
							}
							out_ptr[b * out_features_ + out_c] = sum;
					}
			}
			return output;
	}

	inline Tensor Linear::backward(const Tensor& grad_output) {
			int batch_size = grad_output.shape()[0];
			Tensor grad_input({batch_size, in_features_}, input_cache_.device_, input_cache_.dtype_);
			grad_input.zero(); 
			
			const float* go_ptr = static_cast<const float*>(grad_output.data());
			const float* in_ptr = static_cast<const float*>(input_cache_.data());
			const float* w_ptr  = static_cast<const float*>(weight_.data());
			
			float* gi_ptr = static_cast<float*>(grad_input.data());
			float* gw_ptr = static_cast<float*>(weight_.grad()->data()); 
			float* gb_ptr = static_cast<float*>(bias_.grad()->data());   

#ifdef MUNET_USE_CUDA
			if (grad_output.device_ == Device::CUDA) {
            float alpha = 1.0f;
						float beta = 0.0f;

            // 1. Grad Input = Grad Output * Weight^T
            cublasSgemm(Context::instance().cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    in_features_, batch_size, out_features_,
                                    &alpha,
                                    w_ptr, out_features_,
                                    go_ptr, out_features_,
                                    &beta,
                                    gi_ptr, in_features_);

            // 2. Grad Weight = Input^T * Grad Output
            cublasSgemm(Context::instance().cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_T,
                                    out_features_, in_features_, batch_size,
                                    &alpha,
                                    go_ptr, out_features_,
                                    in_ptr, in_features_,
                                    &beta,
                                    gw_ptr, out_features_);

					return grad_input;
			}
#endif

			for (int b = 0; b < batch_size; ++b) {
					for (int out_c = 0; out_c < out_features_; ++out_c) {
							float go_val = go_ptr[b * out_features_ + out_c];
							gb_ptr[out_c] += go_val; 
							
							for (int in_c = 0; in_c < in_features_; ++in_c) {
									gw_ptr[in_c * out_features_ + out_c] += in_ptr[b * in_features_ + in_c] * go_val; 
									gi_ptr[b * in_features_ + in_c] += go_val * w_ptr[in_c * out_features_ + out_c];  
							}
					}
			}
			return grad_input;
	}
}
