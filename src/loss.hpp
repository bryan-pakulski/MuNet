#pragma once

#include "tensor.hpp"

#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <cstring>

namespace munet {

	inline float cross_entropy_loss(const Tensor& logits, const Tensor& targets, Tensor& grad_output) {
			if (logits.size() != targets.size()) throw std::runtime_error("CE size mismatch");
      if (logits.device_ != targets.device_) throw std::runtime_error("Device mismatch: logits and targets must be on the same device");

			int batch_size = logits.shape().size() > 1 ? logits.shape()[0] : 1;
			int num_classes = logits.shape().size() > 1 ? logits.shape()[1] : logits.shape()[0];

#ifdef MUNET_USE_CUDA
        if (logits.device_ == Device::CUDA && targets.device_ == Device::CUDA) {
            return cuda_kernels::cross_entropy_loss_cuda(
                static_cast<const float*>(logits.data()),
                static_cast<const float*>(targets.data()),
                static_cast<float*>(grad_output.data()),
                batch_size, 
                num_classes
            );
        }
#endif

			const float* l_ptr = static_cast<const float*>(logits.data());
			const float* t_ptr = static_cast<const float*>(targets.data());
			float* g_ptr = static_cast<float*>(grad_output.data());

			float total_loss = 0.0f;

			for (int b = 0; b < batch_size; ++b) {
					const float* logit_row = l_ptr + b * num_classes;
					const float* target_row = t_ptr + b * num_classes;
					float* grad_row = g_ptr + b * num_classes;

					float max_val = logit_row[0];
					for (int i = 1; i < num_classes; ++i) {
							if (logit_row[i] > max_val) max_val = logit_row[i];
					}

					float sum_exp = 0.0f;
					std::vector<float> probs(num_classes);
					for (int i = 0; i < num_classes; ++i) {
							probs[i] = std::exp(logit_row[i] - max_val);
							sum_exp += probs[i];
					}

					for (int i = 0; i < num_classes; ++i) {
							probs[i] /= sum_exp;
							
							if (target_row[i] > 0.0f) {
									total_loss -= target_row[i] * std::log(probs[i] + 1e-9f);
							}

							grad_row[i] = (probs[i] - target_row[i]) / batch_size;
					}
			}

			return total_loss / batch_size;
	}

	inline float mse_loss(const Tensor& pred, const Tensor& target, Tensor& grad_output) {
			if (pred.size() != target.size()) throw std::runtime_error("MSE size mismatch");
			
			const float* p_ptr = static_cast<const float*>(pred.data());
			const float* t_ptr = static_cast<const float*>(target.data());
			float* g_ptr = static_cast<float*>(grad_output.data());

 #ifdef MUNET_USE_CUDA                                                                                                                                                                    
             if (pred.device_ == Device::CUDA) {                                                                                                                                          
                     return cuda_kernels::mse_loss_cuda(p_ptr, t_ptr, g_ptr, pred.size());                                                                                                
             }                                                                                                                                                                            
 #endif 
	
			float loss = 0.0f;
			float n = static_cast<float>(pred.size());
			
			for (size_t i = 0; i < pred.size(); ++i) {
					float diff = p_ptr[i] - t_ptr[i];
					loss += diff * diff;
					g_ptr[i] = 2.0f * diff / n; // Derivative of MSE
			}
			return loss / n;
	}
}
