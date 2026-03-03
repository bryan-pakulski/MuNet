#pragma once
#include "layer.hpp"

#include <vector>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cstring>
#include <limits>

namespace munet {

	class Flatten : public Layer {
	public:
			inline Tensor forward(const Tensor& input) override;
			inline Tensor backward(const Tensor& grad_output) override;
			inline std::string get_onnx_op_type() const override { return "Flatten"; }

	private:
			std::vector<int> input_shape_;
	};

	inline Tensor Flatten::forward(const Tensor& input) {
			input_shape_ = input.shape();
			int batch_size = input_shape_.empty() ? 1 : input_shape_[0];
			int flat_size = 1;
			for (size_t i = 1; i < input_shape_.size(); ++i) flat_size *= input_shape_[i];
			
			Tensor output({batch_size, flat_size}, input.device_, input.dtype_);
			std::memcpy(output.data(), input.data(), input.bytes());
			return output;
	}

	inline Tensor Flatten::backward(const Tensor& grad_output) {
			Tensor grad_input(input_shape_, grad_output.device_, grad_output.dtype_);
			std::memcpy(grad_input.data(), grad_output.data(), grad_output.bytes());
			return grad_input;
	}

	class MaxPool2D : public Layer {
	public:
			inline MaxPool2D(int kernel_size, int stride);
			inline Tensor forward(const Tensor& input) override;
			inline Tensor backward(const Tensor& grad_output) override;
			inline std::string get_onnx_op_type() const override { return "MaxPool"; }

	private:
			int kernel_size_;
			int stride_;
			std::vector<int> input_shape_;
			std::vector<int> max_indices_; // Cache for routing gradients
	};

	inline MaxPool2D::MaxPool2D(int kernel_size, int stride) 
			: kernel_size_(kernel_size), stride_(stride) {}

	inline Tensor MaxPool2D::forward(const Tensor& input) {
			input_shape_ = input.shape();
			int N = input_shape_[0], C = input_shape_[1], H = input_shape_[2], W = input_shape_[3];
			int OH = (H - kernel_size_) / stride_ + 1;
			int OW = (W - kernel_size_) / stride_ + 1;

			Tensor output({N, C, OH, OW}, input.device_, input.dtype_);
			max_indices_.resize(output.size());

			const float* in_ptr = static_cast<const float*>(input.data());
			float* out_ptr = static_cast<float*>(output.data());

			for (int b = 0; b < N; ++b) {
					for (int c = 0; c < C; ++c) {
							for (int oh = 0; oh < OH; ++oh) {
									for (int ow = 0; ow < OW; ++ow) {
											float max_val = std::numeric_limits<float>::lowest();
											int max_idx = -1;

											for (int kh = 0; kh < kernel_size_; ++kh) {
													for (int kw = 0; kw < kernel_size_; ++kw) {
															int ih = oh * stride_ + kh;
															int iw = ow * stride_ + kw;
															int in_idx = b * (C * H * W) + c * (H * W) + ih * W + iw;

															if (in_ptr[in_idx] > max_val) {
																	max_val = in_ptr[in_idx];
																	max_idx = in_idx;
															}
													}
											}
											int out_idx = b * (C * OH * OW) + c * (OH * OW) + oh * OW + ow;
											out_ptr[out_idx] = max_val;
											max_indices_[out_idx] = max_idx;
									}
							}
					}
			}
			return output;
	}

	inline Tensor MaxPool2D::backward(const Tensor& grad_output) {
			Tensor grad_input(input_shape_, grad_output.device_, grad_output.dtype_);
			grad_input.zero();

			const float* go_ptr = static_cast<const float*>(grad_output.data());
			float* gi_ptr = static_cast<float*>(grad_input.data());

			for (size_t i = 0; i < grad_output.size(); ++i) {
					int in_idx = max_indices_[i];
					if (in_idx != -1) {
							gi_ptr[in_idx] += go_ptr[i];
					}
			}
			return grad_input;
	}

	class Conv2D : public Layer {
	public:
			inline Conv2D(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);
			inline Tensor forward(const Tensor& input) override;
			inline Tensor backward(const Tensor& grad_output) override;
			
			inline std::unordered_map<std::string, Tensor*> get_parameters() override {
					return {{"weight", &weight_}, {"bias", &bias_}};
			}
			inline std::unordered_map<std::string, Tensor*> get_gradients() override {
					return {{"weight", weight_.grad()}, {"bias", bias_.grad()}};
			}
			inline std::string get_onnx_op_type() const override { return "Conv"; }

	private:
			int in_channels_, out_channels_, kernel_size_, stride_, padding_;
			Tensor weight_, bias_;
			Tensor input_cache_{std::vector<int>{}};
	};

	inline Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding)
			: in_channels_(in_channels), out_channels_(out_channels), 
				kernel_size_(kernel_size), stride_(stride), padding_(padding),
				weight_({out_channels, in_channels, kernel_size, kernel_size}),
				bias_({out_channels}) 
	{
			float scale = std::sqrt(2.0f / (in_channels_ * kernel_size_ * kernel_size_));
			float* w_ptr = static_cast<float*>(weight_.data());
			for (size_t i = 0; i < weight_.size(); ++i) {
					float r = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
					w_ptr[i] = r * scale;
			}
			
			bias_.zero();
			weight_.allocate_grad();
			bias_.allocate_grad();
	}

	inline Tensor Conv2D::forward(const Tensor& input) {
			input_cache_ = input.clone();
			int N = input.shape()[0], C = input.shape()[1], H = input.shape()[2], W = input.shape()[3];
			int OH = (H + 2 * padding_ - kernel_size_) / stride_ + 1;
			int OW = (W + 2 * padding_ - kernel_size_) / stride_ + 1;

			Tensor output({N, out_channels_, OH, OW}, input.device_, input.dtype_);
			
			const float* in_ptr = static_cast<const float*>(input.data());
			const float* w_ptr  = static_cast<const float*>(weight_.data());
			const float* b_ptr  = static_cast<const float*>(bias_.data());
			float* out_ptr      = static_cast<float*>(output.data());

			for (int b = 0; b < N; ++b) {
					for (int oc = 0; oc < out_channels_; ++oc) {
							for (int oh = 0; oh < OH; ++oh) {
									for (int ow = 0; ow < OW; ++ow) {
											float sum = b_ptr[oc];
											
											for (int ic = 0; ic < in_channels_; ++ic) {
													for (int kh = 0; kh < kernel_size_; ++kh) {
															for (int kw = 0; kw < kernel_size_; ++kw) {
																	int ih = oh * stride_ - padding_ + kh;
																	int iw = ow * stride_ - padding_ + kw;
																	
																	if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
																			int in_idx = b*(C*H*W) + ic*(H*W) + ih*W + iw;
																			int w_idx = oc*(C*kernel_size_*kernel_size_) + ic*(kernel_size_*kernel_size_) + kh*kernel_size_ + kw;
																			sum += in_ptr[in_idx] * w_ptr[w_idx];
																	}
															}
													}
											}
											out_ptr[b*(out_channels_*OH*OW) + oc*(OH*OW) + oh*OW + ow] = sum;
									}
							}
					}
			}
			return output;
	}

	inline Tensor Conv2D::backward(const Tensor& grad_output) {
			int N = input_cache_.shape()[0], C = input_cache_.shape()[1];
			int H = input_cache_.shape()[2], W = input_cache_.shape()[3];
			int OH = grad_output.shape()[2], OW = grad_output.shape()[3];

			Tensor grad_input(input_cache_.shape(), input_cache_.device_, input_cache_.dtype_);
			grad_input.zero();

			const float* go_ptr = static_cast<const float*>(grad_output.data());
			const float* in_ptr = static_cast<const float*>(input_cache_.data());
			const float* w_ptr  = static_cast<const float*>(weight_.data());
			
			float* gi_ptr = static_cast<float*>(grad_input.data());
			float* gw_ptr = static_cast<float*>(weight_.grad()->data());
			float* gb_ptr = static_cast<float*>(bias_.grad()->data());

			for (int b = 0; b < N; ++b) {
					for (int oc = 0; oc < out_channels_; ++oc) {
							for (int oh = 0; oh < OH; ++oh) {
									for (int ow = 0; ow < OW; ++ow) {
											float go_val = go_ptr[b*(out_channels_*OH*OW) + oc*(OH*OW) + oh*OW + ow];
											gb_ptr[oc] += go_val; // d_bias
											
											for (int ic = 0; ic < in_channels_; ++ic) {
													for (int kh = 0; kh < kernel_size_; ++kh) {
															for (int kw = 0; kw < kernel_size_; ++kw) {
																	int ih = oh * stride_ - padding_ + kh;
																	int iw = ow * stride_ - padding_ + kw;
																	
																	if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
																			int in_idx = b*(C*H*W) + ic*(H*W) + ih*W + iw;
																			int w_idx = oc*(C*kernel_size_*kernel_size_) + ic*(kernel_size_*kernel_size_) + kh*kernel_size_ + kw;
																			
																			gw_ptr[w_idx] += in_ptr[in_idx] * go_val; // d_weight
																			gi_ptr[in_idx] += w_ptr[w_idx] * go_val;  // d_input
																	}
															}
													}
											}
									}
							}
					}
			}
			return grad_input;
	}
}
