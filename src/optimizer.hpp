#pragma once

#include "tensor.hpp"

namespace munet {

	class Optimizer {
	public:
			Optimizer(const std::vector<Tensor*>& parameters) : parameters_(parameters) {}
			virtual ~Optimizer() = default;

			virtual void step() = 0;

			virtual void zero_grad() {
					for (auto* p : parameters_) {
							if (p->grad()) {
									p->grad()->zero();
							}
					}
			}

	protected:
			std::vector<Tensor*> parameters_;
	};

	class SGD : public Optimizer {
	public:
			SGD(const std::vector<Tensor*>& parameters, float lr) 
					: Optimizer(parameters), lr_(lr) {}

			void step() override {
					for (auto* p : parameters_) {
							if (!p->grad()) continue;

							float* w_ptr = static_cast<float*>(p->data());
							const float* g_ptr = static_cast<const float*>(p->grad()->data());
							int size = p->size();

#ifdef MUNET_USE_CUDA
							if (p->device_ == Device::CUDA) {
									cuda_kernels::sgd_step(w_ptr, g_ptr, lr_, size);
									continue;
							}
#endif

							for (int i = 0; i < size; ++i) {
									w_ptr[i] -= lr_ * g_ptr[i];
							}
					}
			}

	private:
			float lr_;
	};
}
