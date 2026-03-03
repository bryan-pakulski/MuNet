#pragma once

#ifdef MUNET_USE_CUDA
namespace cuda_kernels {
    void relu_forward(const float* in, float* out, size_t size);
    void relu_backward(const float* go, const float* in, float* gi, size_t size);
    void linear_forward(const float* in, const float* w, const float* b, float* out, int batch_size, int in_f, int out_f);
    void linear_backward_dx(const float* go, const float* w, float* gi, int batch_size, int in_f, int out_f);
    void linear_backward_dw_db(const float* go, const float* in, float* gw, float* gb, int batch_size, int in_f, int out_f);
    void sgd_step(float* w, const float* grad, float lr, int size);
    void add_bias(float* out, const float* b, int batch_size, int out_f);
    void elementwise_mul(const float* a, const float* b, float* out, size_t size);
    float cross_entropy_loss_cuda(const float* logits, const float* targets, float* grad_output, int batch_size, int num_classes);
}
#endif
