#pragma once

#ifdef MUNET_USE_CUDA
namespace cuda_kernels {
void softmax_forward(const float *in, float *out, int batch_size,
                     int num_classes);
void softmax_backward(const float *go, const float *out, float *gi,
                      int batch_size, int num_classes);
void maxpool_forward(const float *in, float *out, int *indices, int N, int C,
                     int H, int W, int OH, int OW, int K, int S);
void maxpool_backward(const float *go, float *gi, const int *indices, int size);
void conv2d_forward(const float *in, const float *w, const float *b, float *out,
                    int N, int IC, int OC, int H, int W, int OH, int OW, int K,
                    int S, int P);
void conv2d_backward(const float *go, const float *in, const float *w,
                     float *gi, float *gw, float *gb, int N, int IC, int OC,
                     int H, int W, int OH, int OW, int K, int S, int P);
void batchnorm2d_forward(const float *in, float *out, float *save_mean,
                         float *save_var, float *save_inv_std, float *run_mean,
                         float *run_var, const float *weight, const float *bias,
                         int N, int C, int H, int W, float eps, float momentum,
                         bool training);

void batchnorm2d_backward(const float *go, const float *in, float *gi,
                          float *gw, float *gb, const float *mean,
                          const float *inv_std, const float *weight, int N,
                          int C, int H, int W);
void upsample_forward(const float *in, float *out, int N, int C, int H, int W,
                      int OH, int OW, int SF);
void upsample_backward(const float *go, float *gi, int N, int C, int H, int W,
                       int OH, int OW, int SF);
float mse_loss_cuda(const float *p, const float *t, float *grad_output,
                    int size);
void relu_forward(const float *in, float *out, size_t size);
void relu_backward(const float *go, const float *in, float *gi, size_t size);
void linear_forward(const float *in, const float *w, const float *b, float *out,
                    int batch_size, int in_f, int out_f);
void linear_backward_dx(const float *go, const float *w, float *gi,
                        int batch_size, int in_f, int out_f);
void linear_backward_dw_db(const float *go, const float *in, float *gw,
                           float *gb, int batch_size, int in_f, int out_f);
void sgd_step(float *w, const float *grad, float lr, int size);
void add_bias(float *out, const float *b, int batch_size, int out_f);
void elementwise_mul(const float *a, const float *b, float *out, size_t size);
float cross_entropy_loss_cuda(const float *logits, const float *targets,
                              float *grad_output, int batch_size,
                              int num_classes);
} // namespace cuda_kernels
#endif
