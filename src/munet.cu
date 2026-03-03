// munet.cu
#include "munet.hpp"

#ifdef MUNET_USE_CUDA
#include <cuda_runtime.h>

namespace cuda_kernels {

__global__ void relu_forward_kernel(const float *in, float *out, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

__global__ void relu_backward_kernel(const float *go, const float *in,
                                     float *gi, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    gi[i] = in[i] > 0.0f ? go[i] : 0.0f;
}

__global__ void linear_forward_kernel(const float *in, const float *w,
                                      const float *b, float *out,
                                      int batch_size, int in_f, int out_f) {
  int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int o_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (b_idx < batch_size && o_idx < out_f) {
    float sum = b[o_idx];
    for (int i = 0; i < in_f; ++i) {
      sum += in[b_idx * in_f + i] * w[i * out_f + o_idx];
    }
    out[b_idx * out_f + o_idx] = sum;
  }
}

__global__ void linear_backward_dx_kernel(const float *go, const float *w,
                                          float *gi, int batch_size, int in_f,
                                          int out_f) {
  int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (b_idx < batch_size && i_idx < in_f) {
    float sum = 0.0f;
    for (int o = 0; o < out_f; ++o) {
      sum += go[b_idx * out_f + o] * w[i_idx * out_f + o];
    }
    gi[b_idx * in_f + i_idx] = sum;
  }
}

__global__ void linear_backward_dw_db_kernel(const float *go, const float *in,
                                             float *gw, float *gb,
                                             int batch_size, int in_f,
                                             int out_f) {
  int i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int o_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (i_idx < in_f && o_idx < out_f) {
    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
      sum += in[b * in_f + i_idx] * go[b * out_f + o_idx];
    }
    gw[i_idx * out_f + o_idx] += sum;
  }

  if (i_idx == 0 && o_idx < out_f) {
    float b_sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
      b_sum += go[b * out_f + o_idx];
    }
    gb[o_idx] += b_sum;
  }
}

__global__ void sgd_step_kernel(float *w, const float *grad, float lr,
                                int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    w[i] -= lr * grad[i];
  }
}

// Add bias 'b' (size out_f) to 'out' (size batch_size * out_f)
// out is [Batch, Out] in memory.
__global__ void add_bias_kernel(float *out, const float *b, int size,
                                int out_f) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int col = idx % out_f;
    out[idx] += b[col];
  }
}

__global__ void elementwise_mul_kernel(const float *a, const float *b,
                                       float *out, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    out[i] = a[i] * b[i];
}

// Flat cross entryopy (for classification)
__global__ void cross_entropy_gradient_kernel(const float *logits,
                                              const float *targets,
                                              float *grad_input,
                                              float *loss_out, int batch_size,
                                              int num_classes) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < batch_size) {
    const float *logit_row = logits + b * num_classes;
    const float *target_row = targets + b * num_classes;
    float *grad_row = grad_input + b * num_classes;

    // 1. Find Max (for numerical stability)
    float max_val = -1e30f;
    for (int i = 0; i < num_classes; ++i) {
      if (logit_row[i] > max_val)
        max_val = logit_row[i];
    }

    // 2. Sum Exponentials
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
      sum_exp += expf(logit_row[i] - max_val);
    }
    // FIX: Prevent division by zero if sum_exp is tiny (unlikely with exp, but
    // safe)
    if (sum_exp < 1e-7f)
      sum_exp = 1e-7f;

    // 3. Calculate Gradient
    float sample_loss = 0.0f;
    const float epsilon = 1e-7f;
    for (int i = 0; i < num_classes; ++i) {
      float p = expf(logit_row[i] - max_val) / sum_exp;

      // Numerical stability clamping
      if (p < epsilon)
        p = epsilon;
      if (p > 1.0f - epsilon)
        p = 1.0f - epsilon;

      float t = target_row[i];

      if (t > 0.0f) {
        sample_loss -= t * logf(p);
      }
      grad_row[i] = (p - t) / (float)batch_size;
    }

    atomicAdd(loss_out, sample_loss / (float)batch_size);
  }
}

// Spatial Cross Entropy (for Segmentation NCHW)
__global__ void spatial_cross_entropy_kernel(const float *logits,
                                             const float *targets, float *grad,
                                             float *loss, int N, int C, int H,
                                             int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int spatial_size = H * W;
  int total_pixels = N * spatial_size;

  if (idx < total_pixels) {
    int n = idx / spatial_size;
    int hw = idx % spatial_size;

    // In NCHW, the c-th class value for pixel hw is at: n*C*HW + c*HW + hw
    // The stride between classes is spatial_size (H*W).
    int base_offset = n * C * spatial_size + hw;
    int stride = spatial_size;

    // 1. Max
    float max_val = -1e30f;
    for (int c = 0; c < C; ++c) {
      float val = logits[base_offset + c * stride];
      if (val > max_val)
        max_val = val;
    }

    // 2. Sum Exp
    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
      sum_exp += expf(logits[base_offset + c * stride] - max_val);
    }
    if (sum_exp < 1e-7f)
      sum_exp = 1e-7f;

    // 3. Grad & Loss
    float pixel_loss = 0.0f;
    const float epsilon = 1e-7f;
    // Normalize by Total Pixels (N * H * W) for resolution invariance
    float norm_factor = 1.0f / (float)(N * H * W);

    for (int c = 0; c < C; ++c) {
      float p = expf(logits[base_offset + c * stride] - max_val) / sum_exp;
      if (p < epsilon)
        p = epsilon;
      if (p > 1.0f - epsilon)
        p = 1.0f - epsilon;

      float t = targets[base_offset + c * stride];

      if (t > 0.0f)
        pixel_loss -= t * logf(p);

      grad[base_offset + c * stride] = (p - t) * norm_factor;
    }
    atomicAdd(loss, pixel_loss * norm_factor);
  }
}

__global__ void softmax_forward_kernel(const float *in, float *out,
                                       int batch_size, int num_classes) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < batch_size) {
    const float *in_row = in + b * num_classes;
    float *out_row = out + b * num_classes;

    float max_val = -1e30f;
    for (int i = 0; i < num_classes; ++i)
      if (in_row[i] > max_val)
        max_val = in_row[i];

    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
      out_row[i] = expf(in_row[i] - max_val);
      sum_exp += out_row[i];
    }
    for (int i = 0; i < num_classes; ++i)
      out_row[i] /= sum_exp;
  }
}

__global__ void softmax_backward_kernel(const float *go, const float *out,
                                        float *gi, int batch_size,
                                        int num_classes) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < batch_size) {
    const float *go_row = go + b * num_classes;
    const float *out_row = out + b * num_classes;
    float *gi_row = gi + b * num_classes;

    float dot = 0.0f;
    for (int i = 0; i < num_classes; ++i)
      dot += out_row[i] * go_row[i];
    for (int i = 0; i < num_classes; ++i)
      gi_row[i] = out_row[i] * (go_row[i] - dot);
  }
}

__global__ void maxpool_forward_kernel(const float *in, float *out,
                                       int *indices, int N, int C, int H, int W,
                                       int OH, int OW, int K, int S) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * C * OH * OW) {
    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int c = (idx / (OW * OH)) % C;
    int n = idx / (OW * OH * C);

    float max_val = -1e30f;
    int max_idx = -1;
    for (int kh = 0; kh < K; ++kh) {
      for (int kw = 0; kw < K; ++kw) {
        int ih = oh * S + kh;
        int iw = ow * S + kw;
        int in_idx = ((n * C + c) * H + ih) * W + iw;
        if (in[in_idx] > max_val) {
          max_val = in[in_idx];
          max_idx = in_idx;
        }
      }
    }
    out[idx] = max_val;
    indices[idx] = max_idx;
  }
}

__global__ void maxpool_backward_kernel(const float *go, float *gi,
                                        const int *indices, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    atomicAdd(&gi[indices[idx]], go[idx]);
}

__global__ void conv2d_forward_kernel(const float *in, const float *w,
                                      const float *b, float *out, int N, int IC,
                                      int OC, int H, int W, int OH, int OW,
                                      int K, int S, int P) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * OC * OH * OW) {
    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int oc = (idx / (OW * OH)) % OC;
    int n = idx / (OW * OH * OC);

    float sum = b[oc];
    for (int ic = 0; ic < IC; ++ic) {
      for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
          int ih = oh * S - P + kh;
          int iw = ow * S - P + kw;
          if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            sum += in[((n * IC + ic) * H + ih) * W + iw] *
                   w[((oc * IC + ic) * K + kh) * K + kw];
          }
        }
      }
    }
    out[idx] = sum;
  }
}

__global__ void upsample_forward_kernel(const float *in, float *out, int N,
                                        int C, int H, int W, int OH, int OW,
                                        int SF) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * C * OH * OW) {
    int ow = idx % OW, oh = (idx / OW) % OH, c = (idx / (OW * OH)) % C,
        n = idx / (OW * OH * C);
    out[idx] = in[((n * C + c) * H + (oh / SF)) * W + (ow / SF)];
  }
}

__global__ void mse_loss_kernel(const float *p, const float *t, float *g,
                                float *loss, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    float diff = p[i] - t[i];
    atomicAdd(loss, (diff * diff) / size);
    g[i] = (2.0f * diff) / size;
  }
}

__global__ void conv2d_backward_bias_kernel(const float *go, float *gb, int N,
                                            int OC, int OH, int OW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * OC * OH * OW) {
    int oc = (idx / (OH * OW)) % OC;
    atomicAdd(&gb[oc], go[idx]);
  }
}

__global__ void conv2d_backward_weight_kernel(const float *go, const float *in,
                                              float *gw, int N, int IC, int OC,
                                              int H, int W, int OH, int OW,
                                              int K, int S, int P) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * OC * OH * OW) {
    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int oc = (idx / (OW * OH)) % OC;
    int n = idx / (OW * OH * OC);

    float go_val = go[idx];

    for (int ic = 0; ic < IC; ++ic) {
      for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
          int ih = oh * S - P + kh;
          int iw = ow * S - P + kw;
          if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            int in_idx = ((n * IC + ic) * H + ih) * W + iw;
            int w_idx = ((oc * IC + ic) * K + kh) * K + kw;
            atomicAdd(&gw[w_idx], in[in_idx] * go_val);
          }
        }
      }
    }
  }
}

__global__ void conv2d_backward_input_kernel(const float *go, const float *w,
                                             float *gi, int N, int IC, int OC,
                                             int H, int W, int OH, int OW,
                                             int K, int S, int P) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * IC * H * W) {
    int iw = idx % W;
    int ih = (idx / W) % H;
    int ic = (idx / (W * H)) % IC;
    int n = idx / (W * H * IC);

    float sum = 0.0f;

    for (int oc = 0; oc < OC; ++oc) {
      for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
          // ih = oh * S - P + kh  =>  oh * S = ih + P - kh
          int num_h = ih + P - kh;
          int num_w = iw + P - kw;

          if (num_h % S == 0 && num_w % S == 0) {
            int oh = num_h / S;
            int ow = num_w / S;

            if (oh >= 0 && oh < OH && ow >= 0 && ow < OW) {
              int go_idx = ((n * OC + oc) * OH + oh) * OW + ow;
              int w_idx = ((oc * IC + ic) * K + kh) * K + kw;
              sum += go[go_idx] * w[w_idx];
            }
          }
        }
      }
    }
    gi[idx] = sum;
  }
}

__global__ void upsample_backward_kernel(const float *go, float *gi, int N,
                                         int C, int H, int W, int OH, int OW,
                                         int SF) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * C * OH * OW) {
    // Map output pixel (grad_output) back to input pixel (grad_input)
    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int c = (idx / (OW * OH)) % C;
    int n = idx / (OW * OH * C);

    int ih = oh / SF;
    int iw = ow / SF;

    int gi_idx = ((n * C + c) * H + ih) * W + iw;
    atomicAdd(&gi[gi_idx], go[idx]);
  }
}

__global__ void bn_mean_kernel(const float *in, float *mean, int N, int C,
                               int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * C * H * W) {
    int c = (idx / (W * H)) % C;
    float val = in[idx] / (N * H * W);
    atomicAdd(&mean[c], val);
  }
}

__global__ void bn_var_kernel(const float *in, const float *mean, float *var,
                              int N, int C, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * C * H * W) {
    int c = (idx / (W * H)) % C;
    float diff = in[idx] - mean[c];
    float val = (diff * diff) / (N * H * W);
    atomicAdd(&var[c], val);
  }
}

__global__ void bn_update_stats_kernel(const float *mean, const float *var,
                                       float *inv_std, float *run_mean,
                                       float *run_var, int M, int C, float eps,
                                       float momentum) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) {
    float m = mean[c];
    float v = var[c];
    if (v < 0.0f)
      v = 0.0f; // Numerical safety
    inv_std[c] = rsqrtf(v + eps);
    run_mean[c] = (1.0f - momentum) * run_mean[c] + momentum * m;
    run_var[c] = (1.0f - momentum) * run_var[c] +
                 momentum * (v * M / (M > 1 ? M - 1 : 1));
  }
}

__global__ void bn_forward_train_kernel(const float *in, float *out,
                                        const float *mean, const float *inv_std,
                                        const float *weight, const float *bias,
                                        int N, int C, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * C * H * W) {
    int c = (idx / (W * H)) % C;
    float norm = (in[idx] - mean[c]) * inv_std[c];
    out[idx] = norm * weight[c] + bias[c];
  }
}

__global__ void bn_forward_eval_kernel(const float *in, float *out,
                                       const float *run_mean,
                                       const float *run_var,
                                       const float *weight, const float *bias,
                                       int N, int C, int H, int W, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * C * H * W) {
    int c = (idx / (W * H)) % C;
    float norm = (in[idx] - run_mean[c]) * rsqrtf(run_var[c] + eps);
    out[idx] = norm * weight[c] + bias[c];
  }
}

__global__ void bn_bw_pass1_kernel(const float *go, const float *in,
                                   const float *mean, const float *inv_std,
                                   float *sum_go_xhat, float *sum_go, int N,
                                   int C, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * C * H * W) {
    int c = (idx / (W * H)) % C;
    float go_val = go[idx];
    float xhat = (in[idx] - mean[c]) * inv_std[c];

    atomicAdd(&sum_go[c], go_val);
    atomicAdd(&sum_go_xhat[c], go_val * xhat);
  }
}

__global__ void bn_bw_pass2_kernel(const float *go, const float *in, float *gi,
                                   const float *mean, const float *inv_std,
                                   const float *weight,
                                   const float *sum_go_xhat,
                                   const float *sum_go, int N, int C, int H,
                                   int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * C * H * W) {
    int c = (idx / (W * H)) % C;
    int M = N * H * W;
    float go_val = go[idx];
    float xhat = (in[idx] - mean[c]) * inv_std[c];

    float dx = (weight[c] * inv_std[c] / M) *
               (M * go_val - sum_go[c] - xhat * sum_go_xhat[c]);
    gi[idx] = dx;
  }
}

__global__ void add_grads_kernel(float *gw, float *gb, const float *sum_go_xhat,
                                 const float *sum_go, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) {
    atomicAdd(&gw[c], sum_go_xhat[c]);
    atomicAdd(&gb[c], sum_go[c]);
  }
}

__global__ void adam_step_kernel(float *w, const float *g, float *m, float *v,
                                 float lr, float beta1, float beta2, float eps,
                                 float weight_decay, int step, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    float grad = g[i];
    if (weight_decay > 0.0f) {
      grad += w[i] * weight_decay;
    }

    float m_t = beta1 * m[i] + (1.0f - beta1) * grad;
    float v_t = beta2 * v[i] + (1.0f - beta2) * grad * grad;

    m[i] = m_t;
    v[i] = v_t;

    // Bias correction
    float bias_correction1 = 1.0f - powf(beta1, step);
    float bias_correction2 = 1.0f - powf(beta2, step);

    float m_hat = m_t / bias_correction1;
    float v_hat = v_t / bias_correction2;

    w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
  }
}

__global__ void sigmoid_forward_kernel(const float *in, float *out,
                                       size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = 1.0f / (1.0f + expf(-in[i]));
  }
}

__global__ void sigmoid_backward_kernel(const float *go, const float *out,
                                        float *gi, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    float s = out[i];
    gi[i] = go[i] * s * (1.0f - s);
  }
}

__global__ void tensor_add_kernel(const float *a, const float *b, float *out,
                                  size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    out[i] = a[i] + b[i];
}

// NCHW Concatenation along C dimension
__global__ void concat_forward_kernel(const float *src, float *dst,
                                      int n_stride_src, int n_stride_dst,
                                      int c_offset, int spatial_size, int C_src,
                                      int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    int hw = idx % spatial_size;
    int c_local = (idx / spatial_size) % C_src;
    int n = idx / (spatial_size * C_src);

    int src_idx = n * n_stride_src + c_local * spatial_size + hw;
    int dst_idx = n * n_stride_dst + (c_local + c_offset) * spatial_size + hw;
    dst[dst_idx] = src[src_idx];
  }
}

// Extract a subset of channels from src to dst (Backward of Concat)
__global__ void slice_kernel(const float *src, float *dst, int n_stride_src,
                             int n_stride_dst, int c_offset, int spatial_size,
                             int C_dst, int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    int hw = idx % spatial_size;
    int c_local = (idx / spatial_size) % C_dst;
    int n = idx / (spatial_size * C_dst);

    // Map dst (small) index back to src (big) index
    int dst_idx = n * n_stride_dst + c_local * spatial_size + hw;
    int src_idx = n * n_stride_src + (c_local + c_offset) * spatial_size + hw;

    dst[dst_idx] = src[src_idx];
  }
}

// Wrapper functions callable from standard C++

void concat_channel_backward(const float *src, float *dst, int N, int H, int W,
                             int C_total, int C_dst, int C_offset) {
  // Parallelize over the DST (smaller) tensor elements
  size_t total_elements = (size_t)N * C_dst * H * W;
  int spatial_size = H * W;
  int n_stride_src = C_total * spatial_size;
  int n_stride_dst = C_dst * spatial_size;

  int threads = 256;
  int blocks = (total_elements + 255) / 256;
  slice_kernel<<<blocks, threads>>>(src, dst, n_stride_src, n_stride_dst,
                                    C_offset, spatial_size, C_dst,
                                    total_elements);
}

void tensor_add(const float *a, const float *b, float *out, size_t size) {
  int threads = 256;
  int blocks = (size + 255) / 256;
  tensor_add_kernel<<<blocks, threads>>>(a, b, out, size);
}

void concat_channel_forward(const float *src, float *dst, int N, int H, int W,
                            int C_src, int C_total, int C_offset) {
  // total threads needed = N * C_src * H * W
  size_t total_elements = (size_t)N * C_src * H * W;
  int spatial_size = H * W;
  int n_stride_src = C_src * spatial_size;
  int n_stride_dst = C_total * spatial_size;

  int threads = 256;
  int blocks = (total_elements + 255) / 256;
  concat_forward_kernel<<<blocks, threads>>>(
      src, dst, n_stride_src, n_stride_dst, C_offset, spatial_size, C_src,
      total_elements);
}

void batchnorm2d_forward(const float *in, float *out, float *save_mean,
                         float *save_var, float *save_inv_std, float *run_mean,
                         float *run_var, const float *weight, const float *bias,
                         int N, int C, int H, int W, float eps, float momentum,
                         bool training) {
  int size = N * C * H * W;
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  if (training) {
    cudaMemset(save_mean, 0, C * sizeof(float));
    cudaMemset(save_var, 0, C * sizeof(float));

    bn_mean_kernel<<<blocks, threads>>>(in, save_mean, N, C, H, W);
    bn_var_kernel<<<blocks, threads>>>(in, save_mean, save_var, N, C, H, W);

    int c_blocks = (C + threads - 1) / threads;
    bn_update_stats_kernel<<<c_blocks, threads>>>(
        save_mean, save_var, save_inv_std, run_mean, run_var, N * H * W, C, eps,
        momentum);

    bn_forward_train_kernel<<<blocks, threads>>>(
        in, out, save_mean, save_inv_std, weight, bias, N, C, H, W);
  } else {
    bn_forward_eval_kernel<<<blocks, threads>>>(in, out, run_mean, run_var,
                                                weight, bias, N, C, H, W, eps);
  }
}

void batchnorm2d_backward(const float *go, const float *in, float *gi,
                          float *gw, float *gb, const float *mean,
                          const float *inv_std, const float *weight, int N,
                          int C, int H, int W) {
  int size = N * C * H * W;
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  float *d_sum_go, *d_sum_go_xhat;
  cudaMalloc(&d_sum_go, C * sizeof(float));
  cudaMalloc(&d_sum_go_xhat, C * sizeof(float));
  cudaMemset(d_sum_go, 0, C * sizeof(float));
  cudaMemset(d_sum_go_xhat, 0, C * sizeof(float));

  bn_bw_pass1_kernel<<<blocks, threads>>>(go, in, mean, inv_std, d_sum_go_xhat,
                                          d_sum_go, N, C, H, W);
  bn_bw_pass2_kernel<<<blocks, threads>>>(go, in, gi, mean, inv_std, weight,
                                          d_sum_go_xhat, d_sum_go, N, C, H, W);

  int c_blocks = (C + threads - 1) / threads;
  add_grads_kernel<<<c_blocks, threads>>>(gw, gb, d_sum_go_xhat, d_sum_go, C);

  cudaFree(d_sum_go);
  cudaFree(d_sum_go_xhat);
}

void conv2d_backward(const float *go, const float *in, const float *w,
                     float *gi, float *gw, float *gb, int N, int IC, int OC,
                     int H, int W, int OH, int OW, int K, int S, int P) {
  int size_go = N * OC * OH * OW;
  int size_gi = N * IC * H * W;

  // 1. Bias Gradients
  conv2d_backward_bias_kernel<<<(size_go + 255) / 256, 256>>>(go, gb, N, OC, OH,
                                                              OW);

  // 2. Weight Gradients
  conv2d_backward_weight_kernel<<<(size_go + 255) / 256, 256>>>(
      go, in, gw, N, IC, OC, H, W, OH, OW, K, S, P);

  // 3. Input Gradients
  conv2d_backward_input_kernel<<<(size_gi + 255) / 256, 256>>>(
      go, w, gi, N, IC, OC, H, W, OH, OW, K, S, P);
}

void upsample_backward(const float *go, float *gi, int N, int C, int H, int W,
                       int OH, int OW, int SF) {
  int size = N * C * OH * OW;
  upsample_backward_kernel<<<(size + 255) / 256, 256>>>(go, gi, N, C, H, W, OH,
                                                        OW, SF);
}

void softmax_forward(const float *in, float *out, int b, int c) {
  softmax_forward_kernel<<<(b + 255) / 256, 256>>>(in, out, b, c);
}
void softmax_backward(const float *go, const float *out, float *gi, int b,
                      int c) {
  softmax_backward_kernel<<<(b + 255) / 256, 256>>>(go, out, gi, b, c);
}
void maxpool_forward(const float *in, float *out, int *ind, int N, int C, int H,
                     int W, int OH, int OW, int K, int S) {
  maxpool_forward_kernel<<<(N * C * OH * OW + 255) / 256, 256>>>(
      in, out, ind, N, C, H, W, OH, OW, K, S);
}
void maxpool_backward(const float *go, float *gi, const int *ind, int size) {
  maxpool_backward_kernel<<<(size + 255) / 256, 256>>>(go, gi, ind, size);
}
void conv2d_forward(const float *in, const float *w, const float *b, float *out,
                    int N, int IC, int OC, int H, int W, int OH, int OW, int K,
                    int S, int P) {
  conv2d_forward_kernel<<<(N * OC * OH * OW + 255) / 256, 256>>>(
      in, w, b, out, N, IC, OC, H, W, OH, OW, K, S, P);
}
void upsample_forward(const float *in, float *out, int N, int C, int H, int W,
                      int OH, int OW, int SF) {
  upsample_forward_kernel<<<(N * C * OH * OW + 255) / 256, 256>>>(
      in, out, N, C, H, W, OH, OW, SF);
}
float mse_loss_cuda(const float *p, const float *t, float *g, int size) {
  float *d_loss, h_loss;
  cudaMalloc(&d_loss, sizeof(float));
  cudaMemset(d_loss, 0, sizeof(float));
  mse_loss_kernel<<<(size + 255) / 256, 256>>>(p, t, g, d_loss, size);
  cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_loss);
  return h_loss;
}

void relu_forward(const float *in, float *out, size_t size) {
  int threads = 256;
  int blocks = (size + 255) / 256;
  relu_forward_kernel<<<blocks, threads>>>(in, out, size);
  // REMOVED: cudaDeviceSynchronize();  <-- ALLOW ASYNC EXECUTION
}

void relu_backward(const float *go, const float *in, float *gi, size_t size) {
  int threads = 256;
  int blocks = (size + 255) / 256;
  relu_backward_kernel<<<blocks, threads>>>(go, in, gi, size);
}

void linear_forward(const float *in, const float *w, const float *b, float *out,
                    int batch_size, int in_f, int out_f) {
  dim3 threads(16, 16);
  dim3 blocks((batch_size + 15) / 16, (out_f + 15) / 16);
  linear_forward_kernel<<<blocks, threads>>>(in, w, b, out, batch_size, in_f,
                                             out_f);
}

void linear_backward_dx(const float *go, const float *w, float *gi,
                        int batch_size, int in_f, int out_f) {
  dim3 threads(16, 16);
  dim3 blocks((batch_size + 15) / 16, (in_f + 15) / 16);
  linear_backward_dx_kernel<<<blocks, threads>>>(go, w, gi, batch_size, in_f,
                                                 out_f);
}

void linear_backward_dw_db(const float *go, const float *in, float *gw,
                           float *gb, int batch_size, int in_f, int out_f) {
  dim3 threads(16, 16);
  dim3 blocks((in_f + 15) / 16, (out_f + 15) / 16);
  linear_backward_dw_db_kernel<<<blocks, threads>>>(go, in, gw, gb, batch_size,
                                                    in_f, out_f);
}

void sgd_step(float *w, const float *grad, float lr, int size) {
  int threads = 256;
  int blocks = (size + 255) / 256;
  sgd_step_kernel<<<blocks, threads>>>(w, grad, lr, size);
}

void add_bias(float *out, const float *b, int batch_size, int out_f) {
  int total_size = batch_size * out_f;
  int threads = 256;
  int blocks = (total_size + 255) / 256;
  add_bias_kernel<<<blocks, threads>>>(out, b, total_size, out_f);
}

void elementwise_mul(const float *a, const float *b, float *out, size_t size) {
  int threads = 256;
  int blocks = (size + 255) / 256;
  elementwise_mul_kernel<<<blocks, threads>>>(a, b, out, size);
}

float cross_entropy_loss_cuda(const float *logits, const float *targets,
                              float *grad_output, int batch_size,
                              int num_classes) {
  // Use persistent workspace (No Malloc/Free)
  // TODO: this causes segfaults when computing loss...
  // float* d_loss = Context::instance().d_workspace;
  // cudaMemsetAsync(d_loss, 0, sizeof(float), 0); // Async set

  float *d_loss;
  cudaMalloc(&d_loss, sizeof(float));
  cudaMemset(d_loss, 0, sizeof(float));

  int threads = 256;
  int blocks = (batch_size + 255) / 256;
  cross_entropy_gradient_kernel<<<blocks, threads>>>(
      logits, targets, grad_output, d_loss, batch_size, num_classes);

  // Only sync here to get the loss value back for logging
  float h_loss;
  cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_loss);

  return h_loss;
}

float spatial_cross_entropy_loss_cuda(const float *logits, const float *targets,
                                      float *grad_output, int N, int C, int H,
                                      int W) {
  float *d_loss;
  cudaMalloc(&d_loss, sizeof(float));
  cudaMemset(d_loss, 0, sizeof(float));

  int total_pixels = N * H * W;
  int threads = 256;
  int blocks = (total_pixels + 255) / 256;
  spatial_cross_entropy_kernel<<<blocks, threads>>>(
      logits, targets, grad_output, d_loss, N, C, H, W);

  float h_loss;
  cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_loss);

  // Average loss over spatial dimensions? usually standard reduction is sum
  // over spatial, avg over batch. The kernel atomicAdds (pixel_loss / N). Sum
  // of (pixel_loss / N) = (Sum pixel_loss) / N. This assumes we want total
  // image loss. If we want per-pixel average, divide by (H*W). Standard
  // CrossEntropy in frameworks is usually mean-over-batch, sum-over-spatial (or
  // mean-over-all). Let's stick to sum-over-spatial, mean-over-batch to match
  // the scale.

  return h_loss;
}

void adam_step(float *w, const float *grad, float *m, float *v, float lr,
               float beta1, float beta2, float eps, float weight_decay,
               int step, size_t size) {
  int threads = 256;
  int blocks = (size + 255) / 256;
  adam_step_kernel<<<blocks, threads>>>(w, grad, m, v, lr, beta1, beta2, eps,
                                        weight_decay, step, size);
}

void sigmoid_forward(const float *in, float *out, size_t size) {
  int threads = 256;
  int blocks = (size + 255) / 256;
  sigmoid_forward_kernel<<<blocks, threads>>>(in, out, size);
}

void sigmoid_backward(const float *go, const float *out, float *gi,
                      size_t size) {
  int threads = 256;
  int blocks = (size + 255) / 256;
  sigmoid_backward_kernel<<<blocks, threads>>>(go, out, gi, size);
}

} // namespace cuda_kernels
#endif
