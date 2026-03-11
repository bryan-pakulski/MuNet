#include "../storage.hpp"
#include "cuda_backend.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA Error: ") +                   \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(std::string("cuBLAS Error: ") +                \
                               std::to_string(status));                        \
    }                                                                          \
  } while (0)

namespace munet {

static std::unordered_map<size_t, std::vector<void *>> free_blocks;
static std::unordered_map<void *, size_t> alloc_sizes;
static cublasHandle_t cublas_handle_ = nullptr;

CUDABackend::CUDABackend(int device_index) : device_index_(device_index) {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_index_ < 0 || device_index_ >= device_count) {
    throw std::runtime_error("Requested CUDA device index out of range: " +
                             std::to_string(device_index_) +
                             " (available: " + std::to_string(device_count) + ")");
  }

  CUDA_CHECK(cudaSetDevice(device_index_));
  if (!cublas_handle_)
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUDA_CHECK(cudaEventCreate((cudaEvent_t *)&start_event_));
  CUDA_CHECK(cudaEventCreate((cudaEvent_t *)&stop_event_));
}

CUDABackend::~CUDABackend() {
  cudaSetDevice(device_index_);
  cudaEventDestroy((cudaEvent_t)start_event_);
  cudaEventDestroy((cudaEvent_t)stop_event_);
}

// --- Kernels ---
__global__ void add_kernel(const float *a, const float *b, float *out,
                           size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    out[idx] = a[idx] + b[idx];
}

__global__ void concat_slice_kernel(float *src, float *dst, int outer_size,
                                    int src_dim_size, int dst_dim_size,
                                    int inner_size, int src_dim_offset,
                                    bool forward) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer_size * src_dim_size * inner_size;

  if (idx < total) {
    int i = idx % inner_size;
    int tmp = idx / inner_size;
    int d = tmp % src_dim_size;
    int o = tmp / src_dim_size;

    int src_idx = (o * src_dim_size + d) * inner_size + i;
    int dst_idx = (o * dst_dim_size + (src_dim_offset + d)) * inner_size + i;

    // If forward is true, copy from src to dst; otherwise, copy from dst to src
    if (forward) {
      dst[dst_idx] = src[src_idx];
    } else {
      src[src_idx] = dst[dst_idx];
    }
  }
}

__global__ void broadcast_row_kernel(const float *src, float *dst, int rows,
                                     int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx < total) {
    int col = idx % cols;
    dst[idx] = src[col];
  }
}

__global__ void relu_kernel(const float *in, float *out, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    out[idx] = in[idx] > 0 ? in[idx] : 0.0f;
}

__global__ void relu_backward_kernel(const float *grad_out, const float *in,
                                     float *grad_in, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    grad_in[idx] = in[idx] > 0 ? grad_out[idx] : 0.0f;
}

__global__ void sigmoid_kernel(const float *in, float *out, size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    out[i] = 1.0f / (1.0f + expf(-in[i]));
}

__global__ void sigmoid_backward_kernel(const float *go, const float *out,
                                        float *gi, size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float s = out[i];
    gi[i] = go[i] * s * (1.0f - s);
  }
}

__global__ void log_kernel(const float *in, float *out, size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    out[i] = logf(in[i]);
}

__global__ void sqrt_kernel(const float *in, float *out, size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    out[i] = sqrtf(in[i]);
}

__global__ void clip_kernel(const float *in, float *out, float min_value,
                            float max_value, size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float v = in[i];
    out[i] = fminf(max_value, fmaxf(min_value, v));
  }
}

__device__ inline float erf_approx_cuda(float x) {
  const float a = 0.147f;
  float sign = x < 0.0f ? -1.0f : 1.0f;
  float ax = fabsf(x);
  float x2 = ax * ax;
  float t = 1.0f + a * x2;
  float inside = 1.0f - expf(-x2 * (4.0f / 3.14159265358979323846f + a * x2) / t);
  return sign * sqrtf(fmaxf(0.0f, inside));
}

__global__ void erf_kernel(const float *in, float *out, size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    out[i] = erf_approx_cuda(in[i]);
}


__global__ void topk_kernel(const float *in, float *outv, float *outi,
                           int outer, int dim_size, int k, int largest,
                           int sorted_flag) {
  const int MAXK = 64;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= outer)
    return;
  if (k > MAXK) {
    for (int j = 0; j < k; ++j) {
      outv[row * k + j] = 0.0f;
      outi[row * k + j] = 0.0f;
    }
    return;
  }

  int sel[MAXK];
  float sel_v[MAXK];
  for (int j = 0; j < k; ++j) {
    sel[j] = -1;
    sel_v[j] = 0.0f;
  }

  const float *r = in + row * dim_size;
  for (int j = 0; j < k; ++j) {
    int best_i = -1;
    float best_v = largest ? -3.402823e38f : 3.402823e38f;
    for (int i = 0; i < dim_size; ++i) {
      bool used = false;
      for (int p = 0; p < j; ++p)
        if (sel[p] == i)
          used = true;
      if (used)
        continue;
      float v = r[i];
      if (largest) {
        if (v > best_v || (v == best_v && i < best_i)) {
          best_v = v;
          best_i = i;
        }
      } else {
        if (v < best_v || (v == best_v && i < best_i)) {
          best_v = v;
          best_i = i;
        }
      }
    }
    sel[j] = best_i;
    sel_v[j] = best_v;
  }

  if (!sorted_flag) {
    // already in selection order
  }

  for (int j = 0; j < k; ++j) {
    outv[row * k + j] = sel_v[j];
    outi[row * k + j] = (float)sel[j];
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

__global__ void mse_loss_kernel(const float *p, const float *t, float *out,
                                size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float diff = p[i] - t[i];
    atomicAdd(out, (diff * diff) / (float)N);
  }
}

__global__ void mse_loss_backward_kernel(const float *go, const float *p,
                                         const float *t, float *gi, size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    gi[i] = go[0] * 2.0f * (p[i] - t[i]) / (float)N;
  }
}

__global__ void cross_entropy_kernel(const float *logits, const float *targets,
                                     float *out_loss, int batch_size,
                                     int num_classes, int spatial) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // We parallelize over (Batch * Spatial) pixels
  if (idx < batch_size * spatial) {
    int b = idx / spatial;
    int s = idx % spatial;

    // Offsets for NCHW: b * (C*S) + c * S + s
    int stride = spatial;
    int base_offset = b * (num_classes * spatial) + s;

    float max_val = -1e30f;
    for (int c = 0; c < num_classes; ++c) {
      float val = logits[base_offset + c * stride];
      if (val > max_val)
        max_val = val;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      sum_exp += expf(logits[base_offset + c * stride] - max_val);
    }

    float loss = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      float p = expf(logits[base_offset + c * stride] - max_val) / sum_exp;
      float t = targets[base_offset + c * stride];
      if (t > 0.0f) {
        loss -= t * logf(p + 1e-9f);
      }
    }
    // Average over batch size
    atomicAdd(out_loss, loss / (float)batch_size);
  }
}

__global__ void cross_entropy_backward_kernel(const float *go,
                                              const float *logits,
                                              const float *targets, float *gi,
                                              int batch_size, int num_classes,
                                              int spatial) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size * spatial) {
    int b = idx / spatial;
    int s = idx % spatial;

    int stride = spatial;
    int base_offset = b * (num_classes * spatial) + s;

    float max_val = -1e30f;
    for (int c = 0; c < num_classes; ++c) {
      float val = logits[base_offset + c * stride];
      if (val > max_val)
        max_val = val;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      sum_exp += expf(logits[base_offset + c * stride] - max_val);
    }

    float go_val = go[0];
    for (int c = 0; c < num_classes; ++c) {
      float p = expf(logits[base_offset + c * stride] - max_val) / sum_exp;
      float t = targets[base_offset + c * stride];
      gi[base_offset + c * stride] = go_val * (p - t) / (float)batch_size;
    }
  }
}

__global__ void sub_kernel(const float *a, const float *b, float *out,
                           size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    out[idx] = a[idx] - b[idx];
}

__global__ void mul_kernel(const float *a, const float *b, float *out,
                           size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    out[idx] = a[idx] * b[idx];
}

__global__ void div_kernel(const float *a, const float *b, float *out,
                           size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    out[idx] = a[idx] / b[idx];
}

__global__ void update_kernel(float *w, const float *g, float lr, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    w[idx] -= lr * g[idx];
}

// Simple PCG Hash for randomness
__device__ uint32_t pcg_hash(uint32_t input) {
  uint32_t state = input * 747796405u + 2891336453u;
  uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

__global__ void uniform_kernel(float *out, float low, float range, size_t N,
                               uint32_t seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    uint32_t r = pcg_hash(idx + seed);
    // Normalize to [0, 1]
    float r_norm = (float)r / 4294967295.0f;
    out[idx] = low + r_norm * range;
  }
}

__global__ void sum_kernel(const float *in, float *out, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    atomicAdd(out, in[idx]);
  }
}

__global__ void conv2d_kernel(const float *__restrict__ in,
                              const float *__restrict__ weight,
                              const float *__restrict__ bias,
                              float *__restrict__ out, int B, int iC, int iH,
                              int iW, int oC, int kH, int kW, int s, int p,
                              int oH, int oW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * oC * oH * oW;
  if (idx < total) {
    int ow = idx % oW;
    int tmp = idx / oW;
    int oh = tmp % oH;
    tmp /= oH;
    int oc = tmp % oC;
    int b = tmp / oC;

    float sum = (bias) ? bias[oc] : 0.0f;

    for (int ic = 0; ic < iC; ++ic) {
      for (int kh = 0; kh < kH; ++kh) {
        for (int kw = 0; kw < kW; ++kw) {
          int ih = oh * s - p + kh;
          int iw = ow * s - p + kw;
          if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
            int i_idx = (b * iC + ic) * (iH * iW) + ih * iW + iw;
            int w_idx = (oc * iC + ic) * (kH * kW) + kh * kW + kw;
            sum += in[i_idx] * weight[w_idx];
          }
        }
      }
    }
    out[idx] = sum;
  }
}

__global__ void conv2d_grad_input_kernel(const float *__restrict__ grad_out,
                                         const float *__restrict__ weight,
                                         float *__restrict__ grad_in, int B,
                                         int iC, int iH, int iW, int oC, int kH,
                                         int kW, int s, int p, int oH, int oW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * iC * iH * iW;
  if (idx < total) {
    int iw = idx % iW;
    int tmp = idx / iW;
    int ih = tmp % iH;
    tmp /= iH;
    int ic = tmp % iC;
    int b = tmp / iC;

    float d_in = 0.0f;
    for (int oc = 0; oc < oC; ++oc) {
      for (int kh = 0; kh < kH; ++kh) {
        for (int kw = 0; kw < kW; ++kw) {
          int num_h = ih + p - kh;
          int num_w = iw + p - kw;
          if (num_h >= 0 && num_w >= 0 && num_h % s == 0 && num_w % s == 0) {
            int oh = num_h / s;
            int ow = num_w / s;
            if (oh >= 0 && oh < oH && ow >= 0 && ow < oW) {
              int go_idx = (b * oC + oc) * (oH * oW) + oh * oW + ow;
              int w_idx = (oc * iC + ic) * (kH * kW) + kh * kW + kw;
              d_in += grad_out[go_idx] * weight[w_idx];
            }
          }
        }
      }
    }
    grad_in[idx] = d_in;
  }
}

__global__ void conv2d_grad_weight_kernel(const float *__restrict__ grad_out,
                                          const float *__restrict__ in,
                                          float *__restrict__ grad_w, int B,
                                          int iC, int iH, int iW, int oC,
                                          int kH, int kW, int s, int p, int oH,
                                          int oW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = oC * iC * kH * kW;
  if (idx < total) {
    int kw = idx % kW;
    int tmp = idx / kW;
    int kh = tmp % kH;
    tmp /= kH;
    int ic = tmp % iC;
    int oc = tmp / iC;

    float dw = 0.0f;
    for (int b = 0; b < B; ++b) {
      for (int oh = 0; oh < oH; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
          int ih = oh * s - p + kh;
          int iw = ow * s - p + kw;
          if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
            int go_idx = (b * oC + oc) * (oH * oW) + oh * oW + ow;
            int in_idx = (b * iC + ic) * (iH * iW) + ih * iW + iw;
            dw += grad_out[go_idx] * in[in_idx];
          }
        }
      }
    }
    grad_w[idx] = dw;
  }
}

__global__ void conv2d_grad_bias_kernel(const float *__restrict__ grad_out,
                                        float *__restrict__ grad_b, int B,
                                        int oC, int oH, int oW) {
  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  if (oc < oC) {
    float db = 0.0f;
    for (int b = 0; b < B; ++b) {
      for (int i = 0; i < oH * oW; ++i) {
        int go_idx = (b * oC + oc) * (oH * oW) + i;
        db += grad_out[go_idx];
      }
    }
    grad_b[oc] = db;
  }
}

__global__ void maxpool2d_kernel(const float *__restrict__ in,
                                 float *__restrict__ out, int B, int C, int iH,
                                 int iW, int k, int s, int p, int oH, int oW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * C * oH * oW;
  if (idx < total) {
    int ow = idx % oW;
    int tmp = idx / oW;
    int oh = tmp % oH;
    tmp /= oH;
    int c = tmp % C;
    int b = tmp / C;

    float max_val = -1e37f;
    for (int kh = 0; kh < k; ++kh) {
      for (int kw = 0; kw < k; ++kw) {
        int ih = oh * s - p + kh;
        int iw = ow * s - p + kw;
        if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
          float val = in[(b * C + c) * (iH * iW) + ih * iW + iw];
          if (val > max_val)
            max_val = val;
        }
      }
    }
    out[idx] = max_val;
  }
}

__global__ void maxpool2d_backward_kernel(const float *__restrict__ grad_out,
                                          const float *__restrict__ in,
                                          float *__restrict__ grad_in, int B,
                                          int C, int iH, int iW, int k, int s,
                                          int p, int oH, int oW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * C * oH * oW;
  if (idx < total) {
    int ow = idx % oW;
    int tmp = idx / oW;
    int oh = tmp % oH;
    tmp /= oH;
    int c = tmp % C;
    int b = tmp / C;

    float go = grad_out[idx];
    float max_val = -1e37f;
    int max_idx = -1;

    for (int kh = 0; kh < k; ++kh) {
      for (int kw = 0; kw < k; ++kw) {
        int ih = oh * s - p + kh;
        int iw = ow * s - p + kw;
        if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
          int in_idx = (b * C + c) * (iH * iW) + ih * iW + iw;
          float val = in[in_idx];
          if (val > max_val) {
            max_val = val;
            max_idx = in_idx;
          }
        }
      }
    }
    if (max_idx != -1)
      atomicAdd(&grad_in[max_idx], go);
  }
}


__device__ inline float grid_src_coord(float g, int size, int align_corners) {
  if (align_corners) {
    if (size <= 1)
      return 0.0f;
    return ((g + 1.0f) * 0.5f) * (float)(size - 1);
  }
  return ((g + 1.0f) * (float)size - 1.0f) * 0.5f;
}

__device__ inline float sample_zero(const float *in, int B, int C, int H,
                                    int W, int b, int c, int y, int x) {
  if (x < 0 || x >= W || y < 0 || y >= H)
    return 0.0f;
  size_t off = (((size_t)b * C + c) * H + y) * W + x;
  return in[off];
}

__global__ void grid_sample_kernel(const float *in, const float *grid,
                                   float *out, int B, int C, int iH, int iW,
                                   int oH, int oW, int mode,
                                   int align_corners) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t N = (size_t)B * C * oH * oW;
  if (idx >= N)
    return;

  int ox = idx % oW;
  size_t t = idx / oW;
  int oy = t % oH;
  t /= oH;
  int c = t % C;
  int b = t / C;

  size_t goff = (((size_t)b * oH + oy) * oW + ox) * 2;
  float gx = grid[goff + 0];
  float gy = grid[goff + 1];
  float sx = grid_src_coord(gx, iW, align_corners);
  float sy = grid_src_coord(gy, iH, align_corners);

  float outv = 0.0f;
  if (mode == 1) {
    int nx = (int)nearbyintf(sx);
    int ny = (int)nearbyintf(sy);
    outv = sample_zero(in, B, C, iH, iW, b, c, ny, nx);
  } else {
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float wx1 = sx - (float)x0;
    float wy1 = sy - (float)y0;
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;
    outv = sample_zero(in, B, C, iH, iW, b, c, y0, x0) * wx0 * wy0 +
           sample_zero(in, B, C, iH, iW, b, c, y0, x1) * wx1 * wy0 +
           sample_zero(in, B, C, iH, iW, b, c, y1, x0) * wx0 * wy1 +
           sample_zero(in, B, C, iH, iW, b, c, y1, x1) * wx1 * wy1;
  }
  out[idx] = outv;
}

__global__ void upsample2d_kernel(const float *__restrict__ in,
                                  float *__restrict__ out, int B, int C, int iH,
                                  int iW, int scale) {
  int oH = iH * scale;
  int oW = iW * scale;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * C * oH * oW;
  if (idx < total) {
    int ow = idx % oW;
    int tmp = idx / oW;
    int oh = tmp % oH;
    tmp /= oH;
    int c = tmp % C;
    int b = tmp / C;
    out[idx] = in[(b * C + c) * (iH * iW) + (oh / scale) * iW + (ow / scale)];
  }
}

__global__ void upsample2d_backward_kernel(const float *__restrict__ grad_out,
                                           float *__restrict__ grad_in, int B,
                                           int C, int iH, int iW, int scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * C * iH * iW;
  if (idx < total) {
    int iw = idx % iW;
    int tmp = idx / iW;
    int ih = tmp % iH;
    tmp /= iH;
    int c = tmp % C;
    int b = tmp / C;

    float sum = 0.0f;
    int oH = iH * scale;
    int oW = iW * scale;
    int oh_start = ih * scale;
    int ow_start = iw * scale;

    for (int y = 0; y < scale; ++y) {
      for (int x = 0; x < scale; ++x) {
        sum += grad_out[(b * C + c) * (oH * oW) + (oh_start + y) * oW +
                        (ow_start + x)];
      }
    }
    grad_in[idx] = sum;
  }
}

// --- BN Kernels ---
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
                                       float *run_mean, float *run_var, int M,
                                       int C, float momentum) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) {
    float m_val = mean[c];
    float v = var[c];
    if (v < 0.0f)
      v = 0.0f;
    run_mean[c] = (1.0f - momentum) * run_mean[c] + momentum * m_val;
    run_var[c] = (1.0f - momentum) * run_var[c] +
                 momentum * (v * M / (M > 1 ? M - 1 : 1));
  }
}

__global__ void bn_forward_kernel(const float *x, const float *g,
                                  const float *b, const float *m,
                                  const float *v, float *y, int B, int C,
                                  int Spatial, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * C * Spatial;
  if (idx < total) {
    int tmp = idx / Spatial;
    int c = tmp % C;
    float mean = m[c];
    float var = v[c];
    float inv_std = rsqrtf(var + eps);
    y[idx] = g[c] * (x[idx] - mean) * inv_std + b[c];
  }
}

__global__ void bn_bw_pass1_kernel(const float *go, const float *in,
                                   const float *mean, const float *var,
                                   float *sum_go_xhat, float *sum_go, int N,
                                   int C, int H, int W, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * C * H * W) {
    int c = (idx / (W * H)) % C;
    float go_val = go[idx];
    float inv_std = rsqrtf(var[c] + eps);
    float xhat = (in[idx] - mean[c]) * inv_std;

    atomicAdd(&sum_go[c], go_val);
    atomicAdd(&sum_go_xhat[c], go_val * xhat);
  }
}

__global__ void bn_bw_pass2_kernel(const float *go, const float *in, float *gi,
                                   const float *mean, const float *var,
                                   const float *weight,
                                   const float *sum_go_xhat,
                                   const float *sum_go, int N, int C, int H,
                                   int W, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * C * H * W) {
    int c = (idx / (W * H)) % C;
    int M = N * H * W;
    float go_val = go[idx];
    float inv_std = rsqrtf(var[c] + eps);
    float xhat = (in[idx] - mean[c]) * inv_std;

    float dx = (weight[c] * inv_std / M) *
               (M * go_val - sum_go[c] - xhat * sum_go_xhat[c]);
    gi[idx] = dx;
  }
}

__global__ void adam_step_kernel(float *p, const float *g, float *m, float *v,
                                 float lr, float beta1, float beta2, float eps,
                                 int step, size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float grad = g[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
    v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;

    float m_hat = m[i] / (1.0f - powf(beta1, (float)step));
    float v_hat = v[i] / (1.0f - powf(beta2, (float)step));

    p[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
  }
}

__device__ __forceinline__ size_t get_offset(size_t idx, int ndim,
                                             const int *shape,
                                             const int *out_strides,
                                             const int *strides) {
  size_t offset = 0;

  for (int i = 0; i < ndim; ++i) {
    int coord = (idx / out_strides[i]) % shape[i];
    offset += coord * strides[i];
  }

  return offset;
}

__global__ void add_broadcast_kernel(const float *a, const float *b, float *out,
                                     size_t N, GPUBroadcastInfo info) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  size_t off_a =
      get_offset(idx, info.ndim, info.shape, info.out_strides, info.strides_a);

  size_t off_b =
      get_offset(idx, info.ndim, info.shape, info.out_strides, info.strides_b);

  out[idx] = a[off_a] + b[off_b];
}

__global__ void sub_broadcast_kernel(const float *a, const float *b, float *out,
                                     size_t N, GPUBroadcastInfo info) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  size_t off_a =
      get_offset(idx, info.ndim, info.shape, info.out_strides, info.strides_a);

  size_t off_b =
      get_offset(idx, info.ndim, info.shape, info.out_strides, info.strides_b);

  out[idx] = a[off_a] - b[off_b];
}

__global__ void mul_broadcast_kernel(const float *a, const float *b, float *out,
                                     size_t N, GPUBroadcastInfo info) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  size_t off_a =
      get_offset(idx, info.ndim, info.shape, info.out_strides, info.strides_a);

  size_t off_b =
      get_offset(idx, info.ndim, info.shape, info.out_strides, info.strides_b);

  out[idx] = a[off_a] * b[off_b];
}

__global__ void div_broadcast_kernel(const float *a, const float *b, float *out,
                                     size_t N, GPUBroadcastInfo info) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  size_t off_a =
      get_offset(idx, info.ndim, info.shape, info.out_strides, info.strides_a);

  size_t off_b =
      get_offset(idx, info.ndim, info.shape, info.out_strides, info.strides_b);

  out[idx] = a[off_a] / b[off_b];
}

__global__ void sum_to_shape_kernel(const float *in, float *out, int ndim,
                                    int out_ndim, const int *in_shape,
                                    const int *out_shape,
                                    const int *out_strides, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  size_t out_off = 0;
  size_t curr = i;

  for (int d = ndim - 1; d >= 0; --d) {

    int coord = curr % in_shape[d];
    curr /= in_shape[d];

    int out_d_idx = d - (ndim - out_ndim);

    if (out_d_idx >= 0 && out_shape[out_d_idx] != 1) {
      out_off += coord * out_strides[out_d_idx];
    }
  }

  atomicAdd(&out[out_off], in[i]);
}


__global__ void gather_elements_kernel(const float *data, const float *indices,
                                       float *out, int ndim, const int *shape,
                                       const int *strides, int axis, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  int coord[6] = {0, 0, 0, 0, 0, 0};
  size_t t = i;
  for (int d = 0; d < ndim; ++d) {
    coord[d] = (int)(t / (size_t)strides[d]);
    t %= (size_t)strides[d];
  }

  int g = (int)llrintf(indices[i]);
  int dim = shape[axis];
  if (g < 0)
    g += dim;
  if (g < 0 || g >= dim) {
    out[i] = 0.0f;
    return;
  }

  coord[axis] = g;
  size_t src = 0;
  for (int d = 0; d < ndim; ++d)
    src += (size_t)coord[d] * (size_t)strides[d];
  out[i] = data[src];
}

void *CUDABackend::allocate(size_t bytes) {
  cudaSetDevice(device_index_);
  cudaEventRecord((cudaEvent_t)start_event_);
  if (!free_blocks[bytes].empty()) {
    void *ptr = free_blocks[bytes].back();
    free_blocks[bytes].pop_back();
    return ptr;
  }
  void *ptr;
  CUDA_CHECK(cudaMalloc(&ptr, bytes));
  alloc_sizes[ptr] = bytes;
  cudaEventRecord((cudaEvent_t)stop_event_);
  return ptr;
}

void CUDABackend::deallocate(void *ptr) {
  cudaSetDevice(device_index_);
  cudaEventRecord((cudaEvent_t)start_event_);
  if (alloc_sizes.count(ptr)) {
    free_blocks[alloc_sizes[ptr]].push_back(ptr);
  } else {
    CUDA_CHECK(cudaFree(ptr));
  }
  cudaEventRecord((cudaEvent_t)stop_event_);
}

void CUDABackend::memset(void *ptr, int value, size_t bytes) {
  cudaSetDevice(device_index_);
  cudaEventRecord((cudaEvent_t)start_event_);
  CUDA_CHECK(cudaMemset(ptr, value, bytes));
  cudaEventRecord((cudaEvent_t)stop_event_);
}

void CUDABackend::copy(const void *src, void *dst, size_t bytes, Device src_dev,
                       Device dst_dev) {
  cudaSetDevice(device_index_);
  cudaEventRecord((cudaEvent_t)start_event_);
  cudaMemcpyKind kind = cudaMemcpyDefault;
  CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
  cudaEventRecord((cudaEvent_t)stop_event_);
}

void CUDABackend::synchronize() {
  cudaSetDevice(device_index_);
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = 0;
  // This should always succeed under the assumption that every op records
  // events
  cudaError_t err = cudaEventElapsedTime(&ms, (cudaEvent_t)start_event_,
                                         (cudaEvent_t)stop_event_);
  if (err == cudaSuccess) {
    last_kernel_us_ = (double)ms * 1000.0;
  } else {
    last_kernel_us_ = 0.0;
  }
}

// TODO: IMPL
void CUDABackend::all_reduce(Storage &buffer, size_t num_elements) {}

void CUDABackend::add(const Storage &a, const Storage &b, Storage &out,
                      const BroadcastInfo &info) {
  size_t total = numel(info.out_shape);
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);

  if (info.strides_a == default_strides(info.out_shape) &&
      info.strides_b == default_strides(info.out_shape)) {
    add_kernel<<<blocks, threads>>>((const float *)a.data(),
                                    (const float *)b.data(),
                                    (float *)out.data(), total);
  } else {
    add_broadcast_kernel<<<blocks, threads>>>(
        (const float *)a.data(), (const float *)b.data(), (float *)out.data(),
        total, to_gpu_info(info));
  }
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::sub(const Storage &a, const Storage &b, Storage &out,
                      const BroadcastInfo &info) {
  size_t total = numel(info.out_shape);
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);

  if (info.strides_a == default_strides(info.out_shape) &&
      info.strides_b == default_strides(info.out_shape)) {
    sub_kernel<<<blocks, threads>>>((const float *)a.data(),
                                    (const float *)b.data(),
                                    (float *)out.data(), total);
  } else {
    sub_broadcast_kernel<<<blocks, threads>>>(
        (const float *)a.data(), (const float *)b.data(), (float *)out.data(),
        total, to_gpu_info(info));
  }

  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::mul(const Storage &a, const Storage &b, Storage &out,
                      const BroadcastInfo &info) {
  size_t total = numel(info.out_shape);
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);

  if (info.strides_a == default_strides(info.out_shape) &&
      info.strides_b == default_strides(info.out_shape)) {
    // Fast path: simple elementwise
    mul_kernel<<<blocks, threads>>>((const float *)a.data(),
                                    (const float *)b.data(),
                                    (float *)out.data(), total);
  } else {
    // Slow path: broadcasting
    mul_broadcast_kernel<<<blocks, threads>>>(
        (const float *)a.data(), (const float *)b.data(), (float *)out.data(),
        total, to_gpu_info(info));
  }
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::div(const Storage &a, const Storage &b, Storage &out,
                      const BroadcastInfo &info) {
  size_t total = numel(info.out_shape);
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);

  if (info.strides_a == default_strides(info.out_shape) &&
      info.strides_b == default_strides(info.out_shape)) {
    div_kernel<<<blocks, threads>>>((const float *)a.data(),
                                    (const float *)b.data(),
                                    (float *)out.data(), total);
  } else {
    div_broadcast_kernel<<<blocks, threads>>>(
        (const float *)a.data(), (const float *)b.data(), (float *)out.data(),
        total, to_gpu_info(info));
  }
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::fill_uniform(Storage &out, float low, float high,
                               size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  static uint32_t seed_counter = 0;
  uint32_t seed = (uint32_t)time(NULL) + (seed_counter++ * 1337);
  cudaEventRecord((cudaEvent_t)start_event_);
  uniform_kernel<<<blocks, threads>>>((float *)out.data(), low, high - low,
                                      num_elements, seed);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::sum(const Storage &in, Storage &out, size_t num_elements) {
  cudaSetDevice(device_index_);
  CUDA_CHECK(cudaMemset(out.data(), 0, sizeof(float)));
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  sum_kernel<<<blocks, threads>>>((const float *)in.data(), (float *)out.data(),
                                  num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::update(Storage &weight, const Storage &grad, float lr,
                         size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256, blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  update_kernel<<<blocks, threads>>>(
      (float *)weight.data(), (const float *)grad.data(), lr, num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::matmul(const Storage &a, const Storage &b, Storage &out,
                         int M, int K, int N, bool transA, bool transB) {
  cudaSetDevice(device_index_);
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasOperation_t cuTransA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = transB ? K : N;
  int ldb = transA ? M : K;
  int ldc = N;

  cudaEventRecord((cudaEvent_t)start_event_);
  cublasStatus_t status =
      cublasSgemm(cublas_handle_, cuTransA, cuTransB, N, M, K, &alpha,
                  (const float *)b.data(), lda, (const float *)a.data(), ldb,
                  &beta, (float *)out.data(), ldc);

  cudaEventRecord((cudaEvent_t)stop_event_);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS SGEMM failed");
  }
}


void CUDABackend::batched_matmul(const Storage &a, const Storage &b, Storage &out,
                                 int B, int M, int K, int N, bool transA,
                                 bool transB) {
  cudaSetDevice(device_index_);
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasOperation_t cuTransA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = transB ? K : N;
  int ldb = transA ? M : K;
  int ldc = N;
  long long strideB = (long long)K * (long long)N;
  long long strideA = (long long)M * (long long)K;
  long long strideC = (long long)M * (long long)N;

  cudaEventRecord((cudaEvent_t)start_event_);
  cublasStatus_t status = cublasSgemmStridedBatched(
      cublas_handle_, cuTransA, cuTransB, N, M, K, &alpha,
      (const float *)b.data(), lda, strideB, (const float *)a.data(), ldb,
      strideA, &beta, (float *)out.data(), ldc, strideC, B);
  cudaEventRecord((cudaEvent_t)stop_event_);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS SGEMM strided batched failed");
  }
}

void CUDABackend::relu(const Storage &in, Storage &out, size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  relu_kernel<<<blocks, threads>>>((const float *)in.data(),
                                   (float *)out.data(), num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::relu_backward(const Storage &grad_out, const Storage &input,
                                Storage &grad_in, size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  relu_backward_kernel<<<blocks, threads>>>(
      (const float *)grad_out.data(), (const float *)input.data(),
      (float *)grad_in.data(), num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}


void CUDABackend::grid_sample(const Storage &in, const Storage &grid,
                              Storage &out, int B, int C, int iH, int iW,
                              int oH, int oW, int mode,
                              bool align_corners) {
  cudaSetDevice(device_index_);
  size_t total = (size_t)B * C * oH * oW;
  int blocks = (total + 255) / 256;
  cudaEventRecord((cudaEvent_t)start_event_);
  grid_sample_kernel<<<blocks, 256>>>((const float *)in.data(),
                                      (const float *)grid.data(),
                                      (float *)out.data(), B, C, iH, iW, oH,
                                      oW, mode, align_corners ? 1 : 0);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::batch_norm(const Storage &in, const Storage &scale,
                             const Storage &bias, Storage &running_mean,
                             Storage &running_var, Storage &save_mean,
                             Storage &save_var, Storage &out, int B, int C,
                             int H, int W, float momentum, float eps,
                             bool training) {
  cudaSetDevice(device_index_);
  int Spatial = H * W;
  int total = B * C * Spatial;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  cudaEventRecord((cudaEvent_t)start_event_);
  if (training) {
    CUDA_CHECK(cudaMemset(save_mean.data(), 0, C * sizeof(float)));
    CUDA_CHECK(cudaMemset(save_var.data(), 0, C * sizeof(float)));

    bn_mean_kernel<<<blocks, threads>>>((const float *)in.data(),
                                        (float *)save_mean.data(), B, C, H, W);
    bn_var_kernel<<<blocks, threads>>>((const float *)in.data(),
                                       (const float *)save_mean.data(),
                                       (float *)save_var.data(), B, C, H, W);

    int c_blocks = (C + threads - 1) / threads;
    bn_update_stats_kernel<<<c_blocks, threads>>>(
        (const float *)save_mean.data(), (const float *)save_var.data(),
        (float *)running_mean.data(), (float *)running_var.data(), B * Spatial,
        C, momentum);
  }

  // Normalize
  const float *m_ptr = training ? (const float *)save_mean.data()
                                : (const float *)running_mean.data();
  const float *v_ptr = training ? (const float *)save_var.data()
                                : (const float *)running_var.data();
  bn_forward_kernel<<<blocks, threads>>>(
      (const float *)in.data(), (const float *)scale.data(),
      (const float *)bias.data(), m_ptr, v_ptr, (float *)out.data(), B, C,
      Spatial, eps);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::batch_norm_backward(const Storage &grad_out,
                                      const Storage &in, const Storage &scale,
                                      const Storage &save_mean,
                                      const Storage &save_var, Storage &grad_in,
                                      Storage &grad_scale, Storage &grad_bias,
                                      int B, int C, int H, int W, float eps) {

  cudaSetDevice(device_index_);
  int total = B * C * H * W;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  CUDA_CHECK(cudaMemset(grad_scale.data(), 0, C * sizeof(float)));
  CUDA_CHECK(cudaMemset(grad_bias.data(), 0, C * sizeof(float)));

  cudaEventRecord((cudaEvent_t)start_event_);
  bn_bw_pass1_kernel<<<blocks, threads>>>(
      (const float *)grad_out.data(), (const float *)in.data(),
      (const float *)save_mean.data(), (const float *)save_var.data(),
      (float *)grad_scale.data(), (float *)grad_bias.data(), B, C, H, W, eps);

  bn_bw_pass2_kernel<<<blocks, threads>>>(
      (const float *)grad_out.data(), (const float *)in.data(),
      (float *)grad_in.data(), (const float *)save_mean.data(),
      (const float *)save_var.data(), (const float *)scale.data(),
      (const float *)grad_scale.data(), (const float *)grad_bias.data(), B, C,
      H, W, eps);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::conv2d(const Storage &in, const Storage &weight,
                         const Storage *bias, Storage &out, int B, int iC,
                         int iH, int iW, int oC, int kH, int kW, int s, int p) {
  cudaSetDevice(device_index_);
  int oH = (iH + 2 * p - kH) / s + 1;
  int oW = (iW + 2 * p - kW) / s + 1;
  size_t total = B * oC * oH * oW;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  conv2d_kernel<<<blocks, threads>>>(
      (const float *)in.data(), (const float *)weight.data(),
      bias ? (const float *)bias->data() : nullptr, (float *)out.data(), B, iC,
      iH, iW, oC, kH, kW, s, p, oH, oW);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::conv2d_backward(const Storage &grad_out, const Storage &in,
                                  const Storage &weight, Storage &grad_in,
                                  Storage &grad_w, Storage *grad_b, int B,
                                  int iC, int iH, int iW, int oC, int kH,
                                  int kW, int s, int p) {
  cudaSetDevice(device_index_);
  int oH = (iH + 2 * p - kH) / s + 1;
  int oW = (iW + 2 * p - kW) / s + 1;
  cudaEventRecord((cudaEvent_t)start_event_);
  {
    size_t total = B * iC * iH * iW;
    int blocks = (total + 255) / 256;
    conv2d_grad_input_kernel<<<blocks, 256>>>(
        (const float *)grad_out.data(), (const float *)weight.data(),
        (float *)grad_in.data(), B, iC, iH, iW, oC, kH, kW, s, p, oH, oW);
  }
  {
    size_t total = oC * iC * kH * kW;
    int blocks = (total + 255) / 256;
    conv2d_grad_weight_kernel<<<blocks, 256>>>(
        (const float *)grad_out.data(), (const float *)in.data(),
        (float *)grad_w.data(), B, iC, iH, iW, oC, kH, kW, s, p, oH, oW);
  }
  if (grad_b) {
    int blocks = (oC + 255) / 256;
    conv2d_grad_bias_kernel<<<blocks, 256>>>(
        (const float *)grad_out.data(), (float *)grad_b->data(), B, oC, oH, oW);
  }
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::max_pool2d(const Storage &in, Storage &out, int B, int C,
                             int iH, int iW, int k, int s, int p) {
  cudaSetDevice(device_index_);
  int oH = (iH + 2 * p - k) / s + 1;
  int oW = (iW + 2 * p - k) / s + 1;
  size_t total = B * C * oH * oW;
  int blocks = (total + 255) / 256;
  cudaEventRecord((cudaEvent_t)start_event_);
  maxpool2d_kernel<<<blocks, 256>>>((const float *)in.data(),
                                    (float *)out.data(), B, C, iH, iW, k, s, p,
                                    oH, oW);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::max_pool2d_backward(const Storage &grad_out,
                                      const Storage &in, Storage &grad_in,
                                      int B, int C, int iH, int iW, int k,
                                      int s, int p) {
  cudaSetDevice(device_index_);
  int oH = (iH + 2 * p - k) / s + 1;
  int oW = (iW + 2 * p - k) / s + 1;
  size_t total = B * C * oH * oW;
  int blocks = (total + 255) / 256;
  cudaEventRecord((cudaEvent_t)start_event_);
  maxpool2d_backward_kernel<<<blocks, 256>>>(
      (const float *)grad_out.data(), (const float *)in.data(),
      (float *)grad_in.data(), B, C, iH, iW, k, s, p, oH, oW);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::upsample2d(const Storage &in, Storage &out, int B, int C,
                             int iH, int iW, int scale) {
  cudaSetDevice(device_index_);
  int oH = iH * scale;
  int oW = iW * scale;
  size_t total = B * C * oH * oW;
  int blocks = (total + 255) / 256;
  cudaEventRecord((cudaEvent_t)start_event_);
  upsample2d_kernel<<<blocks, 256>>>((const float *)in.data(),
                                     (float *)out.data(), B, C, iH, iW, scale);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::upsample2d_backward(const Storage &grad_out, Storage &grad_in,
                                      int B, int C, int iH, int iW, int scale) {
  cudaSetDevice(device_index_);
  size_t total = B * C * iH * iW;
  int blocks = (total + 255) / 256;
  cudaEventRecord((cudaEvent_t)start_event_);
  upsample2d_backward_kernel<<<blocks, 256>>>((const float *)grad_out.data(),
                                              (float *)grad_in.data(), B, C, iH,
                                              iW, scale);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}
void CUDABackend::sigmoid(const Storage &in, Storage &out,
                          size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  sigmoid_kernel<<<blocks, threads>>>((const float *)in.data(),
                                      (float *)out.data(), num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::sigmoid_backward(const Storage &grad_out, const Storage &out,
                                   Storage &grad_in, size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  sigmoid_backward_kernel<<<blocks, threads>>>(
      (const float *)grad_out.data(), (const float *)out.data(),
      (float *)grad_in.data(), num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::log(const Storage &in, Storage &out, size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  log_kernel<<<blocks, threads>>>((const float *)in.data(), (float *)out.data(),
                                  num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::sqrt(const Storage &in, Storage &out, size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  sqrt_kernel<<<blocks, threads>>>((const float *)in.data(),
                                   (float *)out.data(), num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::clip(const Storage &in, Storage &out, float min_value,
                       float max_value, size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  clip_kernel<<<blocks, threads>>>((const float *)in.data(), (float *)out.data(),
                                   min_value, max_value, num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::erf(const Storage &in, Storage &out, size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  erf_kernel<<<blocks, threads>>>((const float *)in.data(), (float *)out.data(),
                                  num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::softmax(const Storage &in, Storage &out, int batch_size,
                          int num_classes) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (batch_size + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  softmax_forward_kernel<<<blocks, threads>>>(
      (const float *)in.data(), (float *)out.data(), batch_size, num_classes);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::softmax_backward(const Storage &grad_out, const Storage &out,
                                   Storage &grad_in, int batch_size,
                                   int num_classes) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (batch_size + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  softmax_backward_kernel<<<blocks, threads>>>(
      (const float *)grad_out.data(), (const float *)out.data(),
      (float *)grad_in.data(), batch_size, num_classes);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::mse_loss(const Storage &pred, const Storage &target,
                           Storage &out_loss, size_t num_elements) {
  cudaSetDevice(device_index_);
  CUDA_CHECK(cudaMemset(out_loss.data(), 0, sizeof(float)));
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  mse_loss_kernel<<<blocks, threads>>>((const float *)pred.data(),
                                       (const float *)target.data(),
                                       (float *)out_loss.data(), num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::mse_loss_backward(const Storage &grad_out,
                                    const Storage &pred, const Storage &target,
                                    Storage &grad_in, size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  mse_loss_backward_kernel<<<blocks, threads>>>(
      (const float *)grad_out.data(), (const float *)pred.data(),
      (const float *)target.data(), (float *)grad_in.data(), num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}


void CUDABackend::topk(const Storage &in, Storage &out_values,
                       Storage &out_indices, int outer, int dim_size, int k,
                       bool largest, bool sorted_flag) {
  cudaSetDevice(device_index_);
  if (k <= 0 || k > dim_size)
    throw std::runtime_error("topk: invalid k");
  if (k > 64)
    throw std::runtime_error("topk: k>64 not supported yet on CUDA");
  int threads = 256;
  int blocks = (outer + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  topk_kernel<<<blocks, threads>>>((const float *)in.data(),
                                   (float *)out_values.data(),
                                   (float *)out_indices.data(), outer,
                                   dim_size, k, largest ? 1 : 0,
                                   sorted_flag ? 1 : 0);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::cross_entropy(const Storage &logits, const Storage &targets,
                                Storage &out_loss, int batch_size,
                                int num_classes, int spatial) {
  cudaSetDevice(device_index_);
  CUDA_CHECK(cudaMemset(out_loss.data(), 0, sizeof(float)));
  int total_pixels = batch_size * spatial;
  int threads = 256;
  int blocks = (total_pixels + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  cross_entropy_kernel<<<blocks, threads>>>(
      (const float *)logits.data(), (const float *)targets.data(),
      (float *)out_loss.data(), batch_size, num_classes, spatial);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::cross_entropy_backward(const Storage &grad_out,
                                         const Storage &logits,
                                         const Storage &targets,
                                         Storage &grad_in, int batch_size,
                                         int num_classes, int spatial) {
  cudaSetDevice(device_index_);
  int total_pixels = batch_size * spatial;
  int threads = 256;
  int blocks = (total_pixels + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  cross_entropy_backward_kernel<<<blocks, threads>>>(
      (const float *)grad_out.data(), (const float *)logits.data(),
      (const float *)targets.data(), (float *)grad_in.data(), batch_size,
      num_classes, spatial);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::concat(const std::vector<Storage *> &inputs, Storage &out,
                         int dim, const std::vector<Shape> &shapes) {
  cudaSetDevice(device_index_);
  int outer_size = 1;
  // Compute outer size (dimensions before the concatenation dimension)
  for (int i = 0; i < dim; ++i)
    outer_size *= shapes[0][i];

  int inner_size = 1;
  // Compute inner size (dimensions after the concatenation dimension)
  for (int i = dim + 1; i < (int)shapes[0].size(); ++i) {
    inner_size *= shapes[0][i];
  }

  int dst_dim_size = 0;
  // Compute the size of the concatenation dimension in the output tensor
  for (const auto &s : shapes)
    dst_dim_size += s[dim];

  int current_offset = 0;
  // Iterate through the inputs and concatenate
  cudaEventRecord((cudaEvent_t)start_event_);
  for (size_t j = 0; j < inputs.size(); ++j) {
    int src_dim_size = shapes[j][dim]; // Get the size of the current input
                                       // tensor along the concat dimension
    int total_elements = outer_size * src_dim_size * inner_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    concat_slice_kernel<<<blocks, threads>>>(
        (float *)inputs[j]->data(), (float *)out.data(), outer_size,
        src_dim_size, dst_dim_size, inner_size, current_offset, true);

    current_offset += src_dim_size;
  }
  cudaEventRecord((cudaEvent_t)stop_event_);

  // Check for errors after kernel execution
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::gather_elements(const Storage &data, const Storage &indices,
                                  Storage &out, const Shape &shape, int axis) {
  cudaSetDevice(device_index_);
  int ndim = (int)shape.size();
  if (ndim <= 0 || ndim > 6)
    throw std::runtime_error("gather_elements: ndim must be in [1,6]");
  int ax = axis < 0 ? axis + ndim : axis;
  if (ax < 0 || ax >= ndim)
    throw std::runtime_error("gather_elements: axis out of range");

  int h_shape[6] = {1, 1, 1, 1, 1, 1};
  int h_strides[6] = {0, 0, 0, 0, 0, 0};
  Strides st = default_strides(shape);
  for (int i = 0; i < ndim; ++i) {
    h_shape[i] = shape[i];
    h_strides[i] = st[i];
  }

  int *d_shape = nullptr;
  int *d_strides = nullptr;
  CUDA_CHECK(cudaMalloc(&d_shape, 6 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_strides, 6 * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_shape, h_shape, 6 * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_strides, h_strides, 6 * sizeof(int), cudaMemcpyHostToDevice));

  size_t N = numel(shape);
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  gather_elements_kernel<<<blocks, threads>>>((const float *)data.data(),
                                              (const float *)indices.data(),
                                              (float *)out.data(), ndim,
                                              d_shape, d_strides, ax, N);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());

  cudaFree(d_shape);
  cudaFree(d_strides);
}

void CUDABackend::concat_backward(const Storage &grad_out,
                                  std::vector<Storage *> &grad_inputs, int dim,
                                  const std::vector<Shape> &shapes) {
  cudaSetDevice(device_index_);
  int outer_size = 1;
  // Compute outer size (dimensions before the concatenation dimension)
  for (int i = 0; i < dim; ++i)
    outer_size *= shapes[0][i];

  int inner_size = 1;
  // Compute inner size (dimensions after the concatenation dimension)
  for (int i = dim + 1; i < (int)shapes[0].size(); ++i) {
    inner_size *= shapes[0][i];
  }

  int dst_dim_size = 0;
  // Compute the size of the concatenation dimension in the output tensor
  for (const auto &s : shapes)
    dst_dim_size += s[dim];

  int current_offset = 0;
  // Iterate through the gradient inputs and backpropagate
  cudaEventRecord((cudaEvent_t)start_event_);
  for (size_t j = 0; j < grad_inputs.size(); ++j) {
    int src_dim_size = shapes[j][dim]; // Get the size of the current input
                                       // tensor along the concat dimension
    int total_elements = outer_size * src_dim_size * inner_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    concat_slice_kernel<<<blocks, threads>>>(
        (float *)grad_inputs[j]->data(), (float *)grad_out.data(), outer_size,
        src_dim_size, dst_dim_size, inner_size, current_offset, false);

    current_offset += src_dim_size;
  }
  cudaEventRecord((cudaEvent_t)stop_event_);

  // Check for errors after kernel execution
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::broadcast_row(const Storage &src, Storage &dst, int rows,
                                int cols) {
  cudaSetDevice(device_index_);
  int total = rows * cols;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  broadcast_row_kernel<<<blocks, threads>>>((const float *)src.data(),
                                            (float *)dst.data(), rows, cols);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::adam_step(Storage &params, const Storage &grads,
                            Storage &exp_avg, Storage &exp_avg_sq, float lr,
                            float beta1, float beta2, float eps, int step,
                            size_t num_elements) {
  cudaSetDevice(device_index_);
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  cudaEventRecord((cudaEvent_t)start_event_);
  adam_step_kernel<<<blocks, threads>>>(
      (float *)params.data(), (const float *)grads.data(),
      (float *)exp_avg.data(), (float *)exp_avg_sq.data(), lr, beta1, beta2,
      eps, step, num_elements);
  cudaEventRecord((cudaEvent_t)stop_event_);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::sum_to_shape(const Storage &in, Storage &out,
                               const Shape &in_shape, const Shape &out_shape) {

  cudaSetDevice(device_index_);

  CUDA_CHECK(cudaMemset(out.data(), 0, out.size_bytes()));

  size_t N = numel(in_shape);

  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  int *d_in_shape;
  int *d_out_shape;
  int *d_out_strides;

  size_t in_bytes = in_shape.size() * sizeof(int);
  size_t out_bytes = out_shape.size() * sizeof(int);

  CUDA_CHECK(cudaMalloc(&d_in_shape, in_bytes));
  CUDA_CHECK(cudaMalloc(&d_out_shape, out_bytes));
  CUDA_CHECK(cudaMalloc(&d_out_strides, out_bytes));

  CUDA_CHECK(cudaMemcpy(d_in_shape, in_shape.data(), in_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_out_shape, out_shape.data(), out_bytes,
                        cudaMemcpyHostToDevice));

  Strides ost = default_strides(out_shape);

  CUDA_CHECK(
      cudaMemcpy(d_out_strides, ost.data(), out_bytes, cudaMemcpyHostToDevice));

  sum_to_shape_kernel<<<blocks, threads>>>(
      (const float *)in.data(), (float *)out.data(), (int)in_shape.size(),
      (int)out_shape.size(), d_in_shape, d_out_shape, d_out_strides, N);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_in_shape));
  CUDA_CHECK(cudaFree(d_out_shape));
  CUDA_CHECK(cudaFree(d_out_strides));
}

} // namespace munet
