#include "../storage.hpp"
#include "cuda_backend.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA Error: ") +                   \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

namespace munet {

void *CUDABackend::allocate(size_t bytes) {
  void *ptr;
  CUDA_CHECK(cudaMalloc(&ptr, bytes));
  return ptr;
}

void CUDABackend::deallocate(void *ptr) { CUDA_CHECK(cudaFree(ptr)); }

void CUDABackend::memset(void *ptr, int value, size_t bytes) {
  CUDA_CHECK(cudaMemset(ptr, value, bytes));
}

void CUDABackend::copy(const void *src, void *dst, size_t bytes, Device src_dev,
                       Device dst_dev) {
  cudaMemcpyKind kind = cudaMemcpyDefault; // CUDA 4.0+ infers direction using
                                           // Unified Virtual Addressing
  CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
}

void CUDABackend::synchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }

void CUDABackend::all_reduce(Storage &buffer, size_t num_elements) {
  // Placeholder for NCCL multi-GPU implementation
}

// --- Kernels ---
__global__ void add_kernel(const float *a, const float *b, float *out,
                           size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    out[idx] = a[idx] + b[idx];
}

void CUDABackend::add(const Storage &a, const Storage &b, Storage &out,
                      size_t num_elements) {
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>((const float *)a.data(),
                                  (const float *)b.data(), (float *)out.data(),
                                  num_elements);
  CUDA_CHECK(cudaGetLastError());
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

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int K, int N, bool transA, bool transB) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      float a_val = transA ? A[k * M + row] : A[row * K + k];
      float b_val = transB ? B[col * K + k] : B[k * N + col];
      sum += a_val * b_val;
    }
    C[row * N + col] = sum;
  }
}

__global__ void sub_kernel(const float *a, const float *b, float *out,
                           size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    out[idx] = a[idx] - b[idx]; // Fixed
}

__global__ void mul_kernel(const float *a, const float *b, float *out,
                           size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    out[idx] = a[idx] * b[idx]; // Fixed
}

__global__ void update_kernel(float *w, const float *g, float lr, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    w[idx] -= lr * g[idx]; // Fixed
}

void CUDABackend::sub(const Storage &a, const Storage &b, Storage &out,
                      size_t num_elements) {
  int threads = 256, blocks = (num_elements + threads - 1) / threads;
  sub_kernel<<<blocks, threads>>>((const float *)a.data(),
                                  (const float *)b.data(), (float *)out.data(),
                                  num_elements);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::mul(const Storage &a, const Storage &b, Storage &out,
                      size_t num_elements) {
  int threads = 256, blocks = (num_elements + threads - 1) / threads;
  mul_kernel<<<blocks, threads>>>((const float *)a.data(),
                                  (const float *)b.data(), (float *)out.data(),
                                  num_elements);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::update(Storage &weight, const Storage &grad, float lr,
                         size_t num_elements) {
  int threads = 256, blocks = (num_elements + threads - 1) / threads;
  update_kernel<<<blocks, threads>>>(
      (float *)weight.data(), (const float *)grad.data(), lr, num_elements);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::matmul(const Storage &a, const Storage &b, Storage &out,
                         int M, int K, int N, bool transA, bool transB) {
  dim3 threads(16, 16);
  dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
  matmul_kernel<<<blocks, threads>>>(
      (const float *)a.data(), (const float *)b.data(), (float *)out.data(), M,
      K, N, transA, transB);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::relu(const Storage &in, Storage &out, size_t num_elements) {
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  relu_kernel<<<blocks, threads>>>((const float *)in.data(),
                                   (float *)out.data(), num_elements);
  CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::relu_backward(const Storage &grad_out, const Storage &input,
                                Storage &grad_in, size_t num_elements) {
  int threads = 256;
  int blocks = (num_elements + threads - 1) / threads;
  relu_backward_kernel<<<blocks, threads>>>(
      (const float *)grad_out.data(), (const float *)input.data(),
      (float *)grad_in.data(), num_elements);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace munet
