#include "matmul.hpp"
#include "common.hpp"
#include "../backend.hpp"
#include "../../tensor.hpp"
#include <iostream>
#include "../op_dispatch.hpp"
#include <sstream>
#include <cmath>
#include <vector>

namespace munet {
namespace ops {

// Check if tensor dimensions are compatible for batched matmul
static bool is_batched_matmul_shape(const Shape& a_shape, const Shape& b_shape) {
  if (a_shape.size() < 2 || b_shape.size() < 2) return false;
  if (a_shape.size() == 2 && b_shape.size() == 2) return false;  // Regular 2D matmul
  return true;
}

// Helper to perform matmul on CPU with dtype conversion
// This is used when the target backend doesn't support the dtype (e.g., Float16)
static Tensor matmul_cpu_fallback(const Tensor& a, const Tensor& b) {
  Device cpu{DeviceType::CPU, 0};
  
  // Convert to CPU if needed
  Tensor a_cpu = a.device() == cpu ? a : a.to(cpu);
  Tensor b_cpu = b.device() == cpu ? b : b.to(cpu);
  
  // For Float16, convert to Float32 for computation since CPU may not support Float16 matmul
  DataType orig_dtype = a.dtype();
  if (orig_dtype == DataType::Float16) {
    a_cpu = a_cpu.to(DataType::Float32);
    b_cpu = b_cpu.to(DataType::Float32);
  }
  
  // Get shapes
  Shape a_shape = a_cpu.shape();
  Shape b_shape = b_cpu.shape();
  
  // Get CPU backend's BLAS capability
  Backend* backend = &a_cpu.impl_->backend();
  auto* blas = backend->blas_capability();
  if (!blas) {
    MUNET_ERROR << "matmul_cpu_fallback: CPU backend does not support BLAS operations" << std::endl;
    return Tensor();
  }
  
  // Perform 2D matmul on CPU
  int M = a_shape[0];
  int K = a_shape[1];
  int N = b_shape[1];
  
  Shape out_shape{static_cast<size_t>(M), static_cast<size_t>(N)};
  Tensor out(out_shape, cpu, a_cpu.dtype());
  
  blas->matmul(*a_cpu.impl_->storage, *b_cpu.impl_->storage, *out.impl_->storage,
               M, K, N, false, false);
  
  // Convert back to original dtype if we converted
  if (orig_dtype == DataType::Float16) {
    out = out.to(DataType::Float16);
  }
  
  // Move back to original device
  return out.to(a.device());
}

// Handle 2D matmul by delegating to regular matmul
Tensor matmul_2d(const Tensor &a, const Tensor &b) {
  const auto dispatch = resolve_dispatch(OpId::Matmul, a);

  // Use CPU fallback if needed (backend doesn't support this dtype)
  if (dispatch.use_cpu_fallback) {
    return matmul_cpu_fallback(a, b);
  }

  // Get shapes
  Shape a_shape = a.shape();
  Shape b_shape = b.shape();
  
  int M = a_shape[0];
  int K = a_shape[1];
  int N = b_shape[1];
  
  // Validate dimensions
  if (a_shape[1] != b_shape[0]) {
    MUNET_ERROR << "matmul: dimension mismatch: a.shape = [" 
                << a_shape[0] << ", " << a_shape[1] << "], b.shape = [" 
                << b_shape[0] << ", " << b_shape[1] << "]" << std::endl;
    return Tensor();
  }
  
  // Get BLAS capability
  Backend* backend = &a.impl_->backend();
  auto* blas = backend->blas_capability();
  if (!blas) {
    MUNET_ERROR << "matmul: Backend does not support BLAS operations" << std::endl;
    return Tensor();
  }
  
  // Create output tensor
  Shape out_shape{static_cast<size_t>(M), static_cast<size_t>(N)};
  Tensor out(out_shape, a.device(), a.dtype());
  
  // Call backend matmul
  blas->matmul(*a.impl_->storage, *b.impl_->storage, *out.impl_->storage,
               M, K, N, false, false);
  
  return out;
}

// Helper for batched matmul with CPU fallback
static Tensor batched_matmul_cpu_fallback(const Tensor &a, const Tensor &b, bool transA, bool transB) {
  Device cpu{DeviceType::CPU, 0};
  
  // Convert to CPU if needed
  Tensor a_cpu = a.device() == cpu ? a : a.to(cpu);
  Tensor b_cpu = b.device() == cpu ? b : b.to(cpu);
  
  // For Float16, convert to Float32 for computation
  DataType orig_dtype = a.dtype();
  if (orig_dtype == DataType::Float16) {
    a_cpu = a_cpu.to(DataType::Float32);
    b_cpu = b_cpu.to(DataType::Float32);
  }
  
  // Get shapes
  Shape a_shape = a_cpu.shape();
  Shape b_shape = b_cpu.shape();
  
  // Handle transposition
  int K_a = a_shape[a_shape.size() - 1];
  int M = a_shape[a_shape.size() - 2];
  int K_b = b_shape.size() >= 2 ? b_shape[b_shape.size() - 2] : 1;
  int N = b_shape[b_shape.size() - 1];
  
  if (transA) std::swap(M, K_a);
  if (transB) std::swap(K_b, N);
  
  int K = K_a;
  
  // Calculate batch size
  int batch = 1;
  size_t a_batch_dims = a_shape.size() - 2;
  size_t b_batch_dims = b_shape.size() - 2;
  
  if (a_batch_dims == 0) {
    // Convert to 2D matmul - call CPU BLAS directly
    Backend* backend = &a_cpu.impl_->backend();
    auto* blas = backend->blas_capability();
    if (!blas) {
      MUNET_ERROR << "batched_matmul_cpu_fallback: CPU backend does not support BLAS" << std::endl;
      return Tensor();
    }
    
    Shape out_shape{static_cast<size_t>(M), static_cast<size_t>(N)};
    Tensor out(out_shape, cpu, a_cpu.dtype());
    blas->matmul(*a_cpu.impl_->storage, *b_cpu.impl_->storage, *out.impl_->storage,
                 M, K, N, transA, transB);
    
    if (orig_dtype == DataType::Float16) {
      out = out.to(DataType::Float16);
    }
    return out.to(a.device());
  }
  
  // Calculate batch size from a
  for (size_t i = 0; i < a_batch_dims; ++i) {
    batch *= static_cast<int>(a_shape[i]);
  }
  
  // Calculate strides
  int64_t stride_a = M * K;
  int64_t stride_b = b_batch_dims > 0 ? K * N : 0;  // 0 means broadcast
  int64_t stride_out = M * N;
  
  // Create output tensor
  std::vector<size_t> out_shape_vec;
  for (size_t i = 0; i < a_batch_dims; ++i) {
    out_shape_vec.push_back(a_shape[i]);
  }
  out_shape_vec.push_back(static_cast<size_t>(M));
  out_shape_vec.push_back(static_cast<size_t>(N));
  Shape out_shape(out_shape_vec.begin(), out_shape_vec.end());
  
  Tensor out(out_shape, cpu, a_cpu.dtype());
  
  // Get CPU backend's BLAS
  Backend* backend = &a_cpu.impl_->backend();
  auto* blas = backend->blas_capability();
  if (!blas) {
    MUNET_ERROR << "batched_matmul_cpu_fallback: CPU backend does not support BLAS" << std::endl;
    return Tensor();
  }
  
  // Call backend batched_matmul
  blas->batched_matmul(*a_cpu.impl_->storage, *b_cpu.impl_->storage, *out.impl_->storage,
                       batch, M, K, N, transA, transB, stride_a, stride_b, stride_out);
  
  // Convert back to original dtype
  if (orig_dtype == DataType::Float16) {
    out = out.to(DataType::Float16);
  }
  
  return out.to(a.device());
}

Tensor batched_matmul_internal(const Tensor &a, const Tensor &b, bool transA, bool transB) {
  // Get shapes
  Shape a_shape = a.shape();
  Shape b_shape = b.shape();
  
  // Check for batched dimensions
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    MUNET_ERROR << "batched_matmul: tensors must have at least 2 dimensions" << std::endl;
    return Tensor();
  }
  
  // Resolve dispatch to check for dtype support
  const auto dispatch = resolve_dispatch(OpId::Matmul, a);
  
  // Use CPU fallback if needed
  if (dispatch.use_cpu_fallback) {
    return batched_matmul_cpu_fallback(a, b, transA, transB);
  }

  int K_a = a_shape[a_shape.size() - 1];
  int M = a_shape[a_shape.size() - 2];
  int K_b = b_shape.size() >= 2 ? b_shape[b_shape.size() - 2] : 1;
  int N = b_shape[b_shape.size() - 1];
  
  // Handle transposition
  if (transA) std::swap(M, K_a);
  if (transB) std::swap(K_b, N);
  
  // Validate K dimensions
  if (K_a != K_b) {
    MUNET_ERROR << "batched_matmul: K dimension mismatch: K_a=" << K_a 
                << ", K_b=" << K_b << std::endl;
    return Tensor();
  }
  int K = K_a;
  
  // Calculate batch size and strides
  int batch = 1;
  size_t a_batch_dims = a_shape.size() - 2;
  size_t b_batch_dims = b_shape.size() - 2;
  
  // For now, we only support:
  // 1. [B, M, K] × [K, N] (broadcasted weights)
  // 2. [B, M, K] × [B, K, N] (same batch dimensions)
  
  if (a_batch_dims == 0) {
    // Convert to 2D matmul
    return matmul_2d(a, b);
  }
  
  // Calculate batch size from a
  for (size_t i = 0; i < a_batch_dims; ++i) {
    batch *= static_cast<int>(a_shape[i]);
  }
  
  // Check batch compatibility for b
  if (b_batch_dims > 0) {
    int b_batch = 1;
    for (size_t i = 0; i < b_batch_dims; ++i) {
      b_batch *= static_cast<int>(b_shape[i]);
    }
    if (b_batch != batch) {
      MUNET_ERROR << "batched_matmul: batch size mismatch: a_batch=" << batch 
                  << ", b_batch=" << b_batch << std::endl;
      return Tensor();
    }
  }
  
  // Calculate strides
  int64_t stride_a = M * K;
  int64_t stride_b = b_batch_dims > 0 ? K * N : 0;  // 0 means broadcast
  int64_t stride_out = M * N;
  
  // Create output tensor
  std::vector<size_t> out_shape_vec;
  for (size_t i = 0; i < a_batch_dims; ++i) {
    out_shape_vec.push_back(a_shape[i]);
  }
  out_shape_vec.push_back(static_cast<size_t>(M));
  out_shape_vec.push_back(static_cast<size_t>(N));
  Shape out_shape(out_shape_vec.begin(), out_shape_vec.end());
  
  Tensor out(out_shape, a.device(), a.dtype());
  
  // Get BLAS capability
  Backend* backend = &a.impl_->backend();
  auto* blas = backend->blas_capability();
  if (!blas) {
    MUNET_ERROR << "batched_matmul: Backend does not support BLAS operations" << std::endl;
    return Tensor();
  }
  
  // Call backend batched_matmul
  blas->batched_matmul(*a.impl_->storage, *b.impl_->storage, *out.impl_->storage,
                       batch, M, K, N, transA, transB, stride_a, stride_b, stride_out);
  
  return out;
}

Tensor matmul_internal(const Tensor &a, const Tensor &b, bool transA, bool transB) {
  // Get shapes
  Shape a_shape = a.shape();
  Shape b_shape = b.shape();
  
  // Handle batched matmul
  if (is_batched_matmul_shape(a_shape, b_shape)) {
    return batched_matmul_internal(a, b, transA, transB);
  }
  
  // Validate 2D shapes
  if (a_shape.size() != 2 || b_shape.size() != 2) {
    MUNET_ERROR << "matmul: tensors must be 2D or batched 3D" << std::endl;
    return Tensor();
  }
  
  // Resolve dispatch to check for dtype support
  const auto dispatch = resolve_dispatch(OpId::Matmul, a);
  
  // Use CPU fallback if needed
  if (dispatch.use_cpu_fallback) {
    return matmul_cpu_fallback(a, b);
  }
  
  // 2D matmul
  int M = a_shape[0];
  int K = a_shape[1];
  int N = b_shape[1];
  
  // Validate dimensions
  if (a_shape[1] != b_shape[0]) {
    MUNET_ERROR << "matmul: dimension mismatch: a.shape = [" 
                << a_shape[0] << ", " << a_shape[1] << "], b.shape = [" 
                << b_shape[0] << ", " << b_shape[1] << "]" << std::endl;
    return Tensor();
  }
  
  // Get BLAS capability
  Backend* backend = &a.impl_->backend();
  auto* blas = backend->blas_capability();
  if (!blas) {
    MUNET_ERROR << "matmul: Backend does not support BLAS operations" << std::endl;
    return Tensor();
  }
  
  // Handle transposition
  int K_a = transA ? M : K;
  int M_eff = transA ? K : M;
  int K_b = transB ? N : K;
  int N_eff = transB ? K : N;
  
  // Create output tensor
  Shape out_shape{static_cast<size_t>(M_eff), static_cast<size_t>(N_eff)};
  Tensor out(out_shape, a.device(), a.dtype());
  
  // Call backend matmul
  blas->matmul(*a.impl_->storage, *b.impl_->storage, *out.impl_->storage,
               M_eff, K, N_eff, transA, transB);
  
  return out;
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  return matmul_internal(a, b, false, false);
}

Tensor batched_matmul(const Tensor &a, const Tensor &b) {
  return batched_matmul_internal(a, b, false, false);
}

} // namespace ops
} // namespace munet