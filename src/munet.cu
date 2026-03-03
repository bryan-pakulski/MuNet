// munet.cu
#include "munet.hpp"

#ifdef MUNET_USE_CUDA
#include <cuda_runtime.h>

namespace cuda_kernels {

__global__ void relu_forward_kernel(const float* in, float* out, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

__global__ void relu_backward_kernel(const float* go, const float* in, float* gi, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) gi[i] = in[i] > 0.0f ? go[i] : 0.0f;
}

__global__ void linear_forward_kernel(const float* in, const float* w, const float* b, float* out, int batch_size, int in_f, int out_f) {
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

__global__ void linear_backward_dx_kernel(const float* go, const float* w, float* gi, int batch_size, int in_f, int out_f) {
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

__global__ void linear_backward_dw_db_kernel(const float* go, const float* in, float* gw, float* gb, int batch_size, int in_f, int out_f) {
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

__global__ void sgd_step_kernel(float* w, const float* grad, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        w[i] -= lr * grad[i];
    }
}


// Add bias 'b' (size out_f) to 'out' (size batch_size * out_f)
// out is [Batch, Out] in memory.
__global__ void add_bias_kernel(float* out, const float* b, int size, int out_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int col = idx % out_f;
        out[idx] += b[col];
    }
}

__global__ void elementwise_mul_kernel(const float* a, const float* b, float* out, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = a[i] * b[i];
}

__global__ void cross_entropy_gradient_kernel(const float* logits, const float* targets, float* grad_input, float* loss_out, int batch_size, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        const float* logit_row = logits + b * num_classes;
        const float* target_row = targets + b * num_classes;
        float* grad_row = grad_input + b * num_classes;

        // 1. Find Max (for numerical stability)
        float max_val = -1e30f; // Simple low value
        for(int i=0; i<num_classes; ++i) {
            if(logit_row[i] > max_val) max_val = logit_row[i];
        }

        // 2. Sum Exponentials
        float sum_exp = 0.0f;
        for(int i=0; i<num_classes; ++i) {
            sum_exp += expf(logit_row[i] - max_val);
        }

        // 3. Calculate Gradient: (p_i - y_i) / batch_size
        //    And accumulate loss for this sample
        float sample_loss = 0.0f;
        const float epsilon = 1e-7f;
        for(int i=0; i<num_classes; ++i) {
            float p = expf(logit_row[i] - max_val) / sum_exp;
            if (p < epsilon) p = epsilon;
            if (p > 1.0f - epsilon) p = 1.0f - epsilon;
            float t = target_row[i];
            
            if (t > 0.0f) {
                sample_loss -= t * logf(p);
            }
            grad_row[i] = (p - t) / (float)batch_size;
        }
        
        // Simple atomic add for total loss (not most efficient reduction, but keeps data on GPU)
        atomicAdd(loss_out, sample_loss / (float)batch_size);
    }
}
















// Wrapper functions callable from standard C++
void relu_forward(const float* in, float* out, size_t size) {
    int threads = 256;
    int blocks = (size + 255) / 256;
    relu_forward_kernel<<<blocks, threads>>>(in, out, size);
    // REMOVED: cudaDeviceSynchronize();  <-- ALLOW ASYNC EXECUTION
}

void relu_backward(const float* go, const float* in, float* gi, size_t size) {
    int threads = 256;
    int blocks = (size + 255) / 256;
    relu_backward_kernel<<<blocks, threads>>>(go, in, gi, size);
}

void linear_forward(const float* in, const float* w, const float* b, float* out, int batch_size, int in_f, int out_f) {
    dim3 threads(16, 16);
    dim3 blocks((batch_size + 15) / 16, (out_f + 15) / 16);
    linear_forward_kernel<<<blocks, threads>>>(in, w, b, out, batch_size, in_f, out_f);
}

void linear_backward_dx(const float* go, const float* w, float* gi, int batch_size, int in_f, int out_f) {
    dim3 threads(16, 16);
    dim3 blocks((batch_size + 15) / 16, (in_f + 15) / 16);
    linear_backward_dx_kernel<<<blocks, threads>>>(go, w, gi, batch_size, in_f, out_f);
}

void linear_backward_dw_db(const float* go, const float* in, float* gw, float* gb, int batch_size, int in_f, int out_f) {
    dim3 threads(16, 16);
    dim3 blocks((in_f + 15) / 16, (out_f + 15) / 16);
    linear_backward_dw_db_kernel<<<blocks, threads>>>(go, in, gw, gb, batch_size, in_f, out_f);
}

void sgd_step(float* w, const float* grad, float lr, int size) {
    int threads = 256;
    int blocks = (size + 255) / 256;
    sgd_step_kernel<<<blocks, threads>>>(w, grad, lr, size);
}

void add_bias(float* out, const float* b, int batch_size, int out_f) {
    int total_size = batch_size * out_f;
    int threads = 256;
    int blocks = (total_size + 255) / 256;
    add_bias_kernel<<<blocks, threads>>>(out, b, total_size, out_f);
}

void elementwise_mul(const float* a, const float* b, float* out, size_t size) {
    int threads = 256;
    int blocks = (size + 255) / 256;
    elementwise_mul_kernel<<<blocks, threads>>>(a, b, out, size);
}

float cross_entropy_loss_cuda(const float* logits, const float* targets, float* grad_output, int batch_size, int num_classes) {
    // Use persistent workspace (No Malloc/Free)
    float* d_loss = Context::instance().d_workspace;
    cudaMemsetAsync(d_loss, 0, sizeof(float), 0); // Async set
    int threads = 256;
    int blocks = (batch_size + 255) / 256;
    cross_entropy_gradient_kernel<<<blocks, threads>>>(logits, targets, grad_output, d_loss, batch_size, num_classes);


    // Only sync here to get the loss value back for logging
    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    
    return h_loss;
}


} // namespace cuda_kernels
#endif
