#pragma once

#ifdef MUNET_USE_CUDA
    #include <cuda_runtime.h>
		#include <cublas_v2.h>
#endif

// Context Singleton to manage CUDA resources efficiently
class Context {
public:
    static Context& instance() {
        static Context inst;
        return inst;
    }

#ifdef MUNET_USE_CUDA
    cublasHandle_t cublas_handle;
    float* d_workspace = nullptr;
#endif

private:
    Context() {
#ifdef MUNET_USE_CUDA
        cublasCreate(&cublas_handle);
        cudaMalloc(&d_workspace, 1024);
#endif
    }
    ~Context() {
#ifdef MUNET_USE_CUDA
        cublasDestroy(cublas_handle);
        cudaFree(d_workspace);
#endif
    }
};

enum class Device { CPU, CUDA };
enum class DataType { FP32, FP16 };
