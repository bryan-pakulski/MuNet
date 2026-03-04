#include "../backend.hpp"
#include <vector>
#include <vulkan/vulkan.h>

namespace munet {

class VulkanBackend : public Backend {
public:
  VulkanBackend();
  ~VulkanBackend() override;

  void *allocate(size_t bytes) override;
  void deallocate(void *ptr) override;
  void memset(void *ptr, int value, size_t bytes) override;

  void copy(const void *src, void *dst, size_t bytes, Device src_dev,
            Device dst_dev) override;
  void synchronize() override;
  void all_reduce(Storage &buffer, size_t num_elements) override;

  // --- Compute Kernels ---
  void add(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override;
  void mul(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override;
  void matmul(const Storage &a, const Storage &b, Storage &out, int M, int K,
              int N, bool transA, bool transB) override;
  void relu(const Storage &in, Storage &out, size_t num_elements) override;
  void relu_backward(const Storage &grad_out, const Storage &input,
                     Storage &grad_in, size_t num_elements) override;
  void sub(const Storage &a, const Storage &b, Storage &out,
           size_t num_elements) override;
  void update(Storage &weight, const Storage &grad, float lr,
              size_t num_elements) override;

  // --- Spatial Compute Stubs ---
  void conv2d(const Storage &in, const Storage &weight, const Storage *bias,
              Storage &out, int B, int iC, int iH, int iW, int oC, int kH,
              int kW, int s, int p) override;
  void conv2d_backward(const Storage &grad_out, const Storage &in,
                       const Storage &weight, Storage &grad_in, Storage &grad_w,
                       Storage *grad_b, int B, int iC, int iH, int iW, int oC,
                       int kH, int kW, int s, int p) override;
  void max_pool2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int k, int s, int p) override;
  void max_pool2d_backward(const Storage &grad_out, const Storage &in,
                           Storage &grad_in, int B, int C, int iH, int iW,
                           int k, int s, int p) override;
  void upsample2d(const Storage &in, Storage &out, int B, int C, int iH, int iW,
                  int scale) override;
  void upsample2d_backward(const Storage &grad_out, Storage &grad_in, int B,
                           int C, int iH, int iW, int scale) override;

private:
  void dispatch_kernel(VkPipeline pipeline, const std::vector<void *> &buffers,
                       void *pc, size_t pcSize, int x, int y, int z);
};

} // namespace munet
