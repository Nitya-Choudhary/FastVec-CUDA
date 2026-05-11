#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ─── KERNELS ────────────────────────────────────────────────────

__global__ void vector_add_kernel(const float* a, const float* b,
                                   float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_multiply_kernel(const float* a, const float* b,
                                        float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void scalar_scale_kernel(const float* a, float scalar,
                                     float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * scalar;
    }
}

// Dot product uses shared memory reduction
__global__ void dot_product_kernel(const float* a, const float* b,
                                    float* partial_sums, int n) {
    extern __shared__ float shared[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    shared[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

// ─── LAUNCHERS ──────────────────────────────────────────────────

torch::Tensor cuda_vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.size(0) == b.size(0), "Vectors must be the same size");

    auto c = torch::zeros_like(a);
    int n = a.size(0);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        c.data_ptr<float>(), n
    );
    return c;
}

torch::Tensor cuda_vector_multiply(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.size(0) == b.size(0), "Vectors must be the same size");

    auto c = torch::zeros_like(a);
    int n = a.size(0);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    vector_multiply_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        c.data_ptr<float>(), n
    );
    return c;
}

torch::Tensor cuda_scalar_scale(torch::Tensor a, float scalar) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");

    auto c = torch::zeros_like(a);
    int n = a.size(0);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    scalar_scale_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), scalar,
        c.data_ptr<float>(), n
    );
    return c;
}

float cuda_dot_product(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.size(0) == b.size(0), "Vectors must be the same size");

    int n = a.size(0);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    auto partial = torch::zeros({blocks}, a.options());

    dot_product_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        partial.data_ptr<float>(), n
    );

    return partial.sum().item<float>();
}
