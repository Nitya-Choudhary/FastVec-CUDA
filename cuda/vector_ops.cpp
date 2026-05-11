#include <torch/extension.h>

// Declare functions from the .cu file
torch::Tensor cuda_vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor cuda_vector_multiply(torch::Tensor a, torch::Tensor b);
torch::Tensor cuda_scalar_scale(torch::Tensor a, float scalar);
float cuda_dot_product(torch::Tensor a, torch::Tensor b);

// Macros to check input tensors
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    return cuda_vector_add(a, b);
}

torch::Tensor vector_multiply(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    return cuda_vector_multiply(a, b);
}

torch::Tensor scalar_scale(torch::Tensor a, float scalar) {
    CHECK_INPUT(a);
    return cuda_scalar_scale(a, scalar);
}

float dot_product(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    return cuda_dot_product(a, b);
}

// Register with Python via pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FastVec CUDA - GPU-accelerated vector operations";
    m.def("vector_add",      &vector_add,      "Vector addition (CUDA)");
    m.def("vector_multiply", &vector_multiply, "Element-wise multiply (CUDA)");
    m.def("scalar_scale",    &scalar_scale,    "Scalar scaling (CUDA)");
    m.def("dot_product",     &dot_product,     "Dot product (CUDA)");
}
