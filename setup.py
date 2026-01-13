from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Check if CUDA is available
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Building CPU-only version.")
    from setuptools import Extension
    extensions = []
else:
    extensions = [
        CUDAExtension(
            name='vector_math_gpu.cuda_ops',
            sources=[
                'cuda/vector_ops.cpp',
                'cuda/vector_ops_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O2', '-std=c++14'],
                'nvcc': ['-O2', '-std=c++14', '-Xcompiler', '-fPIC']
            }
        )
    ]

setup(
    name='vector_math_gpu',
    version='0.1.0',
    packages=['vector_math_gpu'],
    ext_modules=extensions,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
    ],
    python_requires='>=3.8',
)