#!/usr/bin/env python3
"""
Example usage of the Vector Math GPU library
"""

import numpy as np
from vector_math_gpu import VectorOps

def main():
    # Initialize
    ops = VectorOps(device='cuda')  # Will use GPU if available
    
    # Get system info
    print("System Information:")
    info = ops.system_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    print()
    
    # Example 1: Vector operations
    print("Example 1: Vector Operations")
    print("-" * 40)
    
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [5.0, 4.0, 3.0, 2.0, 1.0]
    
    print(f"a = {a}")
    print(f"b = {b}")
    
    # Addition
    c_add = ops.add(a, b)
    print(f"\nAddition (a + b):")
    print(f"Result: {c_add.cpu().numpy() if hasattr(c_add, 'cpu') else c_add}")
    
    # Multiplication
    c_mul = ops.multiply(a, b)
    print(f"\nElement-wise multiplication (a * b):")
    print(f"Result: {c_mul.cpu().numpy() if hasattr(c_mul, 'cpu') else c_mul}")
    
    # Dot product
    dot_result = ops.dot(a, b)
    print(f"\nDot product (a · b): {dot_result}")
    
    # Example 2: Matrix operations
    print("\n" + "=" * 60)
    print("Example 2: Matrix Operations")
    print("-" * 40)
    
    A = np.random.randn(3, 4).astype(np.float32)
    B = np.random.randn(4, 2).astype(np.float32)
    
    print(f"Matrix A (3x4):\n{A}")
    print(f"\nMatrix B (4x2):\n{B}")
    
    # Matrix multiplication
    C = ops.matmul(A, B)
    print(f"\nMatrix multiplication (A @ B):")
    print(f"Result shape: {C.shape}")
    print(f"Result:\n{C.cpu().numpy() if hasattr(C, 'cpu') else C}")
    
    # Example 3: Performance benchmark
    print("\n" + "=" * 60)
    print("Example 3: Quick Performance Test")
    print("-" * 40)
    
    results = ops.benchmark(size=1000000, operations=10)
    print(f"Benchmark completed!")
    
    # Example 4: Large-scale computation
    print("\n" + "=" * 60)
    print("Example 4: Large Vector Computation")
    print("-" * 40)
    
    # Create large vectors
    size = 5000000  # 5 million elements
    print(f"Creating vectors with {size:,} elements...")
    
    large_a = np.random.randn(size).astype(np.float32)
    large_b = np.random.randn(size).astype(np.float32)
    
    print("Performing operations...")
    
    start = time.time()
    large_result = ops.add(large_a, large_b)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    print(f"Operation completed in {elapsed:.4f} seconds")
    print(f"That's {size / elapsed:,.0f} elements per second!")
    
    # Verify with NumPy
    print("\nVerifying with NumPy (CPU)...")
    start = time.time()
    numpy_result = large_a + large_b
    numpy_time = time.time() - start
    
    print(f"NumPy (CPU) time: {numpy_time:.4f} seconds")
    
    if torch.cuda.is_available():
        speedup = numpy_time / elapsed
        print(f"\nGPU Speedup: {speedup:.2f}x faster!")
        
        # Check accuracy
        max_error = np.max(np.abs(large_result.cpu().numpy() - numpy_result))
        print(f"Maximum error: {max_error:.6f}")

if __name__ == "__main__":
    import time
    import torch
    
    main()