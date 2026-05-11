# FastVec-CUDA 🚀

A high-performance Python library that offloads vector operations to the GPU using custom CUDA C++ kernels — achieving up to **3100x speedup** over pure Python.

Built with CUDA C++, Pybind11, and PyTorch Extensions.

---

## 💡 Why FastVec?

Standard Python loops are painfully slow for large-scale numerical computation. FastVec bridges Python's ease of use with CUDA's parallel processing power — letting you run vector math on millions of elements in under a millisecond.

| Method | Time (10M elements) | Speedup |
|---|---|---|
| Python Loop | ~2.5 seconds | 1x (baseline) |
| NumPy (CPU) | ~15 ms | 160x |
| **FastVec (GPU)** | **~0.8 ms** | **3100x** |

> Tested on: Asus Vivobook with NVIDIA RTX GPU

---

## ✨ Features

- **Vector Addition** — element-wise `a + b` on GPU
- **Element-wise Multiplication** — `a * b` in parallel
- **Scalar Scaling** — multiply every element by a scalar
- **Dot Product** — GPU reduction with shared memory
- **Matrix Multiplication** — via PyTorch CUDA backend
- **Benchmarking** — built-in throughput tester
- **CPU Fallback** — works even without a GPU

---

## 🛠️ Tech Stack

- **Languages:** Python & CUDA C++
- **NVIDIA:** CUDA Toolkit 12.x, NVCC
- **Bindings:** Pybind11 / PyTorch `cpp_extension`
- **Environment:** Windows (WSL2 Ubuntu)

---

## 📁 Project Structure
