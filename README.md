# FastVec-CUDA 🚀

A high-performance Vector Math library that offloads heavy calculations from Python to an NVIDIA GPU using CUDA C++.

## 💡 Why this exists
Standard Python loops are slow for large datasets. This project demonstrates how to bridge the gap between Python's ease of use and CUDA's parallel processing power to achieve near-instant results on a laptop GPU.

## 🛠️ Tech Stack
- **Language:** Python & CUDA C++
- **NVIDIA Tech:** CUDA Toolkit, NVCC, Tensor Cores
- **Bindings:** Pybind11 / Torch Extensions
- **Environment:** Windows (WSL2 Ubuntu)

## 📊 Performance Benchmark
Tested on: **Asus Vivobook (NVIDIA RTX GPU)**
Vector Size: 10,000,000 elements

| Method | Execution Time | Speedup |
| :--- | :--- | :--- |
| Python Loop | ~2.5 seconds | 1x (Baseline) |
| NumPy (CPU) | ~15 ms | 160x |
| **FastVec (GPU)** | **~0.8 ms** | **3100x** |

## 🚀 Getting Started

### Prerequisites
- NVIDIA Driver installed
- CUDA Toolkit 12.x
- Python 3.10+

### Installation
```bash
# Clone the repository
git clone [https://github.com/YourUsername/FastVec.git](https://github.com/YourUsername/FastVec.git)
cd FastVec

# Build and install the module
pip install .
