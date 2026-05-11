# Setup Guide — FastVec-CUDA

Complete installation and build instructions for all platforms.

---

## 📋 Requirements

| Requirement | Version |
|---|---|
| Python | 3.8+ |
| PyTorch | 2.0+ (CUDA build) |
| CUDA Toolkit | 12.x |
| NVIDIA Driver | 525+ |
| GCC (Linux) | 9+ |
| Visual Studio (Windows) | 2019 or 2022 |

---

## 🖥️ Platform-Specific Setup

### Windows (WSL2) — Recommended

This project was developed and tested on Windows using WSL2 (Ubuntu).

**Step 1 — Install WSL2**
```bash
wsl --install
# Restart your PC, then open Ubuntu from Start Menu
```

**Step 2 — Install CUDA Toolkit inside WSL2**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

**Step 3 — Add CUDA to PATH**
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Step 4 — Verify CUDA**
```bash
nvcc --version
nvidia-smi
```

---

### Linux (Ubuntu/Debian) — Native

**Step 1 — Install CUDA Toolkit**
```bash
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
```

**Step 2 — Add CUDA to PATH**
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Step 3 — Verify**
```bash
nvcc --version
nvidia-smi
```

---

## 🐍 Python Environment

It is recommended to use a virtual environment:

```bash
python -m venv fastvec-env

# Activate (Linux/WSL2)
source fastvec-env/bin/activate

# Activate (Windows CMD)
fastvec-env\Scripts\activate
```

---

## 🔥 Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify PyTorch sees your GPU:
```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # e.g. NVIDIA GeForce RTX 3050
print(torch.version.cuda)              # 12.4
```

---

## 📦 Install FastVec-CUDA

**Step 1 — Clone the repo**
```bash
git clone https://github.com/Nitya-Choudhary/FastVec-CUDA.git
cd FastVec-CUDA
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Build and install the CUDA extension**
```bash
pip install .
```

> This compiles the `.cu` and `.cpp` files using `nvcc`. May take 1–3 minutes on first build.

---

## ✅ Verify Installation

```bash
python -c "from vector_math_gpu import VectorOps; ops = VectorOps(); print(ops.system_info())"
```

Expected output:
