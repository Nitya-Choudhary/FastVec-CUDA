import torch
import numpy as np
import time

# Try to import the compiled CUDA extension
try:
    from . import cuda_ops
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    cuda_ops = None
    CUDA_AVAILABLE = False


def _to_tensor(data, device):
    if isinstance(data, torch.Tensor):
        return data.float().to(device).contiguous()
    return torch.tensor(data, dtype=torch.float32, device=device).contiguous()


class VectorOps:
    def __init__(self, device='cuda'):
        if device == 'cuda' and CUDA_AVAILABLE:
            self.device = torch.device('cuda')
            self.use_gpu = True
        else:
            self.device = torch.device('cpu')
            self.use_gpu = False

    def system_info(self):
        info = {
            "PyTorch version": torch.__version__,
            "CUDA available": torch.cuda.is_available(),
            "Device": str(self.device),
        }
        if torch.cuda.is_available():
            info["GPU"] = torch.cuda.get_device_name(0)
            info["CUDA version"] = torch.version.cuda
            mem = torch.cuda.get_device_properties(0).total_memory
            info["GPU Memory"] = f"{mem / 1e9:.1f} GB"
        return info

    def add(self, a, b):
        ta = _to_tensor(a, self.device)
        tb = _to_tensor(b, self.device)
        if self.use_gpu and cuda_ops:
            return cuda_ops.vector_add(ta, tb)
        return ta + tb

    def multiply(self, a, b):
        ta = _to_tensor(a, self.device)
        tb = _to_tensor(b, self.device)
        if self.use_gpu and cuda_ops:
            return cuda_ops.vector_multiply(ta, tb)
        return ta * tb

    def scale(self, a, scalar):
        ta = _to_tensor(a, self.device)
        if self.use_gpu and cuda_ops:
            return cuda_ops.scalar_scale(ta, float(scalar))
        return ta * scalar

    def dot(self, a, b):
        ta = _to_tensor(a, self.device)
        tb = _to_tensor(b, self.device)
        if self.use_gpu and cuda_ops:
            return cuda_ops.dot_product(ta, tb)
        return torch.dot(ta, tb).item()

    def matmul(self, A, B):
        tA = _to_tensor(A, self.device)
        tB = _to_tensor(B, self.device)
        return torch.matmul(tA, tB)

    def benchmark(self, size=1_000_000, operations=10):
        print(f"  Benchmarking with {size:,} elements, {operations} runs...")
        a = torch.randn(size, dtype=torch.float32, device=self.device)
        b = torch.randn(size, dtype=torch.float32, device=self.device)

        # Warmup
        _ = self.add(a, b)
        if self.use_gpu:
            torch.cuda.synchronize()

        times = []
        for _ in range(operations):
            start = time.perf_counter()
            self.add(a, b)
            if self.use_gpu:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg = sum(times) / len(times) * 1000
        print(f"  Average time: {avg:.3f} ms")
        print(f"  Throughput:   {size / (avg / 1000):,.0f} elements/sec")
        return times
