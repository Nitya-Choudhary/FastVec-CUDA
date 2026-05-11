import pytest
import torch
import numpy as np
from vector_math_gpu import VectorOps

ops = VectorOps(device='cuda' if torch.cuda.is_available() else 'cpu')

def test_add():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    result = ops.add(a, b)
    expected = [5.0, 7.0, 9.0]
    assert np.allclose(result.cpu().numpy(), expected)

def test_multiply():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    result = ops.multiply(a, b)
    expected = [4.0, 10.0, 18.0]
    assert np.allclose(result.cpu().numpy(), expected)

def test_dot():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    result = ops.dot(a, b)
    assert abs(result - 32.0) < 1e-4  # 1*4 + 2*5 + 3*6 = 32

def test_scale():
    a = [1.0, 2.0, 3.0]
    result = ops.scale(a, 2.0)
    expected = [2.0, 4.0, 6.0]
    assert np.allclose(result.cpu().numpy(), expected)
