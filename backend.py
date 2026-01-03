import os
import numpy as np
from config import BACKEND_TYPE

BACKEND = BACKEND_TYPE.lower()

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

if BACKEND == "cupy" and CUPY_AVAILABLE:
    xp = cp
    USE_GPU = True
    print("Using CuPy (GPU)")
else:
    xp = np
    USE_GPU = False
    print("Using NumPy (CPU)")


def to_cpu(x):
    if CUPY_AVAILABLE and isinstance(x, cp.ndarray):
        return x.get()
    return x


def to_gpu(x):
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available")
    if isinstance(x, cp.ndarray):
        return x
    return cp.asarray(x)


def get_device(x):
    if CUPY_AVAILABLE and isinstance(x, cp.ndarray):
        return f"cuda:{x.device.id}"
    return "cpu"


__all__ = ["xp", "USE_GPU", "to_cpu", "to_gpu", "get_device"]
