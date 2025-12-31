import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

BACKEND = os.getenv("BACKEND_TYPE", "numpy").lower()

if BACKEND == "cupy":
    try:
        import cupy as xp
        print("Using CuPy (GPU)")
        USE_GPU = True
        def to_cpu(x): return x.get() if hasattr(x, 'get') else x
        def to_gpu(x): return xp.asarray(x)
        
        def get_device(x):
            if hasattr(x, 'device'):
                return f"cuda:{x.device.id}"
            return "cpu"

    except ImportError:
        print("CuPy (GPU) not available, using NumPy (CPU)")
        import numpy as xp
        USE_GPU = False
        def to_cpu(x): return x
        def to_gpu(x): raise Exception("NumPy (CPU) does not support GPU")
        def get_device(x): return "cpu" # If cupy is not available, we have to use numpy which only uses cpu
else:
    print("Using NumPy (CPU)")
    import numpy as xp
    USE_GPU = False
    def to_cpu(x): return x
    def to_gpu(x): raise Exception("NumPy (CPU) does not support GPU")
    def get_device(x): return "cpu" # Numpy only uses cpu

for attr in dir(xp):
    if not attr.startswith("_"):
        if attr not in globals():
            globals()[attr] = getattr(xp, attr)