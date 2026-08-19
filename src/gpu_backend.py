"""
GPU and Tensor Acceleration Backend for Ophiocordyceps Algorithm.
Supports PyTorch (CUDA, DirectML, MPS) and CuPy, with CPU vectorized fallback.
"""
import time
import numpy as np
from typing import Callable, Optional, Tuple, Union, Dict, Any

from src.device import detect_gpu


class GPUTensorBackend:
    """
    Acceleratore tensoriale per la popolazione di formiche.
    Permette di eseguire mutazioni, vincoli e valutazioni di fitness vettorizzate su GPU.
    """
    def __init__(self, device: str = "auto", precision: str = "float64"):
        self.device_pref = device
        self.precision = precision
        self.backend = None
        self.device = None
        self.torch = None
        self.cupy = None

        self._initialize_backend()

    def _initialize_backend(self):
        gpu_info = detect_gpu()
        
        # 1. Prova PyTorch
        try:
            import torch
            self.torch = torch
            self.dtype = torch.float64 if self.precision == "float64" else torch.float32

            if self.device_pref in ["cuda", "gpu", "auto"] and torch.cuda.is_available():
                self.backend = "pytorch-cuda"
                self.device = torch.device("cuda:0")
            elif self.device_pref in ["mps", "auto"] and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.backend = "pytorch-mps"
                self.device = torch.device("mps")
            else:
                try:
                    import torch_directml
                    if self.device_pref in ["directml", "gpu", "auto"] and torch_directml.is_available():
                        self.backend = "pytorch-directml"
                        self.device = torch_directml.device()
                except ImportError:
                    pass

            if self.device is None:
                self.backend = "pytorch-cpu"
                self.device = torch.device("cpu")
            return
        except ImportError:
            pass

        # 2. Prova CuPy
        try:
            import cupy
            self.cupy = cupy
            self.backend = "cupy-cuda"
            return
        except ImportError:
            pass

        # 3. Fallback NumPy CPU
        self.backend = "numpy-cpu"

    def get_info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "device": str(self.device) if self.device is not None else "CPU",
            "precision": self.precision,
            "torch_available": self.torch is not None,
            "cupy_available": self.cupy is not None
        }

    def to_tensor(self, array: Union[np.ndarray, list]) -> Any:
        arr = np.array(array, dtype=np.float64)
        if self.torch is not None and self.device is not None:
            return self.torch.tensor(arr, dtype=self.dtype, device=self.device)
        elif self.cupy is not None:
            return self.cupy.asarray(arr)
        return arr

    def to_numpy(self, tensor: Any) -> np.ndarray:
        if self.torch is not None and isinstance(tensor, self.torch.Tensor):
            return tensor.detach().cpu().numpy()
        elif self.cupy is not None and isinstance(tensor, self.cupy.ndarray):
            return self.cupy.asnumpy(tensor)
        return np.array(tensor)

    def batch_evaluate(self, fitness_fn: Callable, population_tensor: Any) -> np.ndarray:
        pop_np = self.to_numpy(population_tensor)
        n_ants = pop_np.shape[0]
        fits = np.empty(n_ants, dtype=float)
        for i in range(n_ants):
            fits[i] = fitness_fn(pop_np[i])
        return fits

    def apply_boundary_tensor(self, pop_tensor: Any, lower_bound: Any, upper_bound: Any) -> Any:
        if self.torch is not None and isinstance(pop_tensor, self.torch.Tensor):
            lb = lower_bound if isinstance(lower_bound, self.torch.Tensor) else self.to_tensor(lower_bound)
            ub = upper_bound if isinstance(upper_bound, self.torch.Tensor) else self.to_tensor(upper_bound)
            return self.torch.clamp(pop_tensor, min=lb, max=ub)
        elif self.cupy is not None and isinstance(pop_tensor, self.cupy.ndarray):
            lb = lower_bound if isinstance(lower_bound, self.cupy.ndarray) else self.to_tensor(lower_bound)
            ub = upper_bound if isinstance(upper_bound, self.cupy.ndarray) else self.to_tensor(upper_bound)
            return self.cupy.clip(pop_tensor, lb, ub)
        else:
            return np.clip(pop_tensor, lower_bound, upper_bound)
