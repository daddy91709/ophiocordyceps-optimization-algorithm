"""
Ophiocordyceps Optimization Algorithm (OOA) Package.
"""
from src.ophiocordyceps import Ant, ophiocordyceps
from src.ophiocordyceps_gpu import ophiocordyceps_gpu
from src.device import get_hardware_summary, detect_gpu, get_cpu_info
import src.benchmark as benchmark

__all__ = [
    "Ant",
    "ophiocordyceps",
    "ophiocordyceps_gpu",
    "get_hardware_summary",
    "detect_gpu",
    "get_cpu_info",
    "benchmark"
]
