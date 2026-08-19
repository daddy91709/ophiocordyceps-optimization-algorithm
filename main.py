"""
Ophiocordyceps Optimization Algorithm (OOA) - Main Execution & Demo Script.
Demonstrates Automatic Hardware Device Dispatch (CPU / GPU Tensor Engine) and Multidimensional Optimization.
"""
import sys
import time
import numpy as np

from src.ophiocordyceps import ophiocordyceps
from src.device import get_hardware_summary, detect_gpu, get_cpu_info
import src.benchmark as bm


def main():
    print("=" * 75)
    print("   OPHIOCORDYCEPS OPTIMIZATION ALGORITHM (OOA) - DEMO EXECUTION")
    print("=" * 75)

    # 0. Hardware and Device Diagnostics
    print("\n" + get_hardware_summary())

    gpu_status = detect_gpu()
    active_device = gpu_status["backend"] if gpu_status["available"] else "cpu"
    print(f"\n[Engine Dispatch] Auto-selected backend: {active_device.upper()}")

    # 1. Ackley Function (2D) - Multimodal non-separable
    print("\n[1/3] Test: Ackley (2D) [Global Minimum: 0.0 at (0, 0)]")
    t0 = time.perf_counter()
    best_ackley = ophiocordyceps(
        n_ants=30,
        n_dims=2,
        lower_bound=[-5, -5],
        upper_bound=[5, 5],
        fitness=bm.ackley,
        minimization=True,
        max_iter=40,
        device="auto"
    )
    t_ackley = time.perf_counter() - t0
    print(f"-> Fitness: {best_ackley.fitness:.6e}")
    print(f"-> Position: {best_ackley.position}")
    print(f"-> Elapsed: {t_ackley:.3f}s")

    # 2. Sphere Function (30D) - High Dimensional
    print("\n[2/3] Test: Sphere (30D) [Global Minimum: 0.0 at (0, ..., 0)]")
    t0 = time.perf_counter()
    best_sphere = ophiocordyceps(
        n_ants=40,
        n_dims=30,
        lower_bound=[-100.0] * 30,
        upper_bound=[100.0] * 30,
        fitness=bm.sphere,
        minimization=True,
        max_iter=60,
        device="auto"
    )
    t_sphere = time.perf_counter() - t0
    print(f"-> Fitness: {best_sphere.fitness:.6e}")
    print(f"-> Norm(Position): {np.linalg.norm(best_sphere.position):.6e}")
    print(f"-> Elapsed: {t_sphere:.3f}s")

    # 3. Alpine1 Function (30D) - Multimodal Separable
    print("\n[3/3] Test: Alpine1 (30D) [Global Minimum: 0.0 at (0, ..., 0)]")
    t0 = time.perf_counter()
    best_alpine = ophiocordyceps(
        n_ants=40,
        n_dims=30,
        lower_bound=[-10.0] * 30,
        upper_bound=[10.0] * 30,
        fitness=bm.alpine1,
        minimization=True,
        max_iter=60,
        device="auto"
    )
    t_alpine = time.perf_counter() - t0
    print(f"-> Fitness: {best_alpine.fitness:.6e}")
    print(f"-> Position (first 3 dims): {best_alpine.position[:3]}")
    print(f"-> Elapsed: {t_alpine:.3f}s")

    print("\n" + "=" * 75)
    print("   All demonstrations completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
