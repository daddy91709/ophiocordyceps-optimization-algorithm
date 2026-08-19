"""
Ophiocordyceps Optimization Algorithm - Main Demo & Test Script
Includes Hardware Acceleration Diagnostics and Multi-Core Benchmarking.
"""
import sys
import time
import numpy as np

from src.ophiocordyceps import ophiocordyceps
import src.benchmark as bm
from src.device import get_hardware_summary, get_cpu_info, detect_gpu


def main():
    print("=" * 70)
    print("   OPHIOCORDYCEPS OPTIMIZATION ALGORITHM - DEMO EXECUTION")
    print("=" * 70)

    # 0. Hardware and Acceleration Summary
    print("\n" + get_hardware_summary())

    # 1. Ackley Function (2D) - Multimodal non-separable
    print("\n[1/3] Test su Funzione Ackley (2D) [Minimo globale noto: 0.0 in (0, 0)]")
    t0 = time.perf_counter()
    best_ackley = ophiocordyceps(
        n_ants=30,
        n_dims=2,
        lower_bound=[-5, -5],
        upper_bound=[5, 5],
        fitness=bm.ackley,
        minimization=True,
        use_best_guidance=True,
        verbose=False,
        max_iter=40,
        convergence_patience=10
    )
    t_ackley = time.perf_counter() - t0
    print(f"-> Ackley - Miglior Fitness: {best_ackley.fitness:.6e}")
    print(f"-> Ackley - Posizione Trovata: {best_ackley.position}")
    print(f"-> Ackley - Tempo di calcolo: {t_ackley:.3f}s")

    # 2. Sphere Function (10D) - Unimodal separable with Multi-Worker evaluation
    print("\n[2/3] Test su Funzione Sphere (10D) [Minimo globale noto: 0.0 in (0, ..., 0)]")
    cpu_info = get_cpu_info()
    workers = min(4, cpu_info["cores_logical"])
    t0 = time.perf_counter()
    best_sphere = ophiocordyceps(
        n_ants=40,
        n_dims=10,
        lower_bound=[-5.12] * 10,
        upper_bound=[5.12] * 10,
        fitness=bm.sphere,
        minimization=True,
        use_best_guidance=True,
        verbose=False,
        max_iter=50,
        convergence_patience=10,
        n_workers=workers
    )
    t_sphere = time.perf_counter() - t0
    print(f"-> Sphere (10D, {workers} CPU Workers) - Miglior Fitness: {best_sphere.fitness:.6e}")
    print(f"-> Sphere (10D) - Norma Posizione: {np.linalg.norm(best_sphere.position):.6e}")
    print(f"-> Sphere (10D) - Tempo di calcolo: {t_sphere:.3f}s")

    # 3. Alpine1 Function (2D) - Multimodal separable
    print("\n[3/3] Test su Funzione Alpine1 (2D) [Minimo globale noto: 0.0 in (0, 0)]")
    t0 = time.perf_counter()
    best_alpine = ophiocordyceps(
        n_ants=30,
        n_dims=2,
        lower_bound=[-10, -10],
        upper_bound=[10, 10],
        fitness=bm.alpine1,
        minimization=True,
        use_best_guidance=True,
        verbose=False,
        max_iter=50,
        convergence_patience=10
    )
    t_alpine = time.perf_counter() - t0
    print(f"-> Alpine1 - Miglior Fitness: {best_alpine.fitness:.6e}")
    print(f"-> Alpine1 - Posizione Trovata: {best_alpine.position}")
    print(f"-> Alpine1 - Tempo di calcolo: {t_alpine:.3f}s")

    print("\n" + "=" * 70)
    print("   Tutti i test completati con successo!")
    print("=" * 70)


if __name__ == "__main__":
    main()
