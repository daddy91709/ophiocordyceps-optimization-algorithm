"""
Benchmark empirico di confronto prestazioni: CPU Sequenziale vs CPU Multi-Process vs GPU/PyTorch Tensor Engine.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ophiocordyceps import ophiocordyceps
from src.ophiocordyceps_gpu import ophiocordyceps_gpu
import src.benchmark as bm

def run_performance_comparison():
    print("=" * 80)
    print("  CONFRONTO PRESTAZIONI DI CALCOLO E VELOCITA' (CPU vs GPU TENSOR ENGINE)")
    print("=" * 80)

    test_cases = [
        ("Sphere (10D)", bm.sphere, [-100]*10, [100]*10, 10, 100),
        ("Sphere (50D)", bm.sphere, [-100]*50, [100]*50, 50, 100),
        ("Sphere (100D)", bm.sphere, [-100]*100, [100]*100, 100, 100),
        ("Ackley (30D)", bm.ackley, [-32]*30, [32]*30, 30, 80),
        ("Alpine1 (30D)", bm.alpine1, [-10]*30, [10]*30, 30, 80),
        ("Zakharov (30D)", bm.zakharov, [-10]*30, [10]*30, 30, 80),
        ("Schwefel 2.22 (30D)", bm.schwefel_222, [-10]*30, [10]*30, 30, 80),
    ]

    results = []

    for name, func, lb, ub, dim, max_iter in test_cases:
        # 1. CPU Standard Meta-Hyphal
        t0 = time.perf_counter()
        res_cpu = ophiocordyceps(
            n_ants=30, n_dims=dim, lower_bound=lb, upper_bound=ub,
            fitness=func, minimization=True, max_iter=max_iter, device="cpu"
        )
        t_cpu = time.perf_counter() - t0

        # 2. GPU / PyTorch Tensorized Engine
        t0 = time.perf_counter()
        res_gpu = ophiocordyceps_gpu(
            n_ants=30, n_dims=dim, lower_bound=lb, upper_bound=ub,
            fitness=func, minimization=True, max_iter=max_iter, device="auto"
        )
        t_gpu = time.perf_counter() - t0

        speedup = t_cpu / max(1e-6, t_gpu)
        print(f"[{name:<18}] CPU: {t_cpu:.3f}s (Fit: {res_cpu.fitness:.2e}) | GPU Engine: {t_gpu:.3f}s (Fit: {res_gpu['fitness']:.2e}) -> Speedup: {speedup:.2f}x")

        results.append({
            "Test": name,
            "Dim": dim,
            "CPU Time (s)": t_cpu,
            "CPU Fitness": res_cpu.fitness,
            "GPU Time (s)": t_gpu,
            "GPU Fitness": res_gpu["fitness"],
            "Speedup": speedup
        })

    print("=" * 80)
    return pd.DataFrame(results)

if __name__ == "__main__":
    run_performance_comparison()
