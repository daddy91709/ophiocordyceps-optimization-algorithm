"""
BBOB (Black-Box Optimization Benchmarking) / COCO Platform Standard Suite
Evaluates OOA (Meta-Hyphal Architecture) across the 5 canonical BBOB problem classes:
1. Separable Functions (Sphere, Ellipsoid, Rastrigin)
2. Low / Moderate Conditioning (Attractive Sector, Step Ellipsoidal, Rosenbrock)
3. High Conditioning / Ill-Conditioned (Rotated Ellipsoid, Discus, Bent Cigar, Different Powers)
4. Multi-Modal with Adequate Global Structure (Rotated Rastrigin, Weierstrass, Schaffer F7)
5. Multi-Modal with Weak Global Structure (Schwefel, Katsuura)

Dimensions: 10D and 30D
Budget: Standard BBOB protocol
"""
import os
import sys
import time
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ophiocordyceps import ophiocordyceps
from src.device import get_cpu_info
import src.benchmark as bm

# --- BBOB Problem Implementations ---

def bbob_sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))

def bbob_ellipsoid_sep(x: np.ndarray) -> float:
    d = len(x)
    weights = np.array([10.0**(6.0 * i / (d - 1)) for i in range(d)])
    return float(np.sum(weights * (x**2)))

def bbob_rastrigin_sep(x: np.ndarray) -> float:
    d = len(x)
    return float(10.0 * d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

def bbob_attractive_sector(x: np.ndarray) -> float:
    d = len(x)
    val = 0.0
    for i in range(d):
        val += (100.0 * x[i])**2 if x[i] > 0 else x[i]**2
    return float(val**0.9)

def bbob_step_ellipsoidal(x: np.ndarray) -> float:
    d = len(x)
    z = np.zeros(d)
    for i in range(d):
        z[i] = np.floor(0.5 + x[i]) if abs(x[i]) > 0.5 else np.floor(0.5 + 10.0 * x[i]) / 10.0
    weights = np.array([10.0**(2.0 * i / (d - 1)) for i in range(d)])
    return float(0.1 * max(abs(x[0])/1e4, np.sum(weights * (z**2))))

def bbob_rosenbrock(x: np.ndarray) -> float:
    d = len(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2))

def bbob_rotated_ellipsoid(x: np.ndarray) -> float:
    # Simulated rotated high-conditioned ellipsoid
    d = len(x)
    np.random.seed(42)
    R, _ = np.linalg.qr(np.random.randn(d, d))
    z = R @ x
    weights = np.array([10.0**(6.0 * i / (d - 1)) for i in range(d)])
    return float(np.sum(weights * (z**2)))

def bbob_discus(x: np.ndarray) -> float:
    d = len(x)
    return float(1e6 * (x[0]**2) + np.sum(x[1:]**2))

def bbob_bent_cigar(x: np.ndarray) -> float:
    d = len(x)
    return float(x[0]**2 + 1e6 * np.sum(x[1:]**2))

def bbob_diff_powers(x: np.ndarray) -> float:
    d = len(x)
    return float(np.sum([abs(x[i])**(2.0 + 4.0 * i / (d - 1)) for i in range(d)]))

def bbob_weierstrass(x: np.ndarray) -> float:
    d = len(x)
    k_max = 12
    a = 0.5
    b = 3.0
    val = 0.0
    c0 = sum([a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max)])
    for i in range(d):
        val += sum([a**k * np.cos(2 * np.pi * b**k * (x[i] + 0.5)) for k in range(k_max)]) - c0
    return float(10.0 * ((val / d)**3))

def bbob_schaffer_f7(x: np.ndarray) -> float:
    d = len(x)
    s = np.sqrt(x[:-1]**2 + x[1:]**2)
    val = np.sum(np.sqrt(s) * (1.0 + np.sin(50.0 * (s**0.2))**2))
    return float((val / (d - 1))**2 * 10.0)

def bbob_schwefel(x: np.ndarray) -> float:
    d = len(x)
    return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(abs(x)))) )

def bbob_katsuura(x: np.ndarray) -> float:
    d = len(x)
    prod = 1.0
    for i in range(d):
        term = sum([abs(2.0**j * x[i] - round(2.0**j * x[i])) / (2.0**j) for j in range(1, 33)])
        prod *= (1.0 + (i + 1) * term)**(10.0 / (d**1.2))
    return float((10.0 / d**2) * (prod - 1.0))

def bbob_rotated_rastrigin(x: np.ndarray) -> float:
    d = len(x)
    np.random.seed(1337)
    R, _ = np.linalg.qr(np.random.randn(d, d))
    z = R @ x
    return float(10.0 * d + np.sum(z**2 - 10.0 * np.cos(2.0 * np.pi * z)))


BBOB_FUNCTIONS = [
    ("F1: Sphere", bbob_sphere, "1. Separable", [-5.0, 5.0]),
    ("F2: Ellipsoid Separable", bbob_ellipsoid_sep, "1. Separable", [-5.0, 5.0]),
    ("F3: Rastrigin Separable", bbob_rastrigin_sep, "1. Separable", [-5.0, 5.0]),
    ("F4: Attractive Sector", bbob_attractive_sector, "2. Moderate Cond", [-5.0, 5.0]),
    ("F5: Step Ellipsoidal", bbob_step_ellipsoidal, "2. Moderate Cond", [-5.0, 5.0]),
    ("F6: Rosenbrock", bbob_rosenbrock, "2. Moderate Cond", [-5.0, 5.0]),
    ("F7: Rotated Ellipsoid", bbob_rotated_ellipsoid, "3. Ill-Conditioned", [-5.0, 5.0]),
    ("F8: Discus Function", bbob_discus, "3. Ill-Conditioned", [-5.0, 5.0]),
    ("F9: Bent Cigar", bbob_bent_cigar, "3. Ill-Conditioned", [-5.0, 5.0]),
    ("F10: Different Powers", bbob_diff_powers, "3. Ill-Conditioned", [-5.0, 5.0]),
    ("F11: Rotated Rastrigin", bbob_rotated_rastrigin, "4. Multi-Modal (Global)", [-5.0, 5.0]),
    ("F12: Weierstrass", bbob_weierstrass, "4. Multi-Modal (Global)", [-5.0, 5.0]),
    ("F13: Schaffer F7", bbob_schaffer_f7, "4. Multi-Modal (Global)", [-5.0, 5.0]),
    ("F14: Schwefel", bbob_schwefel, "5. Multi-Modal (Weak)", [-500.0, 500.0]),
    ("F15: Katsuura", bbob_katsuura, "5. Multi-Modal (Weak)", [-5.0, 5.0]),
]

def _eval_bbob_task(task_args: Dict[str, Any]) -> Dict[str, Any]:
    name, func, group, bounds, dim, num_runs, max_iter = (
        task_args["name"], task_args["func"], task_args["group"], task_args["bounds"],
        task_args["dim"], task_args["num_runs"], task_args["max_iter"]
    )
    lb = [bounds[0]] * dim
    ub = [bounds[1]] * dim
    
    errors = []
    times = []
    
    for r in range(num_runs):
        np.random.seed(r * 1000 + dim * 17 + 42)
        t0 = time.perf_counter()
        res = ophiocordyceps(
            n_ants=40, n_dims=dim, lower_bound=lb, upper_bound=ub,
            fitness=func, minimization=True, max_iter=max_iter, verbose=False
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        raw = res.fitness if res.fitness is not None else np.nan
        err = max(0.0, raw)
        if err < 1e-8:
            err = 0.0
        errors.append(err)
        
    return {
        "Function": name,
        "Group": group,
        "Dim": dim,
        "OOA_Best_Error": float(np.min(errors)),
        "OOA_Mean_Error": float(np.mean(errors)),
        "OOA_Std_Error": float(np.std(errors)),
        "Mean_Time_s": float(np.mean(times))
    }

def run_bbob_benchmark(dims: List[int] = [10, 30], num_runs: int = 5, max_iter: int = 200, csv_out: str = "results/bbob_ooa_results.csv"):
    cpu_info = get_cpu_info()
    workers = max(1, cpu_info["cores_logical"] - 1)
    
    tasks = []
    for name, func, grp, bounds in BBOB_FUNCTIONS:
        for d in dims:
            tasks.append({
                "name": name, "func": func, "group": grp, "bounds": bounds,
                "dim": d, "num_runs": num_runs, "max_iter": max_iter
            })
            
    print("=" * 85)
    print("  BBOB / COCO 15-FUNCTION BENCHMARK SUITE EVALUATION")
    print(f"  Available CPU Workers: {workers} | Tasks: {len(tasks)}")
    print("=" * 85)
    
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_eval_bbob_task, tasks))
    total_time = time.perf_counter() - t0
    
    df = pd.DataFrame(results).sort_values(by=["Dim", "Function"]).reset_index(drop=True)
    for _, r in df.iterrows():
        print(f"• [{r['Group']}] {r['Function']:<25} ({r['Dim']}D): Mean Err = {r['OOA_Mean_Error']:<10.4e} (Best: {r['OOA_Best_Error']:<10.4e}) [{r['Mean_Time_s']:.2f}s]")
        
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    df.to_csv(csv_out, index=False)
    print("=" * 85)
    print(f"BBOB Benchmark Completed in {total_time:.2f}s! Saved to: {csv_out}")
    print("=" * 85)
    return df

if __name__ == "__main__":
    run_bbob_benchmark()
