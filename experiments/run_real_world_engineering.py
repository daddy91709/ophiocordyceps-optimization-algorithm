"""
Real-World Engineering Optimization Benchmark Suite (CEC RWOP)
Formulates 7 canonical engineering design problems with physical constraints:
1. Pressure Vessel Design (4 variables, 4 constraints)
2. Welded Beam Design (4 variables, 7 constraints)
3. Tension/Compression Spring Design (3 variables, 4 constraints)
4. Speed Reducer / Gearbox Design (7 variables, 11 constraints)
5. Gear Train Design (4 variables)
6. Three-Bar Truss Design (2 variables, 3 constraints)
7. Cantilever Beam Design (5 variables, 1 constraint)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ophiocordyceps import ophiocordyceps
from src.device import get_cpu_info

# --- Engineering Problem Formulations ---

def pressure_vessel(x: np.ndarray) -> float:
    # x = [Ts, Th, R, L]
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    # Objective: Minimize manufacturing cost
    f = 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * (x3**2) + 3.1661 * (x1**2) * x4 + 19.84 * (x1**2) * x3
    # Constraints: g(x) <= 0
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -np.pi * (x3**2) * x4 - (4.0/3.0) * np.pi * (x3**3) + 1296000.0
    g4 = x4 - 240.0
    
    penalty = sum(max(0.0, g)**2 for g in [g1, g2, g3, g4]) * 1e5
    return f + penalty

def welded_beam(x: np.ndarray) -> float:
    # x = [h, l, t, b]
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    f = 1.10471 * (x1**2) * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
    
    P = 6000.0
    L = 14.0
    E = 30e6
    G = 12e6
    tau_max = 13600.0
    sigma_max = 30000.0
    delta_max = 0.25
    
    M = P * (L + x2 / 2.0)
    R = np.sqrt(0.25 * (x2**2) + ((x1 + x3) / 2.0)**2)
    J = 2.0 * (np.sqrt(2.0) * x1 * x2 * ((x2**2) / 4.0 + ((x1 + x3) / 2.0)**2))
    tau_prime = P / (np.sqrt(2.0) * x1 * x2 + 1e-12)
    tau_double_prime = (M * R) / (J + 1e-12)
    tau = np.sqrt(tau_prime**2 + 2.0 * tau_prime * tau_double_prime * (x2 / (2.0 * R + 1e-12)) + tau_double_prime**2)
    sigma = (6.0 * P * L) / (x4 * (x3**2) + 1e-12)
    delta = (6.0 * P * (L**3)) / (E * x4 * (x3**3) + 1e-12)
    P_c = (4.013 * E * np.sqrt((x3**2 * x4**6) / 36.0) / (L**2)) * (1.0 - (x3 / (2.0 * L)) * np.sqrt(E / (4.0 * G)))
    
    g1 = tau - tau_max
    g2 = sigma - sigma_max
    g3 = x1 - x4
    g4 = 0.10471 * (x1**2) + 0.04811 * x3 * x4 * (14.0 + x2) - 5.0
    g5 = 0.125 - x1
    g6 = delta - delta_max
    g7 = P - P_c
    
    penalty = sum(max(0.0, g)**2 for g in [g1, g2, g3, g4, g5, g6, g7]) * 1e4
    return f + penalty

def tension_spring(x: np.ndarray) -> float:
    # x = [d, D, N]
    x1, x2, x3 = x[0], x[1], x[2]
    f = (x3 + 2.0) * x2 * (x1**2)
    
    g1 = 1.0 - (x2**3 * x3) / (71785.0 * (x1**4) + 1e-12)
    g2 = (4.0 * (x2**2) - x1 * x2) / (12566.0 * (x2 * x1**3 - x1**4) + 1e-12) + 1.0 / (5108.0 * (x1**2) + 1e-12) - 1.0
    g3 = 1.0 - (140.45 * x1) / (x2**2 * x3 + 1e-12)
    g4 = (x1 + x2) / 1.5 - 1.0
    
    penalty = sum(max(0.0, g)**2 for g in [g1, g2, g3, g4]) * 1e4
    return f + penalty

def speed_reducer(x: np.ndarray) -> float:
    # x = [b, m, z, l1, l2, d1, d2]
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    f = (0.7854 * x1 * (x2**2) * (3.3333 * (x3**2) + 14.9334 * x3 - 43.0934)
         - 1.508 * x1 * (x6**2 + x7**2)
         + 7.4777 * (x6**3 + x7**3)
         + 0.7854 * (x4 * (x6**2) + x5 * (x7**2)))
    
    g1 = 27.0 / (x1 * (x2**2) * x3 + 1e-12) - 1.0
    g2 = 397.5 / (x1 * (x2**2) * (x3**2) + 1e-12) - 1.0
    g3 = 1.93 * (x4**3) / (x2 * x3 * (x6**4) + 1e-12) - 1.0
    g4 = 1.93 * (x5**3) / (x2 * x3 * (x7**4) + 1e-12) - 1.0
    g5 = np.sqrt((745.0 * x4 / (x2 * x3 + 1e-12))**2 + 16.9e6) / (110.0 * (x6**3) + 1e-12) - 1.0
    g6 = np.sqrt((745.0 * x5 / (x2 * x3 + 1e-12))**2 + 157.5e6) / (85.0 * (x7**3) + 1e-12) - 1.0
    g7 = (x2 * x3) / 40.0 - 1.0
    g8 = (5.0 * x2) / (x1 + 1e-12) - 1.0
    g9 = x1 / (12.0 * x2 + 1e-12) - 1.0
    g10 = (1.5 * x6 + 1.9) / (x4 + 1e-12) - 1.0
    g11 = (1.1 * x7 + 1.9) / (x5 + 1e-12) - 1.0
    
    penalty = sum(max(0.0, g)**2 for g in [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]) * 1e4
    return f + penalty

def gear_train(x: np.ndarray) -> float:
    # x = [Td, Tb, Ta, Tf] - integer teeth [12, 60]
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    ratio = (x1 * x2) / (x3 * x4 + 1e-12)
    return (1.0 / 6.931 - ratio)**2

def three_bar_truss(x: np.ndarray) -> float:
    # x = [A1, A2]
    x1, x2 = x[0], x[1]
    l = 100.0
    P = 2.0
    sigma = 2.0
    f = (2.0 * np.sqrt(2.0) * x1 + x2) * l
    
    g1 = (np.sqrt(2.0) * x1 + x2) / (np.sqrt(2.0) * x1**2 + 2.0 * x1 * x2 + 1e-12) * P - sigma
    g2 = x2 / (np.sqrt(2.0) * x1**2 + 2.0 * x1 * x2 + 1e-12) * P - sigma
    g3 = 1.0 / (np.sqrt(2.0) * x2 + x1 + 1e-12) * P - sigma
    
    penalty = sum(max(0.0, g)**2 for g in [g1, g2, g3]) * 1e4
    return f + penalty

def cantilever_beam(x: np.ndarray) -> float:
    # x = [x1, x2, x3, x4, x5]
    f = 0.0624 * sum(x)
    # Deflection constraint: g(x) = (61/x1^3 + 37/x2^3 + 19/x3^3 + 7/x4^3 + 1/x5^3) <= 1.0
    g = sum([61.0/(x[0]**3), 37.0/(x[1]**3), 19.0/(x[2]**3), 7.0/(x[3]**3), 1.0/(x[4]**3)]) - 1.0
    penalty = max(0.0, g)**2 * 1e4
    return f + penalty


RWOP_PROBLEMS = [
    ("Pressure Vessel Design", pressure_vessel, [0.0625, 0.0625, 10.0, 10.0], [99.0, 99.0, 200.0, 240.0], 4, 5885.33, "Cost ($)"),
    ("Welded Beam Design", welded_beam, [0.1, 0.1, 0.1, 0.1], [2.0, 10.0, 10.0, 2.0], 4, 1.7248, "Cost ($)"),
    ("Tension/Compression Spring", tension_spring, [0.05, 0.25, 2.0], [2.0, 1.3, 15.0], 3, 0.012665, "Weight (lb)"),
    ("Speed Reducer (Gearbox)", speed_reducer, [2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0], [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5], 7, 2996.34, "Weight (kg)"),
    ("Gear Train Design", gear_train, [12.0, 12.0, 12.0, 12.0], [60.0, 60.0, 60.0, 60.0], 4, 0.0, "Ratio Error"),
    ("Three-Bar Truss Design", three_bar_truss, [0.0, 0.0], [1.0, 1.0], 2, 263.8958, "Volume (cm^3)"),
    ("Cantilever Beam Design", cantilever_beam, [0.01]*5, [100.0]*5, 5, 1.3399, "Volume"),
]

def _eval_rwop_task(task_args: Dict[str, Any]) -> Dict[str, Any]:
    name, func, lb, ub, dim, best_known, unit, num_runs, max_iter = (
        task_args["name"], task_args["func"], task_args["lb"], task_args["ub"],
        task_args["dim"], task_args["best_known"], task_args["unit"],
        task_args["num_runs"], task_args["max_iter"]
    )
    
    values = []
    times = []
    
    for r in range(num_runs):
        np.random.seed(r * 1000 + dim * 31 + 42)
        t0 = time.perf_counter()
        
        res = ophiocordyceps(
            n_ants=40,
            n_dims=dim,
            lower_bound=lb,
            upper_bound=ub,
            fitness=func,
            minimization=True,
            max_iter=max_iter,
            verbose=False
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        values.append(res.fitness if res.fitness is not None else np.nan)
        
    return {
        "Problem": name,
        "Dim": dim,
        "Unit": unit,
        "Best_Known_Ref": best_known,
        "OOA_Best": float(np.nanmin(values)),
        "OOA_Mean": float(np.nanmean(values)),
        "OOA_Std": float(np.nanstd(values)),
        "Mean_Time_s": float(np.mean(times))
    }

def run_engineering_suite(num_runs: int = 5, max_iter: int = 250, csv_out: str = "results/engineering_ooa_results.csv"):
    cpu_info = get_cpu_info()
    workers = max(1, cpu_info["cores_logical"] - 1)
    
    tasks = []
    for name, func, lb, ub, dim, bk, unit in RWOP_PROBLEMS:
        tasks.append({
            "name": name, "func": func, "lb": lb, "ub": ub, "dim": dim,
            "best_known": bk, "unit": unit, "num_runs": num_runs, "max_iter": max_iter
        })
        
    print("=" * 85)
    print("  REAL-WORLD ENGINEERING OPTIMIZATION PROBLEMS (CEC RWOP)")
    print(f"  Available CPU Workers: {workers} | Tasks: {len(tasks)}")
    print("=" * 85)
    
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_eval_rwop_task, tasks))
    total_time = time.perf_counter() - t0
    
    df = pd.DataFrame(results)
    for _, r in df.iterrows():
        print(f"• {r['Problem']:<28} ({r['Dim']}D): OOA Best = {r['OOA_Best']:<10.4f} (Mean: {r['OOA_Mean']:<10.4f}) | Best Ref: {r['Best_Known_Ref']:<10.4f} [{r['Unit']}] ({r['Mean_Time_s']:.2f}s)")
        
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    df.to_csv(csv_out, index=False)
    print("=" * 85)
    print(f"Engineering Benchmark Completed in {total_time:.2f}s! Saved to: {csv_out}")
    print("=" * 85)
    return df

if __name__ == "__main__":
    run_engineering_suite()
