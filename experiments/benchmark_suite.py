"""
Standardized Rigorous Multi-Core Benchmark Suite for OOA Iterative Optimization.
Runs repeatable comparisons across diverse benchmark topologies in parallel.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ophiocordyceps import ophiocordyceps
import src.benchmark as bm
from src.device import get_cpu_info


BENCHMARK_FUNCS = [
    ("Sphere", bm.sphere, (-5.12, 5.12), 0.0, "Unimodal Separable"),
    ("Ackley", bm.ackley, (-5.0, 5.0), 0.0, "Multimodal Non-Separable"),
    ("Alpine1", bm.alpine1, (-10.0, 10.0), 0.0, "Multimodal Separable"),
    ("Bohachevsky", bm.bohachevsky, (-100.0, 100.0), 0.0, "Multimodal Separable (2D only)"),
    ("Beale", bm.beale, (-10.0, 10.0), 0.0, "Unimodal Non-Separable (2D only)"),
    ("Booth", bm.booth, (-10.0, 10.0), 0.0, "Multimodal Separable (2D only)"),
    ("Zakharov", bm.zakharov, (-5.0, 10.0), 0.0, "Unimodal Non-Separable"),
    ("Chung Reynolds", bm.chung_reynolds, (-100.0, 100.0), 0.0, "Unimodal Non-Separable"),
]

FIXED_2D = {"Bohachevsky", "Beale", "Booth"}


def _run_single_task(task_args: Dict[str, Any]) -> Dict[str, Any]:
    name = task_args["name"]
    d = task_args["dim"]
    bounds = task_args["bounds"]
    opt_val = task_args["opt_val"]
    cat = task_args["cat"]
    num_runs = task_args["num_runs"]
    max_iter = task_args["max_iter"]
    n_ants = task_args["n_ants"]
    round_name = task_args["round_name"]

    fitness_functions = bm.functions_map()
    func = fitness_functions[name]
    lb = [bounds[0]] * d
    ub = [bounds[1]] * d

    run_fitnesses = []
    run_times = []

    for r in range(num_runs):
        np.random.seed(r * 1000 + d * 10 + 42)
        t0 = time.perf_counter()
        
        best_ant = ophiocordyceps(
            n_ants=n_ants,
            n_dims=d,
            lower_bound=lb,
            upper_bound=ub,
            fitness=func,
            minimization=True,
            use_best_guidance=True,
            visualization=False,
            verbose=False,
            max_iter=max_iter
        )
        
        elapsed = time.perf_counter() - t0
        run_times.append(elapsed)
        run_fitnesses.append(best_ant.fitness if (best_ant is not None and best_ant.fitness is not None) else np.nan)

    mean_fit = float(np.nanmean(run_fitnesses))
    std_fit = float(np.nanstd(run_fitnesses))
    best_fit = float(np.nanmin(run_fitnesses))
    worst_fit = float(np.nanmax(run_fitnesses))
    rmse = float(np.sqrt(np.mean([(f - opt_val)**2 for f in run_fitnesses if not np.isnan(f)])))
    mean_time = float(np.mean(run_times))

    return {
        "Round": round_name,
        "Function": name,
        "Category": cat,
        "Dimension": d,
        "Best": best_fit,
        "Mean": mean_fit,
        "Std": std_fit,
        "Worst": worst_fit,
        "RMSE": rmse,
        "Mean_Time_s": mean_time,
        "Total_Time": sum(run_times)
    }


def run_benchmark_round(round_name: str = "Test Round", 
                        dims: List[int] = [2, 10, 30], 
                        num_runs: int = 10,
                        max_iter: int = 200,
                        n_ants: int = 40,
                        n_jobs: int = -1) -> Dict[str, Any]:
    """
    Esegue un round di benchmark completo in parallelo.
    """
    cpu_info = get_cpu_info()
    logical_cores = cpu_info["cores_logical"]
    max_workers = max(1, logical_cores - 1) if n_jobs <= 0 else min(n_jobs, logical_cores)

    tasks = []
    for name, func, bounds, opt_val, cat in BENCHMARK_FUNCS:
        for d in dims:
            if name in FIXED_2D and d > 2:
                continue
            tasks.append({
                "round_name": round_name,
                "name": name,
                "dim": d,
                "bounds": bounds,
                "opt_val": opt_val,
                "cat": cat,
                "num_runs": num_runs,
                "max_iter": max_iter,
                "n_ants": n_ants
            })

    t_start = time.perf_counter()
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_task, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)

    t_total = time.perf_counter() - t_start
    df = pd.DataFrame(results).sort_values(by=["Function", "Dimension"]).reset_index(drop=True)
    
    overall_mean_rmse = df["RMSE"].mean()
    overall_median_rmse = df["RMSE"].median()
    overall_mean_time = df["Mean_Time_s"].mean()
    
    print(f"\n==================== RESULTS FOR [{round_name}] ====================")
    print(f"Total Wall Time: {t_total:.2f}s | Mean RMSE: {overall_mean_rmse:.4e} | Median RMSE: {overall_median_rmse:.4e}")
    for _, row in df.iterrows():
        print(f"• {row['Function']:<15} ({row['Dimension']:>2}D) -> Mean: {row['Mean']:<12.4e} | Best: {row['Best']:<12.4e} | RMSE: {row['RMSE']:<12.4e}")
    print("====================================================================\n")

    return {
        "round_name": round_name,
        "total_time": t_total,
        "mean_rmse": overall_mean_rmse,
        "median_rmse": overall_median_rmse,
        "mean_time": overall_mean_time,
        "dataframe": df
    }


if __name__ == "__main__":
    res = run_benchmark_round("Baseline Multi-Core Check")
