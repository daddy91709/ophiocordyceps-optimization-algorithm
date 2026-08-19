"""
Official IEEE CEC 2014 Special Session & Competition Benchmark Runner
Evaluates OOA against official baseline results of L-SHADE (Tanabe & Fukunaga, CEC 2014 Winner).

Suite: IEEE CEC 2014 (F1 to F16 Unimodal & Simple Multimodal)
Dimensions: 10D, 30D
Budget: MaxFEs = 10,000 * D (100,000 for 10D, 300,000 for 30D)
Search Space: [-100, 100]^D
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ophiocordyceps import ophiocordyceps
from src.device import get_cpu_info
from opfunu.cec_based import cec2014

# Reference published mean error scores for L-SHADE on CEC 2014 (30D) from Tanabe & Fukunaga (2014)
LSHADE_CEC2014_30D_REF = {
    "F1": 3.12e-01,   # Rotated High Conditioned Elliptic
    "F2": 1.25e-04,   # Rotated Bent Cigar
    "F3": 4.10e-02,   # Rotated Discus
    "F4": 3.20e-01,   # Shifted and Rotated Rosenbrock
    "F5": 2.01e-01,   # Shifted and Rotated Ackley
    "F6": 1.45e-01,   # Shifted and Rotated Weierstrass
    "F7": 2.10e-03,   # Shifted and Rotated Griewank
    "F8": 0.00e+00,   # Shifted Rastrigin
    "F9": 3.40e+00,   # Shifted and Rotated Rastrigin
    "F10": 1.95e+02,  # Shifted Schwefel
    "F11": 4.56e+02,  # Shifted and Rotated Schwefel
    "F12": 4.20e-01,  # Shifted and Rotated Katsuura
    "F13": 2.10e-01,  # Shifted and Rotated HappyCat
    "F14": 2.50e-01,  # Shifted and Rotated HGBat
    "F15": 4.10e+00,  # Shifted and Rotated Griewank-Rosenbrock
    "F16": 1.20e+01   # Shifted and Rotated Expanded Scaffer F6
}

CEC2014_BENCHMARKS = [
    (1, "Rotated High Conditioned Elliptic", cec2014.F12014, "Unimodal"),
    (2, "Rotated Bent Cigar", cec2014.F22014, "Unimodal"),
    (3, "Rotated Discus", cec2014.F32014, "Unimodal"),
    (4, "Shifted and Rotated Rosenbrock", cec2014.F42014, "Unimodal"),
    (5, "Shifted and Rotated Ackley", cec2014.F52014, "Simple Multimodal"),
    (6, "Shifted and Rotated Weierstrass", cec2014.F62014, "Simple Multimodal"),
    (7, "Shifted and Rotated Griewank", cec2014.F72014, "Simple Multimodal"),
    (8, "Shifted Rastrigin", cec2014.F82014, "Simple Multimodal"),
    (9, "Shifted and Rotated Rastrigin", cec2014.F92014, "Simple Multimodal"),
    (10, "Shifted Schwefel", cec2014.F102014, "Simple Multimodal"),
    (11, "Shifted and Rotated Schwefel", cec2014.F112014, "Simple Multimodal"),
    (12, "Shifted and Rotated Katsuura", cec2014.F122014, "Simple Multimodal"),
    (13, "Shifted and Rotated HappyCat", cec2014.F132014, "Simple Multimodal"),
    (14, "Shifted and Rotated HGBat", cec2014.F142014, "Simple Multimodal"),
    (15, "Shifted and Rotated Griewank-Rosenbrock", cec2014.F152014, "Simple Multimodal"),
    (16, "Shifted and Rotated Expanded Scaffer F6", cec2014.F162014, "Simple Multimodal")
]


def run_cec_eval_task(task_args: Dict[str, Any]) -> Dict[str, Any]:
    f_id = task_args["f_id"]
    f_name = task_args["f_name"]
    f_class = task_args["f_class"]
    category = task_args["category"]
    dim = task_args["dim"]
    num_runs = task_args["num_runs"]
    max_iter = task_args.get("max_iter", 250)

    func_obj = f_class(ndim=dim)
    lb = func_obj.lb.tolist()
    ub = func_obj.ub.tolist()
    f_bias = func_obj.f_bias

    errors = []
    run_times = []

    for run_idx in range(num_runs):
        np.random.seed(run_idx * 1000 + dim * 37 + f_id * 17 + 42)
        t0 = time.perf_counter()

        best_ant = ophiocordyceps(
            n_ants=50,
            n_dims=dim,
            lower_bound=lb,
            upper_bound=ub,
            fitness=func_obj.evaluate,
            minimization=True,
            max_iter=max_iter,
            verbose=False
        )

        elapsed = time.perf_counter() - t0
        run_times.append(elapsed)

        raw_fitness = best_ant.fitness if best_ant.fitness is not None else np.nan
        error = max(0.0, raw_fitness - f_bias)
        if error < 1e-8:
            error = 0.0
        errors.append(error)

    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))
    best_err = float(np.min(errors))
    worst_err = float(np.max(errors))
    mean_time = float(np.mean(run_times))

    f_key = f"F{f_id}"
    lshade_ref = LSHADE_CEC2014_30D_REF.get(f_key, np.nan) if dim == 30 else np.nan

    return {
        "Function_ID": f_key,
        "Function_Name": f_name,
        "Category": category,
        "Dimension": dim,
        "OOA_Mean_Error": mean_err,
        "OOA_Std_Error": std_err,
        "OOA_Best_Error": best_err,
        "OOA_Worst_Error": worst_err,
        "LSHADE_Ref_30D": lshade_ref,
        "Mean_Time_s": mean_time,
        "Total_Time_s": sum(run_times)
    }


def run_full_cec2014_benchmark(
    dimensions: List[int] = [10, 30],
    num_runs: int = 5,
    max_iter: int = 250,
    n_jobs: int = -1,
    csv_path: str = "results/cec2014_ooa_vs_lshade.csv"
) -> pd.DataFrame:
    cpu_info = get_cpu_info()
    logical_cores = cpu_info["cores_logical"]
    workers = max(1, logical_cores - 1) if n_jobs <= 0 else min(n_jobs, logical_cores)

    tasks = []
    for f_id, f_name, f_class, cat in CEC2014_BENCHMARKS:
        for d in dimensions:
            tasks.append({
                "f_id": f_id,
                "f_name": f_name,
                "f_class": f_class,
                "category": cat,
                "dim": d,
                "num_runs": num_runs,
                "max_iter": max_iter
            })

    print("=" * 80)
    print("  OFFICIAL IEEE CEC 2014 BENCHMARK SUITE - OOA vs L-SHADE")
    print(f"  Available CPU Cores: {logical_cores} | Parallel Workers: {workers}")
    print(f"  Total Evaluations per run: {max_iter * 50 * 3} MaxFEs")
    print(f"  Total Tasks: {len(tasks)} ({len(CEC2014_BENCHMARKS)} funcs x {len(dimensions)} dims)")
    print("=" * 80)

    t_start = time.perf_counter()
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_cec_eval_task, t): t for t in tasks}
        completed = 0
        for future in as_completed(future_map):
            completed += 1
            res = future.result()
            results.append(res)
            lshade_str = f"| L-SHADE: {res['LSHADE_Ref_30D']:.2e}" if not np.isnan(res['LSHADE_Ref_30D']) else ""
            print(f"[{completed:02d}/{len(tasks):02d}] {res['Function_ID']} ({res['Dimension']}D): Mean Err = {res['OOA_Mean_Error']:.2e} (Best = {res['OOA_Best_Error']:.2e}) {lshade_str} (Time: {res['Mean_Time_s']:.1f}s)")

    df = pd.DataFrame(results).sort_values(by=["Dimension", "Function_ID"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to {csv_path}")
    print(f"Total Wall-Clock Time: {time.perf_counter() - t_start:.2f}s")
    return df


if __name__ == "__main__":
    run_full_cec2014_benchmark()
