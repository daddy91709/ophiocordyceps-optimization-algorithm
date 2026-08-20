"""
Official Modern Benchmark Suite: IEEE CEC 2022 Special Session & Competition
Evaluates OOA (Meta-Hyphal Architecture) across all 12 official competition functions:
- F1: Shifted & Full Rotated Zakharov (Unimodal)
- F2: Shifted & Rotated Rosenbrock (Unimodal)
- F3: Shifted & Full Rotated Expanded Schaffer F7 (Multimodal)
- F4: Shifted & Rotated Non-Continuous Rastrigin (Multimodal)
- F5: Shifted & Rotated Levy (Multimodal)
- F6: Hybrid Function 1 (Hybrid)
- F7: Hybrid Function 2 (Hybrid)
- F8: Hybrid Function 3 (Hybrid)
- F9: Composition Function 1 (Composition)
- F10: Composition Function 2 (Composition)
- F11: Composition Function 3 (Composition)
- F12: Composition Function 4 (Composition)

Dimensions: 10D and 20D (Official CEC 2022 competition dimensions)
Search Space: [-100, 100]^D
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ophiocordyceps import ophiocordyceps
from src.device import get_cpu_info
from opfunu.cec_based import cec2022

# Mappatura delle 12 funzioni ufficiali CEC 2022
CEC2022_CLASSES = [
    (1, "F1: Shifted & Full Rotated Zakharov", cec2022.F12022, "Unimodal"),
    (2, "F2: Shifted & Rotated Rosenbrock", cec2022.F22022, "Unimodal"),
    (3, "F3: Shifted & Full Rotated Exp Schaffer F7", cec2022.F32022, "Basic Multimodal"),
    (4, "F4: Shifted & Rotated Non-Continuous Rastrigin", cec2022.F42022, "Basic Multimodal"),
    (5, "F5: Shifted & Rotated Levy", cec2022.F52022, "Basic Multimodal"),
    (6, "F6: Hybrid Function 1", cec2022.F62022, "Hybrid"),
    (7, "F7: Hybrid Function 2", cec2022.F72022, "Hybrid"),
    (8, "F8: Hybrid Function 3", cec2022.F82022, "Hybrid"),
    (9, "F9: Composition Function 1", cec2022.F92022, "Composition"),
    (10, "F10: Composition Function 2", cec2022.F102022, "Composition"),
    (11, "F11: Composition Function 3", cec2022.F112022, "Composition"),
    (12, "F12: Composition Function 4", cec2022.F122022, "Composition"),
]


def _run_single_cec2022_task(args: Dict[str, Any]) -> Dict[str, Any]:
    f_id = args["f_id"]
    f_name = args["f_name"]
    f_cls = args["f_cls"]
    f_cat = args["f_cat"]
    dim = args["dim"]
    num_runs = args["num_runs"]
    max_iter = args["max_iter"]

    func_obj = f_cls(ndim=dim)
    lb = func_obj.lb.tolist()
    ub = func_obj.ub.tolist()
    f_bias = func_obj.f_bias

    errors = []
    times = []

    for r in range(num_runs):
        np.random.seed(r * 1000 + dim * 10 + f_id * 7)
        t0 = time.perf_counter()

        best_ant = ophiocordyceps(
            n_ants=40,
            n_dims=dim,
            lower_bound=lb,
            upper_bound=ub,
            fitness=func_obj.evaluate,
            minimization=True,
            use_best_guidance=True,
            visualization=False,
            verbose=False,
            max_iter=max_iter
        )

        elapsed = time.perf_counter() - t0
        raw_fitness = best_ant.fitness if best_ant.fitness is not None else np.nan
        err = max(0.0, raw_fitness - f_bias)
        if err < 1e-8:
            err = 0.0
        errors.append(err)
        times.append(elapsed)

    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))
    best_err = float(np.min(errors))
    worst_err = float(np.max(errors))
    mean_time = float(np.mean(times))

    return {
        "Function_ID": f"F{f_id}",
        "Name": f_name,
        "Category": f_cat,
        "Dimension": dim,
        "OOA_Best_Error": best_err,
        "OOA_Mean_Error": mean_err,
        "OOA_Std_Error": std_err,
        "Worst_Error": worst_err,
        "Mean_Time_s": mean_time
    }


def run_cec2022_benchmark(dims: List[int] = [10, 20], num_runs: int = 5, max_iter: int = 200, n_jobs: int = -1, csv_out: str = "results/cec2022_ooa_results.csv"):
    cpu_info = get_cpu_info()
    workers = max(1, cpu_info["cores_logical"] - 1) if n_jobs <= 0 else n_jobs

    tasks = []
    for f_id, f_name, f_cls, f_cat in CEC2022_CLASSES:
        for d in dims:
            tasks.append({
                "f_id": f_id,
                "f_name": f_name,
                "f_cls": f_cls,
                "f_cat": f_cat,
                "dim": d,
                "num_runs": num_runs,
                "max_iter": max_iter
            })

    print("=" * 80)
    print(f"  OFFICIAL IEEE CEC 2022 MODERN BENCHMARK SUITE - OOA EVALUATION")
    print(f"  Tasks totali: {len(tasks)} (F1-F12 x {dims}D x {num_runs} runs)")
    print(f"  CPU Workers: {workers}")
    print("=" * 80)

    t_start = time.perf_counter()
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single_cec2022_task, t): t for t in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            results.append(res)
            print(f"[{completed:02d}/{len(tasks):02d}] {res['Function_ID']} ({res['Dimension']}D) [{res['Category']}]: Mean Err = {res['OOA_Mean_Error']:<10.4e} | Best: {res['OOA_Best_Error']:<10.4e} (Time: {res['Mean_Time_s']:.2f}s)")

    t_total = time.perf_counter() - t_start
    df = pd.DataFrame(results).sort_values(by=["Dimension", "Function_ID"]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    df.to_csv(csv_out, index=False, encoding="utf-8")
    print("=" * 80)
    print(f"Benchmark completato in {t_total:.2f}s! Risultati salvati in: {csv_out}")
    print("=" * 80)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IEEE CEC 2022 Benchmark Suite")
    parser.add_argument("--dims", nargs="+", type=int, default=[10, 20], help="Dimensions to test (default: 10 20)")
    parser.add_argument("--runs", type=int, default=5, help="Runs per test (default: 5)")
    parser.add_argument("--iter", type=int, default=200, help="Max iterations (default: 200)")
    parser.add_argument("--jobs", "-j", type=int, default=-1, help="Parallel worker jobs")
    parser.add_argument("--output", type=str, default="results/cec2022_ooa_results.csv", help="CSV output")
    args = parser.parse_args()

    run_cec2022_benchmark(dims=args.dims, num_runs=args.runs, max_iter=args.iter, n_jobs=args.jobs, csv_out=args.output)
