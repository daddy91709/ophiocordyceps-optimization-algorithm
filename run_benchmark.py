"""
Parallel Multi-Core Benchmark Suite Runner for Ophiocordyceps Optimization Algorithm
Supports multiprocessing across all available CPU cores.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.ophiocordyceps import ophiocordyceps
import src.benchmark as bm
from src.device import get_cpu_info, detect_gpu


def run_single_benchmark_task(task_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Esegue un singolo test di benchmark (una combinazione funzione + dimensione + N run).
    Funzione standalone per consentire la serializzazione nei processi worker di multiprocessing.
    """
    func_name = task_args["func_name"]
    dim = task_args["dim"]
    num_runs = task_args["num_runs"]
    n_ants = task_args["n_ants"]
    max_iter = task_args["max_iter"]
    n_eval_workers = task_args.get("n_eval_workers", 1)

    fitness_functions = bm.functions_map()
    valori_ottimi = bm.best_values_map()
    limiti_funzioni = bm.bounds_map()

    func = fitness_functions[func_name]
    lim_inf, lim_sup = limiti_funzioni.get(func_name, (-10, 10))
    if isinstance(lim_inf, list):
        limiti_inf = lim_inf
        limiti_sup = lim_sup
    else:
        limiti_inf = [lim_inf] * dim
        limiti_sup = [lim_sup] * dim

    best_values = []
    run_times = []

    for run_idx in range(num_runs):
        np.random.seed(run_idx * 1000 + dim * 10 + 42)
        t0 = time.perf_counter()

        best_ant = ophiocordyceps(
            n_ants=n_ants,
            n_dims=dim,
            lower_bound=limiti_inf,
            upper_bound=limiti_sup,
            fitness=func,
            minimization=True,
            use_best_guidance=True,
            visualization=False,
            stochastic=False,
            verbose=False,
            dispersion_radius=0.2,
            base_infection_prob=0.1,
            base_death_prob=0.01,
            ant_step_size=0.01,
            max_iter=max_iter,
            convergence_patience=7,
            convergence_tolerance=1e-3,
            learning_rate=0.1,
            best_influence=0.05,
            n_workers=n_eval_workers
        )

        elapsed = time.perf_counter() - t0
        run_times.append(elapsed)
        best_values.append(best_ant.fitness if best_ant.fitness is not None else np.nan)

    ottimo = 25 + dim * (-6) if (valori_ottimi.get(func_name) is None and func_name == "StepInt") else valori_ottimi.get(func_name, 0)
    if ottimo is not None:
        rmse = float(np.sqrt(np.mean([(val - ottimo)**2 for val in best_values if not np.isnan(val)])))
    else:
        rmse = np.nan

    return {
        "Funzione": func_name,
        "Dimensioni": dim,
        "Best": float(np.nanmin(best_values)),
        "Mean": float(np.nanmean(best_values)),
        "Std": float(np.nanstd(best_values)),
        "Worst": float(np.nanmax(best_values)),
        "RMSE": rmse,
        "Tempo medio (s)": float(np.mean(run_times)),
        "Valore ottimo teorico": ottimo,
        "Total_Time": sum(run_times)
    }


def run_parallel_benchmarks(
    functions_to_test: Optional[List[str]] = None,
    dimensions_to_test: Optional[List[int]] = None,
    num_runs: int = 5,
    n_ants: int = 40,
    max_iter: int = 200,
    n_jobs: int = -1,
    csv_path: str = "results/risultati.csv",
    save_csv: bool = True
) -> pd.DataFrame:
    """
    Esegue la suite di benchmark in parallelo su CPU.
    """
    cpu_info = get_cpu_info()
    logical_cores = cpu_info["cores_logical"]

    if n_jobs <= 0:
        max_workers = max(1, logical_cores - 1)
    else:
        max_workers = min(n_jobs, logical_cores)

    fitness_functions = bm.functions_map()
    fixed_dim_functions = bm.fixed_dim_functions_map()

    if dimensions_to_test is None:
        dimensions_to_test = [2, 10, 30]

    if functions_to_test is None:
        funzioni = list(fitness_functions.keys())
    else:
        funzioni = [f for f in functions_to_test if f in fitness_functions]

    # Prepara tutti i task indipendenti da eseguire in parallelo
    tasks = []
    for name in funzioni:
        for dim in dimensions_to_test:
            if name == "Colville" and dim == 2:
                dim_effettiva = 4
            elif name == "Colville" and dim > 2:
                continue
            else:
                dim_effettiva = dim

            if name in fixed_dim_functions and fixed_dim_functions[name] != dim_effettiva:
                continue

            tasks.append({
                "func_name": name,
                "dim": dim_effettiva,
                "num_runs": num_runs,
                "n_ants": n_ants,
                "max_iter": max_iter,
                "n_eval_workers": 1
            })

    print("=" * 70)
    print(f"  BENCHMARK SUITE - PARALLEL EXECUTION (CPU: {logical_cores} Cores)")
    print(f"  Workers paralleli attivi: {max_workers}")
    print(f"  Task totali da eseguire: {len(tasks)} ({len(funzioni)} funzioni x {num_runs} runs)")
    print("=" * 70)

    t_start_total = time.perf_counter()
    results = []

    if max_workers == 1 or len(tasks) == 1:
        for idx, task in enumerate(tasks):
            print(f"[{idx+1}/{len(tasks)}] Esecuzione {task['func_name']} ({task['dim']}D)...", end=" ", flush=True)
            res = run_single_benchmark_task(task)
            results.append(res)
            print(f"Fatto! Best: {res['Best']:.4e} (T: {res['Tempo medio (s)']:.2f}s)")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(run_single_benchmark_task, t): t for t in tasks}
            completed_count = 0
            
            for future in as_completed(future_to_task):
                completed_count += 1
                task = future_to_task[future]
                try:
                    res = future.result()
                    results.append(res)
                    print(f"[{completed_count}/{len(tasks)}] [OK] {res['Funzione']} ({res['Dimensioni']}D) -> Best: {res['Best']:.4e}, Mean: {res['Mean']:.4e} (Tempo: {res['Tempo medio (s)']:.2f}s)")
                except Exception as exc:
                    print(f"[{completed_count}/{len(tasks)}] [ERR] Errore in {task['func_name']} ({task['dim']}D): {exc}")

    t_total_elapsed = time.perf_counter() - t_start_total
    total_cpu_work_time = sum(r.get("Total_Time", 0) for r in results)
    speedup = total_cpu_work_time / t_total_elapsed if t_total_elapsed > 0 else 1.0

    df_results = pd.DataFrame(results)
    
    if "Total_Time" in df_results.columns:
        df_results = df_results.drop(columns=["Total_Time"])

    if save_csv:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_results.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\nRisultati salvati con successo in: {csv_path}")

    print("\n" + "=" * 70)
    print(f"  RIASSUNTO PRESTAZIONI PARALLELE:")
    print(f"  * Tempo Reale (Wall-Clock): {t_total_elapsed:.2f} secondi")
    print(f"  * Tempo CPU Sequenziale Stimato: {total_cpu_work_time:.2f} secondi")
    print(f"  * Speedup Multi-Core Effettivo: {speedup:.2f}x (sfruttando {max_workers} worker CPU)")
    print("=" * 70)

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esecuzione Benchmark Parallelo su CPU Multi-Core")
    parser.add_argument("--runs", type=int, default=5, help="Numero di run per configurazione (default: 5)")
    parser.add_argument("--ants", type=int, default=40, help="Numero di formiche (default: 40)")
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 10], help="Dimensioni da testare (default: 2 10)")
    parser.add_argument("--funcs", nargs="+", type=str, default=None, help="Lista funzioni (default: tutte)")
    parser.add_argument("--jobs", "-j", type=int, default=-1, help="Numero di core/processi paralleli (-1 per tutti i core disponibili)")
    parser.add_argument("--output", type=str, default="results/risultati.csv", help="Percorso file CSV di output")

    args = parser.parse_args()

    run_parallel_benchmarks(
        functions_to_test=args.funcs,
        dimensions_to_test=args.dims,
        num_runs=args.runs,
        n_ants=args.ants,
        n_jobs=args.jobs,
        csv_path=args.output
    )
