"""
Benchmark Suite Runner for Ophiocordyceps Optimization Algorithm
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from src.ophiocordyceps import ophiocordyceps
import src.benchmark as bm


def run_benchmarks(
    functions_to_test=None,
    dimensions_to_test=None,
    num_runs=5,
    n_ants=40,
    max_iter=200,
    csv_path="results/risultati.csv",
    verbose=False
):
    """
    Esegue la suite di benchmark salvando i risultati in un file CSV.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    passo_formiche = 0.01
    learning_rate = 0.1
    best_influence = 0.05
    raggio_nuove_formiche = 0.2
    probabilita_infezione = 0.1
    probabilita_morte = 0.01
    tolerance = 1e-3
    patience = 7

    fitness_functions = bm.functions_map()
    fixed_dim_functions = bm.fixed_dim_functions_map()
    valori_ottimi = bm.best_values_map()
    limiti_funzioni = bm.bounds_map()

    if dimensions_to_test is None:
        dimensions_to_test = [2, 10, 30]

    if functions_to_test is None:
        funzioni = list(fitness_functions.keys())
    else:
        funzioni = [f for f in functions_to_test if f in fitness_functions]

    header = [
        "Funzione", "Dimensioni", "Best", "Mean", "Std", "Worst", "RMSE", "Tempo medio (s)", "Valore ottimo teorico"
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")

    print(f"Avvio Benchmark su {len(funzioni)} funzioni e dimensioni {dimensions_to_test} ({num_runs} run ciascuna)...")

    for name in funzioni:
        func = fitness_functions[name]
        for dim in dimensions_to_test:
            if name == "Colville" and dim == 2:
                dim_effettiva = 4
            elif name == "Colville" and dim > 2:
                continue
            else:
                dim_effettiva = dim

            if name in fixed_dim_functions and fixed_dim_functions[name] != dim_effettiva:
                continue

            print(f"-> Testing {name} ({dim_effettiva}D)...", end=" ", flush=True)

            lim_inf, lim_sup = limiti_funzioni.get(name, (-10, 10))
            if isinstance(lim_inf, list):
                limiti_inf = lim_inf
                limiti_sup = lim_sup
            else:
                limiti_inf = [lim_inf] * dim_effettiva
                limiti_sup = [lim_sup] * dim_effettiva

            best_values = []
            run_times = []

            for run in range(num_runs):
                np.random.seed(run)
                start = time.time()

                best_ant = ophiocordyceps(
                    n_ants=n_ants,
                    n_dims=dim_effettiva,
                    lower_bound=limiti_inf,
                    upper_bound=limiti_sup,
                    fitness=func,
                    minimization=True,
                    use_best_guidance=True,
                    visualization=False,
                    stochastic=False,
                    verbose=verbose,
                    dispersion_radius=raggio_nuove_formiche,
                    base_infection_prob=probabilita_infezione,
                    base_death_prob=probabilita_morte,
                    ant_step_size=passo_formiche,
                    max_iter=max_iter,
                    convergence_patience=patience,
                    convergence_tolerance=tolerance,
                    learning_rate=learning_rate,
                    best_influence=best_influence
                )

                elapsed = time.time() - start
                run_times.append(elapsed)
                best_values.append(best_ant.fitness)

            ottimo = 25 + dim_effettiva * (-6) if (valori_ottimi.get(name) is None and name == "StepInt") else valori_ottimi.get(name, 0)
            if ottimo is not None:
                rmse = np.sqrt(np.mean([(val - ottimo)**2 for val in best_values]))
            else:
                rmse = np.nan

            row = [
                name,
                dim_effettiva,
                np.min(best_values),
                np.mean(best_values),
                np.std(best_values),
                np.max(best_values),
                rmse,
                np.mean(run_times),
                ottimo
            ]

            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(",".join(map(str, row)) + "\n")

            print(f"Completato! Best: {np.min(best_values):.4e}, Tempo medio: {np.mean(run_times):.2f}s")

    print(f"\nTutti i benchmark completati. Risultati salvati in: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui i benchmark dell'algoritmo Ophiocordyceps")
    parser.add_argument("--runs", type=int, default=5, help="Numero di run per configurazione (default: 5)")
    parser.add_argument("--ants", type=int, default=40, help="Numero di formiche (default: 40)")
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 10], help="Dimensioni da testare (default: 2 10)")
    parser.add_argument("--funcs", nargs="+", type=str, default=["Ackley", "Sphere", "Alpine1"], help="Funzioni da testare")
    parser.add_argument("--output", type=str, default="results/risultati.csv", help="Percorso del CSV di output")

    args = parser.parse_args()
    run_benchmarks(
        functions_to_test=args.funcs,
        dimensions_to_test=args.dims,
        num_runs=args.runs,
        n_ants=args.ants,
        csv_path=args.output
    )
