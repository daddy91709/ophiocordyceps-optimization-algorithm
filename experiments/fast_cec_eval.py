"""
Fast validation harness for CEC 2014 during iterative optimization.
Evaluates representative subset of CEC 2014 (F1, F2, F3, F4, F5, F8, F10, F12) in 10D and 30D.
Executes in parallel in < 20 seconds.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ophiocordyceps import ophiocordyceps
from src.device import get_cpu_info
from opfunu.cec_based import cec2014

FAST_CEC_FUNCS = [
    (1, "F1: Rotated Elliptic (Unimodal 10^6 Cond)", cec2014.F12014, 0.312),
    (2, "F2: Rotated Bent Cigar (Unimodal)", cec2014.F22014, 0.000125),
    (3, "F3: Rotated Discus (Unimodal)", cec2014.F32014, 0.041),
    (4, "F4: Shifted & Rotated Rosenbrock", cec2014.F42014, 0.32),
    (5, "F5: Shifted & Rotated Ackley", cec2014.F52014, 0.201),
    (8, "F8: Shifted Rastrigin", cec2014.F82014, 0.0),
    (10, "F10: Shifted Schwefel", cec2014.F102014, 195.0),
    (12, "F12: Shifted & Rotated Katsuura", cec2014.F122014, 0.42),
]


def _eval_fast_task(args: Dict[str, Any]) -> Dict[str, Any]:
    f_id, f_name, f_cls, lshade_ref, dim, num_runs, max_iter = (
        args["f_id"], args["f_name"], args["f_cls"], args["lshade_ref"],
        args["dim"], args["num_runs"], args["max_iter"]
    )
    func_obj = f_cls(ndim=dim)
    lb = func_obj.lb.tolist()
    ub = func_obj.ub.tolist()
    f_bias = func_obj.f_bias

    errors = []
    for r in range(num_runs):
        np.random.seed(r * 500 + dim * 13 + f_id * 19)
        best_ant = ophiocordyceps(
            n_ants=50, n_dims=dim, lower_bound=lb, upper_bound=ub,
            fitness=func_obj.evaluate, minimization=True, max_iter=max_iter
        )
        raw_fit = best_ant.fitness if best_ant.fitness is not None else np.nan
        err = max(0.0, raw_fit - f_bias)
        if err < 1e-8:
            err = 0.0
        errors.append(err)

    mean_err = float(np.mean(errors))
    best_err = float(np.min(errors))
    return {
        "f_id": f"F{f_id}",
        "name": f_name,
        "dim": dim,
        "mean_err": mean_err,
        "best_err": best_err,
        "lshade_ref": lshade_ref if dim == 30 else np.nan
    }


def evaluate_fast_cec(num_runs: int = 3, max_iter: int = 200, dims: List[int] = [10, 30]) -> pd.DataFrame:
    workers = max(1, get_cpu_info()["cores_logical"] - 1)
    tasks = []
    for f_id, f_name, f_cls, lshade_ref in FAST_CEC_FUNCS:
        for d in dims:
            tasks.append({
                "f_id": f_id, "f_name": f_name, "f_cls": f_cls, "lshade_ref": lshade_ref,
                "dim": d, "num_runs": num_runs, "max_iter": max_iter
            })

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_eval_fast_task, tasks))
    t_tot = time.perf_counter() - t0

    df = pd.DataFrame(results).sort_values(by=["dim", "f_id"]).reset_index(drop=True)
    print(f"\n--- Fast CEC Evaluation ({t_tot:.2f}s) ---")
    for _, row in df.iterrows():
        ref_s = f"| L-SHADE Ref: {row['lshade_ref']:.2e}" if not np.isnan(row['lshade_ref']) else ""
        print(f"[{row['f_id']}] {row['name']:<35} ({row['dim']}D) -> Mean Err: {row['mean_err']:<10.2e} (Best: {row['best_err']:<10.2e}) {ref_s}")
    return df


if __name__ == "__main__":
    evaluate_fast_cec()
