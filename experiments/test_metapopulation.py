import os
import sys
import time
import math
import numpy as np
import pandas as pd
from typing import Callable, List, Optional, Tuple, Union, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ophiocordyceps import Ant, dispatch_ants, repair_boundary, forage_healthy_ant
from opfunu.cec_based import cec2014


def ooa_metapopulation(n_dims: int, lower_bound: list, upper_bound: list, fitness: Callable,
                       max_evals: int = 50000) -> Ant:
    """
    Multi-Colony Mycelial Metapopulation with Spore Wind Drift (MM-SWD).
    3 Sub-colonies: Exploiter, Explorer, Bridge.
    """
    lb = np.array(lower_bound, dtype=float)
    ub = np.array(upper_bound, dtype=float)
    span = ub - lb

    num_colonies = 3
    pop_per_colony = max(25, 8 * n_dims)
    n_min_colony = 4
    
    colonies = [
        dispatch_ants(pop_per_colony, n_dims, lb, ub, infection=0.3, death=0.08)
        for _ in range(num_colonies)
    ]
    spore_archive = []
    max_archive = pop_per_colony * 3

    eval_count = 0
    for col in colonies:
        for a in col:
            a.fitness = float(fitness(a.position))
            a.best_fitness = a.fitness
            a.best_position = a.position.copy()
            a.is_infected = (np.random.rand() < 0.65)
            eval_count += 1

    all_ants = [a for col in colonies for a in col]
    global_best = min(all_ants, key=lambda a: a.fitness)

    H_size = 10
    cr_init = np.linspace(0.1, 0.9, H_size)
    f_init = np.linspace(0.3, 0.8, H_size)
    mem_F = [f_init.copy() for _ in range(num_colonies)]
    mem_CR = [cr_init.copy() for _ in range(num_colonies)]
    mem_idx = [0] * num_colonies

    eigen_bases = [np.eye(n_dims) for _ in range(num_colonies)]

    gen = 0
    while eval_count < max_evals:
        gen += 1
        progress = eval_count / float(max_evals)
        target_col_size = int(round(pop_per_colony - (pop_per_colony - n_min_colony) * (progress ** 1.5)))
        target_col_size = max(n_min_colony, target_col_size)

        for c_id, col in enumerate(colonies):
            col.sort(key=lambda a: a.fitness)
            if len(col) > target_col_size:
                pruned = col[target_col_size:]
                colonies[c_id] = col[:target_col_size]
                col = colonies[c_id]
                for p in pruned:
                    if len(spore_archive) < max_archive:
                        spore_archive.append(p.position.copy())
                    else:
                        spore_archive[np.random.randint(max_archive)] = p.position.copy()
            num_c = len(col)
            
            if c_id == 0:  # Exploiter
                p_i_range = (0.05, 0.12)
                p_eigen = 0.60
            elif c_id == 1:  # Explorer
                p_i_range = (0.15, 0.30)
                p_eigen = 0.20
            else:  # Bridge
                p_i_range = (0.08, 0.20)
                p_eigen = 0.40

            if gen % 4 == 0 and num_c >= max(4, min(8, n_dims)):
                sample = np.array([a.position for a in col[:max(4, num_c // 2)]])
                cov = np.cov(sample, rowvar=False)
                if cov.ndim == 2 and cov.shape[0] == n_dims:
                    try:
                        _, e_vecs = np.linalg.eigh(cov + 1e-8 * np.eye(n_dims))
                        eigen_bases[c_id] = e_vecs
                    except Exception:
                        pass

            succ_F = []
            succ_CR = []
            diffs = []

            for idx, ant in enumerate(col):
                if eval_count >= max_evals:
                    break

                r_m = np.random.randint(H_size)
                while True:
                    f_cand = mem_F[c_id][r_m] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                    if f_cand > 0:
                        F_i = min(f_cand, 1.0)
                        break
                CR_i = np.clip(np.random.normal(mem_CR[c_id][r_m], 0.1), 0.0, 1.0)

                p_best_k = max(2, int(round(np.random.uniform(*p_i_range) * num_c)))
                x_pbest = col[np.random.randint(p_best_k)].position

                r1 = np.random.randint(num_c)
                while r1 == idx and num_c > 1:
                    r1 = np.random.randint(num_c)
                x_r1 = col[r1].position

                tot_pool = num_c + len(spore_archive)
                r2 = np.random.randint(tot_pool)
                while r2 == idx or r2 == r1:
                    r2 = np.random.randint(tot_pool)
                x_r2 = col[r2].position if r2 < num_c else spore_archive[r2 - num_c]

                v_donor = ant.position + F_i * (x_pbest - ant.position) + F_i * (x_r1 - x_r2)

                if np.random.rand() < p_eigen:
                    z_target = eigen_bases[c_id].T @ ant.position
                    z_donor = eigen_bases[c_id].T @ v_donor
                    j_rand = np.random.randint(n_dims)
                    mask = (np.random.rand(n_dims) < CR_i)
                    mask[j_rand] = True
                    trial_pos = eigen_bases[c_id] @ np.where(mask, z_donor, z_target)
                else:
                    j_rand = np.random.randint(n_dims)
                    mask = (np.random.rand(n_dims) < CR_i)
                    mask[j_rand] = True
                    trial_pos = np.where(mask, v_donor, ant.position)

                trial_pos = repair_boundary(trial_pos, lb, ub, ant.position)
                trial_fit = float(fitness(trial_pos))
                eval_count += 1

                if trial_fit < ant.fitness:
                    diff = ant.fitness - trial_fit
                    if len(spore_archive) < max_archive:
                        spore_archive.append(ant.position.copy())
                    else:
                        spore_archive[np.random.randint(max_archive)] = ant.position.copy()

                    ant.position = trial_pos.copy()
                    ant.fitness = trial_fit
                    succ_F.append(F_i)
                    succ_CR.append(CR_i)
                    diffs.append(diff)

                    if trial_fit < global_best.fitness:
                        global_best = ant.copy()

            if succ_F:
                w = np.array(diffs, dtype=float)
                w_s = np.sum(w)
                if w_s > 1e-15:
                    w_n = w / w_s
                    lehmer_F = np.sum(w_n * (np.array(succ_F) ** 2)) / np.sum(w_n * np.array(succ_F))
                    mean_CR = np.sum(w_n * np.array(succ_CR))
                    m_i = mem_idx[c_id]
                    mem_F[c_id][m_i] = float(np.clip(lehmer_F, 0.1, 1.0))
                    mem_CR[c_id][m_i] = float(np.clip(mean_CR, 0.0, 1.0))
                    mem_idx[c_id] = (m_i + 1) % H_size

        if gen % 12 == 0:
            for c in colonies:
                c.sort(key=lambda a: a.fitness)
                c[-1] = global_best.copy()

    return global_best


if __name__ == "__main__":
    test_funcs = [
        (1, "F1: Rotated Elliptic", cec2014.F12014, 0.312),
        (2, "F2: Rotated Bent Cigar", cec2014.F22014, 0.000125),
        (3, "F3: Rotated Discus", cec2014.F32014, 0.041),
        (4, "F4: Shifted & Rotated Rosenbrock", cec2014.F42014, 0.32),
        (5, "F5: Shifted & Rotated Ackley", cec2014.F52014, 0.201),
        (8, "F8: Shifted Rastrigin", cec2014.F82014, 0.0),
        (10, "F10: Shifted Schwefel", cec2014.F102014, 195.0),
        (12, "F12: Shifted & Rotated Katsuura", cec2014.F122014, 0.42),
    ]
    print("=" * 70)
    print("  TEST METAPOPULATION ON CEC 2014 (30D)")
    print("=" * 70)
    for f_id, name, f_cls, lshade_ref in test_funcs:
        fn = f_cls(ndim=30)
        t0 = time.perf_counter()
        res = ooa_metapopulation(30, fn.lb.tolist(), fn.ub.tolist(), fn.evaluate, max_evals=300000)
        dur = time.perf_counter() - t0
        err = max(0.0, res.fitness - fn.f_bias)
        if err < 1e-8:
            err = 0.0
        print(f"[{f_id}] {name:<35} (30D) -> Error: {err:<10.4e} | L-SHADE Ref: {lshade_ref:<10.4e} ({dur:.1f}s)")
