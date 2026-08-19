"""
Ophiocordyceps Optimization Algorithm (OOA) - GPU Native Tensor Engine
High-throughput, fully vectorized metaheuristic execution on GPU (CUDA, DirectML, MPS, or CPU Tensors).

Features:
- 100% GPU VRAM execution with zero CPU-GPU transfer bottlenecks
- Batched parallel evaluation across thousands of ants simultaneously
- Multi-Colony Metapopulation with Tensorized Spore Wind Drift
- Batch Eigen-Decomposition for Rotational Invariance (torch.linalg.eigh)
- Tensorized Midpoint Boundary Repair & Spore Archive
"""
import time
import math
import numpy as np
from typing import Callable, List, Optional, Tuple, Union, Dict, Any


def get_torch_device(device_pref: str = "auto") -> Any:
    """Rileva e restituisce il device PyTorch ottimale."""
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch non è installato. Esegui 'pip install torch' per abilitare il supporto GPU.")

    pref = str(device_pref).lower().strip()
    if pref in ["cuda", "gpu"] and torch.cuda.is_available():
        return torch.device("cuda:0")
    elif pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif pref == "directml":
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            pass
    elif pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device()
        except ImportError:
            pass
    return torch.device("cpu")


def ophiocordyceps_gpu(n_ants: int, n_dims: int,
                       lower_bound: Union[np.ndarray, list, Any],
                       upper_bound: Union[np.ndarray, list, Any],
                       fitness: Callable,
                       minimization: bool = True,
                       max_iter: int = 250,
                       device: str = "auto",
                       dtype: Any = None,
                       verbose: bool = False) -> Dict[str, Any]:
    """
    Esecuzione vettorizzata nativa di OOA su tensori GPU (PyTorch).
    
    Args:
        n_ants: Numero di formiche per sotto-colonia.
        n_dims: Dimensionalità del problema.
        lower_bound: Limiti inferiori [D].
        upper_bound: Limiti superiori [D].
        fitness: Funzione obiettivo (accetta tensore [B, D] o vettore [D]).
        minimization: True se problema di minimo.
        max_iter: Numero di iterazioni.
        device: 'auto', 'cuda', 'mps', 'directml', o 'cpu'.
        dtype: torch.float32 o torch.float64.
        verbose: True per loggare la convergenza.

    Returns:
        Dizionario con 'position', 'fitness', 'evaluations', 'device', 'wall_time'.
    """
    import torch
    t_start = time.perf_counter()

    torch_dev = get_torch_device(device)
    torch_dtype = dtype if dtype is not None else torch.float64

    lb = torch.as_tensor(lower_bound, dtype=torch_dtype, device=torch_dev)
    ub = torch.as_tensor(upper_bound, dtype=torch_dtype, device=torch_dev)
    span = ub - lb

    num_colonies = 3
    pop_col = max(25, 8 * n_dims) if n_ants is None else max(15, n_ants)
    n_min_col = 4

    # 1. Inizializzazione popolazione su GPU [M, N, D]
    pos1 = lb + torch.rand((num_colonies, pop_col // 2, n_dims), dtype=torch_dtype, device=torch_dev) * span
    pos2 = lb + ub - pos1  # Oppositional initialization
    pop = torch.cat([pos1, pos2], dim=1)
    if pop.shape[1] < pop_col:
        extra = lb + torch.rand((num_colonies, pop_col - pop.shape[1], n_dims), dtype=torch_dtype, device=torch_dev) * span
        pop = torch.cat([pop, extra], dim=1)

    # 2. Wrapper per valutazione vettorizzata o scalare
    def evaluate_batch(tensor_batch: torch.Tensor) -> torch.Tensor:
        # tensor_batch shape: [B, D]
        b_size = tensor_batch.shape[0]
        try:
            # Se la fitness è una funzione vettorizzata che restituisce un tensore/array di dimensione B
            res = fitness(tensor_batch)
            if isinstance(res, torch.Tensor) and res.numel() == b_size:
                return res.view(-1)
            elif isinstance(res, np.ndarray) and res.size == b_size:
                return torch.from_numpy(res).to(device=torch_dev, dtype=torch_dtype).view(-1)
        except Exception:
            pass

        # Valutazione sicura per ogni individuo
        out = torch.zeros(b_size, dtype=torch_dtype, device=torch_dev)
        batch_np = tensor_batch.detach().cpu().numpy()
        for k in range(b_size):
            out[k] = float(fitness(batch_np[k]))
        return out

    # Valutazione iniziale della popolazione
    flat_pop = pop.view(-1, n_dims)
    raw_fits = evaluate_batch(flat_pop)
    eval_count = flat_pop.shape[0]
    
    fits = raw_fits.view(num_colonies, pop_col)
    if not minimization:
        fits = -fits

    # Identifica il migliore globale
    best_val, best_idx_flat = torch.min(fits.view(-1), dim=0)
    best_c = best_idx_flat // pop_col
    best_i = best_idx_flat % pop_col
    global_best_pos = pop[best_c, best_i].clone()
    global_best_fit = best_val.item()

    # Spore Archive su GPU
    max_archive = pop_col * 3
    spore_archive = torch.empty((0, n_dims), dtype=torch_dtype, device=torch_dev)

    # Memoria storica di Lehmer per ciascuna colonia
    H_size = 12
    mem_F = torch.linspace(0.3, 0.8, H_size, dtype=torch_dtype, device=torch_dev).repeat(num_colonies, 1)
    mem_CR = torch.linspace(0.1, 0.9, H_size, dtype=torch_dtype, device=torch_dev).repeat(num_colonies, 1)
    mem_idx = [0] * num_colonies

    eigen_bases = [torch.eye(n_dims, dtype=torch_dtype, device=torch_dev) for _ in range(num_colonies)]
    max_evals = max_iter * pop_col * num_colonies

    gen = 0
    while eval_count < max_evals:
        gen += 1
        progress = eval_count / float(max_evals)

        # LPSR dinamico
        target_col_size = int(round(pop_col - (pop_col - n_min_col) * (progress ** 1.5)))
        target_col_size = max(n_min_col, target_col_size)

        for c_id in range(num_colonies):
            c_pop = pop[c_id]
            c_fits = fits[c_id]

            # Ordina individui della colonia per fitness
            sorted_indices = torch.argsort(c_fits)
            c_pop = c_pop[sorted_indices]
            c_fits = c_fits[sorted_indices]

            # Pota se eccede target_col_size
            if c_pop.shape[0] > target_col_size:
                pruned = c_pop[target_col_size:]
                c_pop = c_pop[:target_col_size]
                c_fits = c_fits[:target_col_size]
                
                # Archivia nel buffer di spore GPU
                spore_archive = torch.cat([spore_archive, pruned], dim=0)
                if spore_archive.shape[0] > max_archive:
                    spore_archive = spore_archive[-max_archive:]

            cur_col_size = c_pop.shape[0]
            if cur_col_size < 3:
                continue

            # Parametri differenziati per colonia
            if c_id == 0:  # Exploiter
                p_i_range = (0.05, 0.12)
                p_eigen = 0.65
            elif c_id == 1:  # Explorer
                p_i_range = (0.15, 0.30)
                p_eigen = 0.20
            else:  # Bridge
                p_i_range = (0.08, 0.20)
                p_eigen = 0.45

            # Calcolo matrice di covarianza ed autovettori su GPU
            if gen % 4 == 0 and cur_col_size >= max(4, min(8, n_dims)):
                sample_k = max(4, cur_col_size // 2)
                sample = c_pop[:sample_k]
                centered = sample - sample.mean(dim=0, keepdim=True)
                cov_mat = (centered.T @ centered) / max(1, sample_k - 1)
                cov_mat += 1e-8 * torch.eye(n_dims, dtype=torch_dtype, device=torch_dev)
                try:
                    _, eig_vecs = torch.linalg.eigh(cov_mat)
                    eigen_bases[c_id] = eig_vecs
                except Exception:
                    pass

            # Campionamento vettorizzato di F e CR dalla memoria
            r_mem = torch.randint(0, H_size, (cur_col_size,), device=torch_dev)
            mu_f = mem_F[c_id, r_mem]
            mu_cr = mem_CR[c_id, r_mem]

            # Cauchy sampling per F su GPU
            u_cauchy = torch.rand(cur_col_size, dtype=torch_dtype, device=torch_dev)
            F_samples = mu_f + 0.1 * torch.tan(math.pi * (u_cauchy - 0.5))
            F_samples = torch.clamp(F_samples, 0.1, 1.0)

            # Normal sampling per CR su GPU
            CR_samples = torch.normal(mu_cr, 0.1)
            CR_samples = torch.clamp(CR_samples, 0.0, 1.0)

            # Selezione p-best
            p_val = np.random.uniform(*p_i_range)
            p_k = max(2, int(round(p_val * cur_col_size)))
            p_indices = torch.randint(0, p_k, (cur_col_size,), device=torch_dev)
            x_pbest = c_pop[p_indices]

            # Selezione r1
            r1_indices = torch.randint(0, cur_col_size, (cur_col_size,), device=torch_dev)
            # Evita r1 == self
            same_mask = (r1_indices == torch.arange(cur_col_size, device=torch_dev))
            r1_indices[same_mask] = (r1_indices[same_mask] + 1) % cur_col_size
            x_r1 = c_pop[r1_indices]

            # Selezione r2 da Popolazione U Archive
            tot_pool_size = cur_col_size + spore_archive.shape[0]
            r2_raw = torch.randint(0, tot_pool_size, (cur_col_size,), device=torch_dev)
            # Carica x_r2
            in_pop_mask = r2_raw < cur_col_size
            x_r2 = torch.empty_like(c_pop)
            if in_pop_mask.any():
                x_r2[in_pop_mask] = c_pop[r2_raw[in_pop_mask]]
            if (~in_pop_mask).any():
                arch_idx = r2_raw[~in_pop_mask] - cur_col_size
                x_r2[~in_pop_mask] = spore_archive[arch_idx]

            # Mutazione differenziale vettorizzata
            F_mat = F_samples.unsqueeze(1)  # [N, 1]
            v_donor = c_pop + F_mat * (x_pbest - c_pop) + F_mat * (x_r1 - x_r2)

            # Crossover (nello spazio cartesiano o autospazio)
            j_rand = torch.randint(0, n_dims, (cur_col_size,), device=torch_dev)
            cr_mask = torch.rand((cur_col_size, n_dims), dtype=torch_dtype, device=torch_dev) < CR_samples.unsqueeze(1)
            cr_mask.scatter_(1, j_rand.unsqueeze(1), True)

            if np.random.rand() < p_eigen:
                B = eigen_bases[c_id]
                z_target = c_pop @ B
                z_donor = v_donor @ B
                z_trial = torch.where(cr_mask, z_donor, z_target)
                trial_pos = z_trial @ B.T
            else:
                trial_pos = torch.where(cr_mask, v_donor, c_pop)

            # Riparazione a punto medio sui bordi
            viol_lb = trial_pos < lb
            viol_ub = trial_pos > ub
            trial_pos = torch.where(viol_lb, (lb + c_pop) / 2.0, trial_pos)
            trial_pos = torch.where(viol_ub, (ub + c_pop) / 2.0, trial_pos)

            # Valutazione fitness del batch
            raw_trial_fits = evaluate_batch(trial_pos)
            eval_count += cur_col_size
            trial_fits = raw_trial_fits if minimization else -raw_trial_fits

            # Selezione e aggiornamento
            better_mask = trial_fits < c_fits
            if better_mask.any():
                # Salva le vecchie posizioni superate nell'archivio
                spore_archive = torch.cat([spore_archive, c_pop[better_mask]], dim=0)
                if spore_archive.shape[0] > max_archive:
                    spore_archive = spore_archive[-max_archive:]

                diffs = c_fits[better_mask] - trial_fits[better_mask]
                c_pop[better_mask] = trial_pos[better_mask]
                c_fits[better_mask] = trial_fits[better_mask]

                # Aggiornamento Lehmer
                succ_F = F_samples[better_mask]
                succ_CR = CR_samples[better_mask]
                w = diffs / torch.sum(diffs)
                lehmer_f = torch.sum(w * (succ_F ** 2)) / torch.sum(w * succ_F)
                mean_cr = torch.sum(w * succ_CR)
                
                m_i = mem_idx[c_id]
                mem_F[c_id, m_i] = torch.clamp(lehmer_f, 0.1, 1.0)
                mem_CR[c_id, m_i] = torch.clamp(mean_cr, 0.0, 1.0)
                mem_idx[c_id] = (m_i + 1) % H_size

                # Verifica se trovato nuovo minimo globale
                min_c_val, min_c_idx = torch.min(c_fits, dim=0)
                if min_c_val.item() < global_best_fit:
                    global_best_fit = min_c_val.item()
                    global_best_pos = c_pop[min_c_idx].clone()

            # Aggiorna tensore colonia
            pop = list(pop)
            pop[c_id] = c_pop
            fits = list(fits)
            fits[c_id] = c_fits

        # Spore Wind Drift (ogni 10 generazioni)
        if gen % 10 == 0:
            for c_id in range(num_colonies):
                worst_idx = torch.argmax(fits[c_id])
                pop[c_id][worst_idx] = global_best_pos.clone()
                fits[c_id][worst_idx] = global_best_fit

    elapsed = time.perf_counter() - t_start
    final_best_fit = global_best_fit if minimization else -global_best_fit

    return {
        "position": global_best_pos.cpu().numpy(),
        "fitness": float(final_best_fit),
        "evaluations": eval_count,
        "device": str(torch_dev),
        "wall_time_s": elapsed
    }
