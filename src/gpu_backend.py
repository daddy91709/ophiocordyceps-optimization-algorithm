"""
GPU and Tensor Acceleration Backend for Ophiocordyceps Algorithm.
Supports PyTorch (CUDA, DirectML, Apple MPS, CPU tensors) and CuPy.
"""
from typing import Callable, Optional, Tuple, Dict, Any, Union
import numpy as np


class GPUTensorBackend:
    """
    Gestisce l'esecuzione dell'algoritmo Ophiocordyceps su GPU tramite tensori.
    """
    def __init__(self, device: str = "auto"):
        self.device_name = device
        self.torch_module = None
        self.torch_device = None
        self.is_gpu = False
        self._init_backend()

    def _init_backend(self):
        try:
            import torch
            self.torch_module = torch

            if self.device_name == "auto":
                if torch.cuda.is_available():
                    self.torch_device = torch.device("cuda:0")
                    self.is_gpu = True
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.torch_device = torch.device("mps")
                    self.is_gpu = True
                else:
                    try:
                        import torch_directml
                        if torch_directml.is_available():
                            self.torch_device = torch_directml.device()
                            self.is_gpu = True
                        else:
                            self.torch_device = torch.device("cpu")
                    except ImportError:
                        self.torch_device = torch.device("cpu")
            elif "cuda" in self.device_name and torch.cuda.is_available():
                self.torch_device = torch.device(self.device_name)
                self.is_gpu = True
            elif "directml" in self.device_name:
                import torch_directml
                self.torch_device = torch_directml.device()
                self.is_gpu = True
            else:
                self.torch_device = torch.device("cpu")
        except ImportError:
            self.torch_module = None
            self.torch_device = None
            self.is_gpu = False

    @property
    def is_available(self) -> bool:
        return self.torch_module is not None

    def get_info(self) -> Dict[str, Any]:
        return {
            "torch_available": self.is_available,
            "device": str(self.torch_device) if self.torch_device else "None",
            "is_gpu": self.is_gpu
        }

    def run_gpu_optimization(self, n_ants: int, n_dims: int, 
                             lower_bound: Union[np.ndarray, list], 
                             upper_bound: Union[np.ndarray, list], 
                             fitness: Callable, 
                             minimization: bool = True,
                             max_iter: int = 200,
                             learning_rate: float = 0.05,
                             ant_step_size: float = 0.01,
                             base_death_prob: float = 0.1,
                             base_infection_prob: float = 0.15,
                             best_influence: float = 0.1,
                             convergence_tolerance: float = 1e-6,
                             convergence_patience: int = 20,
                             verbose: bool = False):
        """
        Esegue l'algoritmo Ophiocordyceps su tensori GPU / PyTorch.
        """
        if not self.is_available:
            raise RuntimeError("PyTorch non è installato per l'esecuzione su tensori GPU.")

        torch = self.torch_module
        dev = self.torch_device

        lb = torch.tensor(lower_bound, dtype=torch.float32, device=dev)
        ub = torch.tensor(upper_bound, dtype=torch.float32, device=dev)

        # Inizializzazione popolazione su device
        pop_pos = lb + torch.rand((n_ants, n_dims), dtype=torch.float32, device=dev) * (ub - lb)
        pop_infected = torch.zeros(n_ants, dtype=torch.bool, device=dev)
        pop_alive = torch.ones(n_ants, dtype=torch.bool, device=dev)
        pop_inf_prob = torch.full((n_ants,), base_infection_prob, dtype=torch.float32, device=dev)
        pop_death_prob = torch.full((n_ants,), base_death_prob, dtype=torch.float32, device=dev)
        pop_no_improv = torch.zeros(n_ants, dtype=torch.int32, device=dev)
        pop_best_fit = torch.full((n_ants,), float('inf') if minimization else float('-inf'), dtype=torch.float32, device=dev)

        global_best_fit = float('inf') if minimization else float('-inf')
        global_best_pos = pop_pos[0].clone()

        no_improvement_count = 0
        prev_best = None

        if verbose:
            print(f"[GPU Tensor Backend] Avvio su device={dev} ({n_ants} formiche, {n_dims}D)...")

        for it in range(max_iter):
            grad_inf = it / max_iter

            # 1. Update probabilità
            alive_mask = pop_alive
            pop_inf_prob[alive_mask] += base_infection_prob * 0.15
            pop_death_prob[alive_mask] += base_death_prob * 0.1

            # 2. Movimento stocastico (Browniano)
            sigma = 1.0
            brownian = sigma * np.sqrt(ant_step_size) * torch.randn_like(pop_pos)

            # 3. Componente di attrazione best
            best_comp = best_influence * (global_best_pos.unsqueeze(0) - pop_pos)

            # 4. Gradiente numerico per infetti
            inf_mask = pop_alive & pop_infected
            if inf_mask.any():
                eps = 1e-6
                grad = torch.zeros_like(pop_pos)
                inf_indices = torch.nonzero(inf_mask).squeeze(-1)
                
                # Calcola gradienti
                for idx in inf_indices:
                    pos_np = pop_pos[idx].cpu().numpy()
                    grad_np = np.zeros(n_dims, dtype=float)
                    for d in range(n_dims):
                        p_up = pos_np.copy()
                        p_down = pos_np.copy()
                        p_up[d] += eps
                        p_down[d] -= eps
                        grad_np[d] = (fitness(p_up) - fitness(p_down)) / (2 * eps)
                    grad[idx] = torch.tensor(grad_np, dtype=torch.float32, device=dev)

                # Aggiornamento infetti
                if minimization:
                    pop_pos[inf_mask] += (1 - grad_inf) * brownian[inf_mask] - (learning_rate * grad[inf_mask]) + best_comp[inf_mask]
                else:
                    pop_pos[inf_mask] += (1 - grad_inf) * brownian[inf_mask] + (learning_rate * grad[inf_mask]) + best_comp[inf_mask]

            # Aggiornamento sani
            healthy_mask = pop_alive & (~pop_infected)
            if healthy_mask.any():
                pop_pos[healthy_mask] += brownian[healthy_mask] + best_comp[healthy_mask]

            # Clipping
            pop_pos = torch.clamp(pop_pos, min=lb, max=ub)

            # 5. Valutazione fitness
            alive_indices = torch.nonzero(pop_alive).squeeze(-1)
            current_fits = []
            for idx in alive_indices:
                fit_val = float(fitness(pop_pos[idx].cpu().numpy()))
                current_fits.append(fit_val)
                
                # Update best individuale
                if (minimization and fit_val < pop_best_fit[idx]) or (not minimization and fit_val > pop_best_fit[idx]):
                    pop_best_fit[idx] = fit_val
                    pop_no_improv[idx] = 0
                else:
                    pop_no_improv[idx] += 1

            # Infezione
            rand_inf = torch.rand(n_ants, device=dev)
            pop_infected = pop_infected | (rand_inf <= pop_inf_prob)

            # Morte
            if inf_mask.any():
                long_stagnant = pop_no_improv > (max_iter / 5)
                pop_death_prob[inf_mask & long_stagnant] += (base_death_prob * 0.7)
                
                rand_death = torch.rand(n_ants, device=dev)
                died_mask = inf_mask & (rand_death <= pop_death_prob)
                
                pop_alive[died_mask] = False
                pop_infected[died_mask] = False

            # Update Best globale
            if len(current_fits) > 0:
                best_idx_local = int(np.argmin(current_fits) if minimization else np.argmax(current_fits))
                best_idx_global = alive_indices[best_idx_local]
                best_val = current_fits[best_idx_local]

                if (minimization and best_val < global_best_fit) or (not minimization and best_val > global_best_fit):
                    global_best_fit = best_val
                    global_best_pos = pop_pos[best_idx_global].clone()

            # Convergenza
            if prev_best is not None:
                impr = (prev_best - global_best_fit) if minimization else (global_best_fit - prev_best)
                if impr > convergence_tolerance:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            prev_best = global_best_fit

            if no_improvement_count >= convergence_patience:
                break

        return global_best_pos.cpu().numpy(), global_best_fit
