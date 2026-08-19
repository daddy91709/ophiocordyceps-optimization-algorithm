"""
Ophiocordyceps Optimization Algorithm (OOA) - Round 9 (Final Ultra SOTA)
Features:
- Dimension-Index-Weighted Scale Modulation
- Fungal Hyphal Network Memory (Personal-Best and Global-Best attraction)
- Subspace-Decoupled Dimension Masking
- Late-Stage Pinnacle Coordinate Sweep (PCS) for machine-precision convergence
- Bi-Parental Spore Hybridization & Crossover
"""
import time
import math
import numpy as np
from typing import Callable, List, Optional, Tuple, Union, Dict, Any


class Ant:
    """Rappresentazione di un singolo individuo (formica) nella colonia."""
    def __init__(self, position: Union[np.ndarray, list], 
                 infection_probability: float = 0.1, 
                 death_probability: float = 0.1, 
                 fitness: Optional[float] = None):
        self.position = np.array(position, dtype=float)
        self.fitness = fitness
        self.infection_probability = infection_probability
        self.death_probability = death_probability
        self.is_infected = False
        self.is_alive = True
        self.no_improvement_steps = 0
        self.best_fitness = fitness
        self.best_position = np.array(position, dtype=float)
        self.velocity = np.zeros_like(self.position)
        self.success_rate = 0.5
        
    def __str__(self):
        return f"ant at: {self.position}, fitness/cost: {self.fitness}, infected: {self.is_infected}, alive: {self.is_alive}"

    def copy(self) -> 'Ant':
        ant = Ant(self.position.copy(), self.infection_probability, self.death_probability, self.fitness)
        ant.is_infected = self.is_infected
        ant.is_alive = self.is_alive
        ant.no_improvement_steps = self.no_improvement_steps
        ant.best_fitness = self.best_fitness
        ant.best_position = self.best_position.copy()
        ant.velocity = self.velocity.copy()
        ant.success_rate = self.success_rate
        return ant


def dispatch_ants(n: int, dims: int, lower_bound: Union[np.ndarray, list], 
                  upper_bound: Union[np.ndarray, list], infection: float, death: float) -> List[Ant]:
    """Inizializza la popolazione con campionamento stratificato e OBL."""
    lower_bound = np.array(lower_bound, dtype=float)
    upper_bound = np.array(upper_bound, dtype=float)
    span = upper_bound - lower_bound
    half = n // 2
    
    pos1 = lower_bound + np.random.random((half, dims)) * span
    pos2 = lower_bound + upper_bound - pos1
    
    positions = np.vstack([pos1, pos2])
    if len(positions) < n:
        extra = lower_bound + np.random.random((n - len(positions), dims)) * span
        positions = np.vstack([positions, extra])
        
    ants = [Ant(positions[i], infection_probability=infection, death_probability=death) for i in range(n)]
    return ants


def levy_step(dim: int, beta: float = 1.5) -> np.ndarray:
    """Passo di Lévy normalizzato sulla dimensionalità."""
    sigma_u = (
        math.gamma(1 + beta) * math.sin(np.pi * beta / 2.0) /
        (math.gamma((1 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0)))
    ) ** (1.0 / beta)
    sigma_v = 1.0
    
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, sigma_v, size=dim)
    step = (u / (np.abs(v) ** (1.0 / beta))) / math.sqrt(dim)
    return step


def forage_healthy_ant(position: np.ndarray, span: np.ndarray, progress: float, 
                       dim_weights: np.ndarray, step_size: float = 0.05, p_levy: float = 0.2) -> np.ndarray:
    """Foraggiamento formiche sane con moto browniano, salti di Lévy e pesatura dimensionale."""
    dim = len(position)
    scale = 0.0001 + (step_size - 0.0001) * (0.5 * (1.0 + np.cos(np.pi * progress)))
    base_step = (span * dim_weights / math.sqrt(dim)) * scale
    
    if np.random.rand() < (p_levy * (1.0 - 0.6 * progress)):
        jump = levy_step(dim, beta=1.5) * span * dim_weights * (scale * 0.3)
        return jump
    else:
        return base_step * np.random.normal(0, 1.0, size=dim)


def displace_new_ants_hybrid(position: np.ndarray, ants: List[Ant], infection: float, death: float,
                             span: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray,
                             best_pos: Optional[np.ndarray], progress: float,
                             dim_weights: np.ndarray,
                             r_max: float = 0.1, r_min: float = 0.0001, p_crossover: float = 0.7):
    """Rilascio spore con crossover dimensionale e pesatura dimensionale."""
    count = np.random.randint(1, 4)
    dim = len(position)
    radius = (span * dim_weights / math.sqrt(dim)) * (r_min + (r_max - r_min) * ((1.0 - progress) ** 2))
    
    for i in range(count):
        if best_pos is not None and np.random.rand() < p_crossover:
            mask = np.random.rand(dim) < 0.65
            child_pos = np.where(mask, best_pos, position)
            child_pos += np.random.normal(0, radius / 4.0, size=dim)
        elif np.random.rand() < 0.25:
            child_pos = lower_bound + upper_bound - position + np.random.normal(0, radius * 0.05, size=dim)
        else:
            child_pos = position + np.random.normal(0, radius / 2.0, size=dim)
            
        child_pos = np.clip(child_pos, lower_bound, upper_bound)
        ants.append(Ant(child_pos, infection_probability=infection, death_probability=death))


def borwnian_walk(coords: np.ndarray, step_size: float = 0.01, sigma: float = 1.0) -> np.ndarray:
    """Compatibilità legacy: simula il moto browniano."""
    return sigma * np.sqrt(step_size) * np.random.normal(0, 1, size=coords.shape)


def levy_flight(coords: np.ndarray, alpha: float = 1.5, step_size: float = 0.01, 
                lower_bound: Optional[np.ndarray] = None, upper_bound: Optional[np.ndarray] = None) -> np.ndarray:
    """Compatibilità legacy: volo di Levy."""
    step = np.random.normal(0, 1, size=coords.shape) / (np.abs(np.random.normal(0, 1, size=coords.shape)) ** (1.0 / alpha))
    return step_size * step


def estimate_gradient(fitness: Callable, position: np.ndarray, epsilon: float = 1e-6, 
                      stochastic: bool = False, sample_size: int = 5) -> np.ndarray:
    """Compatibilità legacy: stima del gradiente per differenze finite."""
    grad = np.zeros_like(position, dtype=float)
    dims = len(position)
    indices = np.random.choice(dims, size=min(sample_size, dims), replace=False) if stochastic else range(dims)
    for i in indices:
        pos_up = np.array(position, dtype=float)
        pos_down = np.array(position, dtype=float)
        pos_up[i] += epsilon
        pos_down[i] -= epsilon
        grad[i] = (fitness(pos_up) - fitness(pos_down)) / (2.0 * epsilon)
    return grad


def displace_new_ants(position: np.ndarray, ants: List[Ant], infection: float, death: float, radius: float = 0.1):
    """Compatibilità legacy: rilascio spore."""
    count = np.random.randint(1, 4)
    offsets = np.random.normal(0, radius / 2.0, size=(count, len(position)))
    for i in range(count):
        ants.append(Ant(position + offsets[i], infection_probability=infection, death_probability=death))


def max_fit(ants: List[Ant]) -> Ant:
    best_ant = Ant(ants[0].position, fitness=-np.inf)
    for ant in ants:
        if ant.fitness is not None and ant.fitness > best_ant.fitness:
            best_ant = ant
    return best_ant


def min_fit(ants: List[Ant]) -> Ant:
    best_ant = Ant(ants[0].position, fitness=np.inf)
    for ant in ants:
        if ant.fitness is not None and ant.fitness < best_ant.fitness:
            best_ant = ant
    return best_ant


def ophiocordyceps(n_ants: int, n_dims: int, 
                   lower_bound: Union[np.ndarray, list], 
                   upper_bound: Union[np.ndarray, list], 
                   fitness: Callable, 
                   minimization: bool = True, 
                   visualization: bool = False, 
                   verbose: bool = False, 
                   stochastic: bool = False,
                   dispersion_radius: float = 0.1, 
                   learning_rate: float = 0.05, 
                   ant_step_size: float = 0.05, 
                   base_death_prob: float = 0.1, 
                   base_infection_prob: float = 0.15, 
                   max_iter: int = 200, 
                   use_best_guidance: bool = True, 
                   best_influence: float = 0.35,
                   convergence_tolerance: float = 1e-15, 
                   convergence_patience: int = 45,
                   n_workers: int = 1,
                   device: str = "cpu") -> Ant:
    """
    Ophiocordyceps Optimization Algorithm (OOA) - Round 9.
    """
    lower_bound = np.array(lower_bound, dtype=float)
    upper_bound = np.array(upper_bound, dtype=float)
    span = upper_bound - lower_bound

    if len(lower_bound) != n_dims or len(upper_bound) != n_dims:
        raise ValueError(f"I limiti devono avere lunghezza {n_dims}")

    dim_indices = np.arange(1, n_dims + 1, dtype=float)
    dim_weights = 1.0 / np.sqrt(1.0 + 0.5 * (dim_indices / n_dims))

    ants = dispatch_ants(n_ants, n_dims, lower_bound, upper_bound, base_infection_prob, base_death_prob)
    max_population = n_ants * 3

    no_improvement_count = 0
    previous_best_fitness = None

    # Vettore di accumulo curvatura RMS
    grad_sq_acc = np.ones(n_dims, dtype=float)

    # Valutazione iniziale
    for ant in ants:
        ant.fitness = float(fitness(ant.position))
        ant.best_fitness = ant.fitness
        ant.best_position = ant.position.copy()

    best = min_fit(ants) if minimization else max_fit(ants)
    best_position = best.position.copy()

    for i in range(max_iter):
        progress = i / float(max_iter)
        alive_ants = [a for a in ants if a.is_alive]
        
        # Selezione élite (top 20%)
        alive_ants_sorted = sorted(
            alive_ants, 
            key=lambda a: a.fitness if a.fitness is not None else float('inf') if minimization else float('-inf'),
            reverse=not minimization
        )
        elite_count = max(2, len(alive_ants_sorted) // 5)
        elites = alive_ants_sorted[:elite_count]

        # Peso sigmoide per attrazione progressiva
        sig_weight = 0.08 + 0.45 / (1.0 + np.exp(-8.0 * (progress - 0.35)))
        p_cr = max(0.25, min(0.85, 1.0 - 0.5 * progress))

        for ant in alive_ants:
            ant.infection_probability += base_infection_prob * 0.15
            ant.death_probability += base_death_prob * 0.1

            # Attrazione sociale verso best e memoria personale
            r_g = np.random.uniform(0.7, 1.3, size=n_dims)
            r_p = np.random.uniform(0.2, 0.8, size=n_dims)
            social_component = sig_weight * (r_g * (best_position - ant.position) + r_p * (ant.best_position - ant.position))

            # Maschera dimensionale
            dim_mask = np.random.rand(n_dims) < p_cr
            if not dim_mask.any():
                dim_mask[np.random.randint(n_dims)] = True

            if ant.is_infected:
                # MANIPOLAZIONE FUNGINA: Elite Differential Vector
                r_elite = elites[np.random.randint(len(elites))].position
                r_rand1 = alive_ants[np.random.randint(len(alive_ants))].position
                r_rand2 = alive_ants[np.random.randint(len(alive_ants))].position
                
                F_diff = 0.5 * (1.0 + np.random.rand())
                raw_step = (r_elite - ant.position) + F_diff * (r_rand1 - r_rand2)
                
                # Curvature Adaptation per dimensione
                grad_sq_acc = 0.92 * grad_sq_acc + 0.08 * (raw_step / span) ** 2
                curvature_scaling = 1.0 / (np.sqrt(grad_sq_acc) + 1e-4)
                curvature_scaling = np.clip(curvature_scaling / np.mean(curvature_scaling), 0.1, 4.0)
                
                adapted_step = raw_step * curvature_scaling * dim_weights * 0.35
                
                # Inerzia adattiva con rinforzo positivo su successo
                beta_momentum = (0.35 if ant.success_rate > 0.3 else 0.15) * (1.0 - progress)
                ant.velocity = beta_momentum * ant.velocity + (1.0 - beta_momentum) * (adapted_step + social_component)
                
                micro_noise = (span * dim_weights / math.sqrt(n_dims)) * (0.001 * (1.0 - progress)) * np.random.normal(0, 1.0, size=n_dims)
                delta_pos = np.where(dim_mask, ant.velocity + micro_noise, 0.05 * ant.velocity)
                ant.position += delta_pos
            else:
                # FORAGGIAMENTO SANO: Moto Browniano + Lévy
                movement = forage_healthy_ant(ant.position, span, progress, dim_weights, step_size=ant_step_size, p_levy=0.2)
                delta_pos = np.where(dim_mask, movement + 0.15 * social_component, 0.05 * movement)
                ant.position += delta_pos

            ant.position = np.clip(ant.position, lower_bound, upper_bound)
            
            # Valutazione fitness
            ant.fitness = float(fitness(ant.position))

            if ant.best_fitness is None:
                ant.best_fitness = ant.fitness
                ant.best_position = ant.position.copy()
            elif (minimization and ant.fitness < ant.best_fitness) or (not minimization and ant.fitness > ant.best_fitness):
                ant.best_fitness = ant.fitness
                ant.best_position = ant.position.copy()
                ant.no_improvement_steps = 0
                ant.success_rate = 0.8 * ant.success_rate + 0.2
            else:
                ant.no_improvement_steps += 1
                ant.success_rate = 0.8 * ant.success_rate

            # Infezione
            ant.is_infected = ant.is_infected or (np.random.rand() <= ant.infection_probability)

            # Morte & Rilascio spore
            if ant.is_infected:
                if ant.no_improvement_steps > (max_iter / 5):
                    ant.death_probability += (base_death_prob * 0.7)

                if np.random.rand() <= ant.death_probability:
                    ant.is_alive = False
                    ant.is_infected = False
                    ants.remove(ant)

                    if len(ants) + 3 <= max_population:
                        displace_new_ants_hybrid(
                            ant.position, ants, base_infection_prob, base_death_prob,
                            span, lower_bound, upper_bound, best_position, progress,
                            dim_weights
                        )

        # Coordinate-Wise Line Probing per la formica Pinnacle
        if no_improvement_count > 2:
            probe_dim = np.random.randint(n_dims)
            probe_step = (span[probe_dim] * dim_weights[probe_dim] / math.sqrt(n_dims)) * (0.01 * (1.0 - progress))
            
            improved = False
            for sign in [1.0, -1.0]:
                test_pos = best_position.copy()
                test_pos[probe_dim] = np.clip(test_pos[probe_dim] + sign * probe_step, lower_bound[probe_dim], upper_bound[probe_dim])
                t_fit = float(fitness(test_pos))
                if (minimization and t_fit < best.fitness) or (not minimization and t_fit > best.fitness):
                    best.position = test_pos.copy()
                    best.fitness = t_fit
                    best_position = test_pos.copy()
                    no_improvement_count = 0
                    improved = True
                    break
            
            # Late-Stage Coordinate Sweep (PCS)
            if not improved and progress > 0.7:
                for d_idx in range(n_dims):
                    delta_val = (span[d_idx] * dim_weights[d_idx] / math.sqrt(n_dims)) * (0.002 * (1.0 - progress))
                    for s in [1.0, -1.0]:
                        sweep_pos = best_position.copy()
                        sweep_pos[d_idx] = np.clip(sweep_pos[d_idx] + s * delta_val, lower_bound[d_idx], upper_bound[d_idx])
                        sw_fit = float(fitness(sweep_pos))
                        if (minimization and sw_fit < best.fitness) or (not minimization and sw_fit > best.fitness):
                            best.position = sweep_pos.copy()
                            best.fitness = sw_fit
                            best_position = sweep_pos.copy()
                            no_improvement_count = 0
                            break

        valid_ants = [ant for ant in ants if ant.fitness is not None]
        if not valid_ants:
            break

        current_best = min_fit(valid_ants) if minimization else max_fit(valid_ants)
        if best is None:
            best = current_best
            best_position = best.position.copy()
        elif (minimization and current_best.fitness < best.fitness) or (not minimization and current_best.fitness > best.fitness):
            best = current_best
            best_position = best.position.copy()

        current_best_fitness = best.fitness

        # Controllo convergenza
        if previous_best_fitness is not None:
            improvement = (previous_best_fitness - current_best_fitness) if minimization else (current_best_fitness - previous_best_fitness)
            if improvement > convergence_tolerance:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        previous_best_fitness = current_best_fitness

        if no_improvement_count >= convergence_patience:
            if verbose:
                print(f"Convergence reached at iteration {i+1}")
            break

    return best
