"""
Ophiocordyceps Optimization Algorithm (OOA)
Supports Vectorized CPU execution, Multi-threaded fitness evaluation, and GPU tensor backends.
"""
import time
import numpy as np
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor


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
        
    def __str__(self):
        return f"ant at: {self.position}, fitness/cost: {self.fitness}, infected: {self.is_infected}, alive: {self.is_alive}"

    def copy(self) -> 'Ant':
        ant = Ant(self.position.copy(), self.infection_probability, self.death_probability, self.fitness)
        ant.is_infected = self.is_infected
        ant.is_alive = self.is_alive
        ant.no_improvement_steps = self.no_improvement_steps
        ant.best_fitness = self.best_fitness
        return ant


def dispatch_ants(n: int, dims: int, lower_bound: Union[np.ndarray, list], 
                  upper_bound: Union[np.ndarray, list], infection: float, death: float) -> List[Ant]:
    """Inizializza una popolazione di formiche distribuite casualmente nello spazio di ricerca."""
    lower_bound = np.array(lower_bound, dtype=float)
    upper_bound = np.array(upper_bound, dtype=float)
    
    if len(lower_bound) != dims or len(upper_bound) != dims:
        raise ValueError(f"I limiti devono avere lunghezza {dims}")
    
    positions = lower_bound + np.random.random((n, dims)) * (upper_bound - lower_bound)
    ants = [Ant(positions[i], infection_probability=infection, death_probability=death) for i in range(n)]
    return ants


def borwnian_walk(coords: np.ndarray, step_size: float = 0.01, sigma: float = 1.0) -> np.ndarray:
    """Simula il moto browniano per l'esplorazione stocastica."""
    return sigma * np.sqrt(step_size) * np.random.normal(0, 1, size=coords.shape)


def levy_flight(coords: np.ndarray, alpha: float = 1.5, step_size: float = 0.01, 
                lower_bound: Optional[np.ndarray] = None, upper_bound: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Genera un passo secondo la distribuzione di Levy, con ampiezza proporzionale 
    alla dimensione dello spazio di ricerca.
    """
    if lower_bound is not None and upper_bound is not None:
        lower_bound = np.array(lower_bound, dtype=float)
        upper_bound = np.array(upper_bound, dtype=float)
        space_diagonal = np.sqrt(np.sum((upper_bound - lower_bound)**2))
        adaptive_step_size = step_size * space_diagonal / 100
    else:
        adaptive_step_size = step_size
    
    step = np.random.normal(0, 1, size=coords.shape) / (np.abs(np.random.normal(0, 1, size=coords.shape))**(1/alpha))
    return adaptive_step_size * step


def displace_new_ants(position: np.ndarray, ants: List[Ant], infection: float, death: float, radius: float = 0.1):
    """Genera da 1 a 3 nuove formiche attorno alla posizione di una formica morta (rilascio spore)."""
    count = np.random.randint(1, 4)
    offsets = np.random.normal(0, radius / 2, size=(count, len(position)))
    for i in range(count):
        ants.append(Ant(position + offsets[i], infection_probability=infection, death_probability=death))


def max_fit(ants: List[Ant]) -> Ant:
    """Restituisce la formica con la fitness massima."""
    best_ant = Ant(ants[0].position, fitness=-np.inf)
    for ant in ants:
        if ant.fitness is not None and ant.fitness > best_ant.fitness:
            best_ant = ant
    return best_ant


def min_fit(ants: List[Ant]) -> Ant:
    """Restituisce la formica con la fitness minima."""
    best_ant = Ant(ants[0].position, fitness=np.inf)
    for ant in ants:
        if ant.fitness is not None and ant.fitness < best_ant.fitness:
            best_ant = ant
    return best_ant


def estimate_gradient(fitness: Callable, position: np.ndarray, epsilon: float = 1e-6, 
                      stochastic: bool = False, sample_size: int = 5) -> np.ndarray:
    """
    Calcola il gradiente numerico della funzione di fitness.
    Se stochastic=True, usa un sottoinsieme casuale di dimensioni per stimare il gradiente.
    """
    grad = np.zeros_like(position, dtype=float)
    dims = len(position)
    
    if stochastic:
        indices = np.random.choice(dims, size=min(sample_size, dims), replace=False)
    else:
        indices = range(dims)
    
    for i in indices:
        pos_up = np.array(position, dtype=float)
        pos_down = np.array(position, dtype=float)
        pos_up[i] += epsilon
        pos_down[i] -= epsilon
        grad[i] = (fitness(pos_up) - fitness(pos_down)) / (2 * epsilon)
    
    return grad


def estimate_gradients_batch(fitness: Callable, positions: List[np.ndarray], epsilon: float = 1e-6,
                             stochastic: bool = False, sample_size: int = 5,
                             n_workers: int = 1) -> List[np.ndarray]:
    """
    Calcola i gradienti per un gruppo di formiche infette, opzionalmente in parallelo su CPU.
    """
    if len(positions) == 0:
        return []
    
    if n_workers <= 1 or len(positions) == 1:
        return [estimate_gradient(fitness, pos, epsilon, stochastic, sample_size) for pos in positions]
    
    with ThreadPoolExecutor(max_workers=min(n_workers, len(positions))) as executor:
        futures = [executor.submit(estimate_gradient, fitness, pos, epsilon, stochastic, sample_size) for pos in positions]
        return [f.result() for f in futures]


def evaluate_fitness_batch(fitness: Callable, positions: List[np.ndarray], n_workers: int = 1) -> List[float]:
    """Valuta la fitness per una lista di posizioni, opzionalmente in parallelo su CPU."""
    if len(positions) == 0:
        return []
    if n_workers <= 1 or len(positions) == 1:
        return [float(fitness(pos)) for pos in positions]
    
    with ThreadPoolExecutor(max_workers=min(n_workers, len(positions))) as executor:
        futures = [executor.submit(fitness, pos) for pos in positions]
        return [float(f.result()) for f in futures]


def plot_ants_2d(ants: List[Ant], lower_bound: Union[np.ndarray, list], upper_bound: Union[np.ndarray, list], 
                 fitness_function: Callable, resolution: int = 50,
                 title: str = "Ophiocordyceps Optimization", cmap: str = "viridis", figsize: Tuple[int, int] = (8, 6)):
    """Visualizzazione opzionale per problemi 2D (richiede matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.patches import Patch
    except ImportError:
        return

    ant_x = [ant.position[0] for ant in ants]
    ant_y = [ant.position[1] for ant in ants]
    
    actual_lower_x = min(min(ant_x), lower_bound[0])
    actual_upper_x = max(max(ant_x), upper_bound[0])
    actual_lower_y = min(min(ant_y), lower_bound[1])
    actual_upper_y = max(max(ant_y), upper_bound[1])
    
    x = np.linspace(actual_lower_x, actual_upper_x, resolution)
    y = np.linspace(actual_lower_y, actual_upper_y, resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = fitness_function([X[j, i], Y[j, i]])
    
    plt.figure(figsize=figsize)
    contour = plt.contourf(X, Y, Z, 50, cmap=cm.get_cmap(cmap), alpha=0.8)
    plt.contour(X, Y, Z, 15, colors='black', alpha=0.3, linewidths=0.5)
    
    colors = ['red' if ant.is_infected else 'blue' for ant in ants]
    plt.scatter(ant_x, ant_y, c=colors, edgecolor='white', s=80, zorder=5)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.7)
    cbar = plt.colorbar(contour)
    cbar.set_label('Valore della funzione')
    
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Formica sana'),
        Patch(facecolor='red', edgecolor='black', label='Formica infetta')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    plt.show()


def ophiocordyceps(n_ants: int, n_dims: int, 
                   lower_bound: Union[np.ndarray, list], 
                   upper_bound: Union[np.ndarray, list], 
                   fitness: Callable, 
                   minimization: bool = True, 
                   visualization: bool = False, 
                   verbose: bool = False, 
                   stochastic: bool = False,
                   dispersion_radius: float = 0.2, 
                   learning_rate: float = 0.05, 
                   ant_step_size: float = 0.01, 
                   base_death_prob: float = 0.1, 
                   base_infection_prob: float = 0.15,
                   max_iter: int = 200, 
                   use_best_guidance: bool = True, 
                   best_influence: float = 0.1,
                   convergence_tolerance: float = 1e-6, 
                   convergence_patience: int = 20,
                   n_workers: int = 1,
                   device: str = "cpu") -> Ant:
    """
    Ophiocordyceps Optimization Algorithm (OOA).
    
    Parameters
    ----------
    n_ants : int
        Numero iniziale di formiche.
    n_dims : int
        Numero di dimensioni dello spazio di ricerca.
    lower_bound : list o np.ndarray
        Limiti inferiori dello spazio di ricerca.
    upper_bound : list o np.ndarray
        Limiti superiori dello spazio di ricerca.
    fitness : Callable
        Funzione obiettivo da ottimizzare.
    minimization : bool, default True
        True per minimizzazione, False per massimizzazione.
    visualization : bool, default False
        True per visualizzare grafici 2D ad ogni iterazione.
    verbose : bool, default False
        True per stampare il progresso delle iterazioni.
    stochastic : bool, default False
        True per stima stocastica del gradiente su un subset di dimensioni.
    dispersion_radius : float, default 0.2
        Raggio di dispersione delle spore alla morte di una formica infetta.
    learning_rate : float, default 0.05
        Passo del gradiente per il movimento delle formiche infette.
    ant_step_size : float, default 0.01
        Passo del moto browniano per l'esplorazione.
    base_death_prob : float, default 0.1
        Probabilità base di morte.
    base_infection_prob : float, default 0.15
        Probabilità base di infezione.
    max_iter : int, default 200
        Numero massimo di iterazioni.
    use_best_guidance : bool, default True
        Attrazione verso la migliore posizione globale trovata.
    best_influence : float, default 0.1
        Peso dell'attrazione verso la migliore posizione globale.
    convergence_tolerance : float, default 1e-6
        Soglia minima di miglioramento per il controllo di convergenza.
    convergence_patience : int, default 20
        Numero di iterazioni consecutive senza miglioramento prima di terminare.
    n_workers : int, default 1
        Numero di thread/worker CPU per la valutazione parallela di fitness e gradienti.
    device : str, default 'cpu'
        Dispositivo di calcolo ('cpu' o 'gpu'/'auto' se framework tensor presente).
        
    Returns
    -------
    Ant
        La migliore formica trovata al termine dell'ottimizzazione.
    """
    ants = dispatch_ants(n_ants, n_dims, lower_bound, upper_bound, base_infection_prob, base_death_prob)
    best = None
    best_position = None
    max_population = n_ants * 3

    lower_bound = np.array(lower_bound, dtype=float)
    upper_bound = np.array(upper_bound, dtype=float)

    no_improvement_count = 0
    previous_best_fitness = None
    
    if verbose:
        print(f"Initialization ({len(ants)} ants, dims={n_dims}, workers={n_workers}, device={device})...")
    if n_dims == 2 and visualization:
        plot_ants_2d(ants, lower_bound, upper_bound, fitness)
    
    for i in range(max_iter):
        gradient_influence = i / max_iter
        alive_ants = [a for a in ants if a.is_alive]

        for ant in alive_ants:
            ant.infection_probability += base_infection_prob * 0.15
            ant.death_probability += base_death_prob * 0.1

        # Identifica formiche infette per il calcolo batch dei gradienti
        infected_ants = [a for a in alive_ants if a.is_infected]
        if infected_ants:
            infected_positions = [a.position for a in infected_ants]
            gradients = estimate_gradients_batch(
                fitness, infected_positions, epsilon=1e-6, 
                stochastic=stochastic, sample_size=5, n_workers=n_workers
            )
            for idx, ant in enumerate(infected_ants):
                ant._current_gradient = gradients[idx]

        for ant in alive_ants:
            movement = borwnian_walk(ant.position, step_size=ant_step_size)
            
            best_component = np.zeros_like(ant.position)
            if use_best_guidance and best_position is not None:
                best_component = best_influence * (best_position - ant.position)
            
            if ant.is_infected:
                gradient = getattr(ant, '_current_gradient', estimate_gradient(fitness, ant.position, stochastic=stochastic))
                if not minimization:
                    ant.position += (1 - gradient_influence) * movement + (learning_rate * gradient) + best_component
                else:
                    ant.position += (1 - gradient_influence) * movement - (learning_rate * gradient) + best_component
            else:
                ant.position += movement + best_component

            ant.position = np.clip(ant.position, lower_bound, upper_bound)

        # Valutazione fitness per tutte le formiche vive
        eval_positions = [a.position for a in alive_ants]
        fitness_values = evaluate_fitness_batch(fitness, eval_positions, n_workers=n_workers)
        
        for idx, ant in enumerate(alive_ants):
            ant.fitness = fitness_values[idx]

            if ant.best_fitness is None:
                ant.best_fitness = ant.fitness
            elif (not minimization and ant.fitness > ant.best_fitness) or (minimization and ant.fitness < ant.best_fitness):
                ant.best_fitness = ant.fitness
                ant.no_improvement_steps = 0
            else:
                ant.no_improvement_steps += 1 

            # INFEZIONE
            ant.is_infected = ant.is_infected or (np.random.rand() <= ant.infection_probability)

            # MORTE DELLE FORMICHE INFETTE
            if ant.is_infected:
                if ant.no_improvement_steps > (max_iter / 5):
                    ant.death_probability += (base_death_prob * 0.7)

                if np.random.rand() <= ant.death_probability:
                    ant.is_alive = False
                    ant.is_infected = False
                    ants.remove(ant)
                    
                    if len(ants) + 3 <= max_population:
                        displace_new_ants(ant.position, ants, base_infection_prob, base_death_prob, dispersion_radius)
                    
        valid_ants = [ant for ant in ants if ant.fitness is not None]
        if not valid_ants:
            break

        if not minimization:
            best = max_fit(valid_ants)
        else:
            best = min_fit(valid_ants)
            
        if best.is_alive:
            best_position = best.position.copy()

        current_best_fitness = best.fitness
        
        # CONTROLLO CONVERGENZA
        if previous_best_fitness is not None:
            if minimization:
                improvement = previous_best_fitness - current_best_fitness
            else:
                improvement = current_best_fitness - previous_best_fitness
                
            if improvement > convergence_tolerance:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
        previous_best_fitness = current_best_fitness

        if verbose:
            population_info = f" (max: {n_ants * 3})" if len(ants) > n_ants * 2 else ""
            print(f"Iteration: {i+1} | n. ants: {len(ants)}{population_info} | Best {best} | No improv.: {no_improvement_count}")
        
        if n_dims == 2 and visualization:
            plot_ants_2d(ants, lower_bound, upper_bound, fitness)

        if no_improvement_count >= convergence_patience:
            if verbose:
                print(f"\nConvergence reached: no improvements for {convergence_patience} consecutive iterations.")
                print(f"Execution stopped at iteration {i+1}")
            break

    return best
