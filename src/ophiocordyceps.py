import numpy as np

# CLASSE ANT E FUNZIONI DI SUPPORTO

class Ant:
    def __init__(self, position, infection_probability=0.1, death_probability=0.1, fitness=None):
        self.position = np.array(position, dtype=float)  # Vettore posizione nello spazio degli stati
        self.fitness = fitness                            # Valore di fitness (None finché non viene calcolato)
        self.infection_probability = infection_probability  # Probabilità di infezione
        self.death_probability = death_probability          # Probabilità di morte quando infetta
        self.is_infected = False                          # Flag per indicare se la formica è infetta
        self.is_alive = True                              # Flag per indicare se la formica è in vita
        self.no_improvement_steps = 0                     # Iterazioni senza miglioramenti
        self.best_fitness = fitness                       # Miglior fitness raggiunto finora
        
    def __str__(self):
        return f"ant at: {self.position}, fitness/cost: {self.fitness}, infected: {self.is_infected}, alive: {self.is_alive}"


def dispatch_ants(n, dims, lower_bound, upper_bound, infection, death):
    ants = []
    lower_bound = np.array(lower_bound, dtype=float)
    upper_bound = np.array(upper_bound, dtype=float)
    
    if len(lower_bound) != dims or len(upper_bound) != dims:
        raise ValueError(f"I limiti devono avere lunghezza {dims}")
    
    for _ in range(n):
        position = lower_bound + np.random.random(dims) * (upper_bound - lower_bound)
        ant = Ant(position, infection_probability=infection, death_probability=death)
        ants.append(ant)
    
    return ants


def borwnian_walk(coords, step_size=0.01, sigma=1.0):
    """Simula il moto browniano."""
    return sigma * np.sqrt(step_size) * np.random.normal(0, 1, size=coords.shape)


def levy_flight(coords, alpha=1.5, step_size=0.01, lower_bound=None, upper_bound=None):
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


def displace_new_ants(position, ants, infection, death, radius=0.1):
    """Genera nuove formiche attorno alla posizione di una formica morta (spore)."""
    for _ in range(np.random.randint(1, 4)):
        offset = np.random.normal(0, radius / 2, size=len(position))
        ants.append(Ant(position + offset, 
                        infection_probability=infection, 
                        death_probability=death))


def max_fit(ants):
    best_ant = Ant(ants[0].position, fitness=-np.inf)
    for ant in ants:
        if ant.fitness is not None and ant.fitness > best_ant.fitness:
            best_ant = ant
    return best_ant


def min_fit(ants):
    best_ant = Ant(ants[0].position, fitness=np.inf)
    for ant in ants:
        if ant.fitness is not None and ant.fitness < best_ant.fitness:
            best_ant = ant
    return best_ant


def estimate_gradient(fitness, position, epsilon=1e-6, stochastic=False, sample_size=5):
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


def plot_ants_2d(ants, lower_bound, upper_bound, fitness_function, resolution=50,
                 title="Ophiocordyceps Optimization", cmap="viridis", figsize=(8, 6)):
    """Visualizzazione opzionale per problemi 2D (richiede matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.patches import Patch
    except ImportError:
        print("[Warning] Matplotlib non disponibile per la visualizzazione 2D.")
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


def ophiocordyceps(n_ants, n_dims, lower_bound, upper_bound, fitness, 
                   minimization=True, visualization=False, verbose=False, stochastic=False,
                   dispersion_radius=0.2, 
                   learning_rate=0.05, ant_step_size=0.01, 
                   base_death_prob=0.1, base_infection_prob=0.15,
                   max_iter=200, use_best_guidance=True, best_influence=0.1,
                   convergence_tolerance=1e-6, 
                   convergence_patience=20):
    """
    Ophiocordyceps Optimization Algorithm.
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
        print("Initialization...")
    if n_dims == 2 and visualization:
        plot_ants_2d(ants, lower_bound, upper_bound, fitness)
    
    for i in range(max_iter):
        gradient_influence = i / max_iter

        for ant in [a for a in ants if a.is_alive]:
            ant.infection_probability += base_infection_prob * 0.15
            ant.death_probability += base_death_prob * 0.1
        
            movement = borwnian_walk(ant.position, step_size=ant_step_size)
            
            best_component = np.zeros_like(ant.position)
            if use_best_guidance and best_position is not None:
                best_component = best_influence * (best_position - ant.position)
            
            if ant.is_infected:
                gradient = estimate_gradient(fitness, ant.position, stochastic=stochastic)
                if not minimization:
                    ant.position += (1 - gradient_influence) * movement + (learning_rate * gradient) + best_component
                else:
                    ant.position += (1 - gradient_influence) * movement - (learning_rate * gradient) + best_component
            else:
                ant.position += movement + best_component

            ant.position = np.clip(ant.position, lower_bound, upper_bound)
                
            ant.fitness = fitness(ant.position)

            if ant.best_fitness is None:
                ant.best_fitness = ant.fitness
            elif (not minimization and ant.fitness > ant.best_fitness) or (minimization and ant.fitness < ant.best_fitness):
                ant.best_fitness = ant.fitness
                ant.no_improvement_steps = 0
            else:
                ant.no_improvement_steps += 1 

            ant.is_infected = ant.is_infected or (np.random.rand() <= ant.infection_probability)

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
