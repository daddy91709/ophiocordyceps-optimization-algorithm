# Ophiocordyceps Optimization Algorithm

This repository contains an implementation of the **Ophiocordyceps Optimization Algorithm**, a bio-inspired optimization technique modeled after the behavior of ants and the infection process of the Ophiocordyceps fungus. The algorithm is designed to solve optimization problems by exploring and exploiting the solution space.

## Features of the Algorithm

1. **Initialization**: Ants are randomly distributed in the solution space, representing potential solutions.
2. **Exploration**: Ants move based on their personal best solutions, with some randomness to explore new areas.
3. **Infection**: Ants can become infected with a probability, transitioning from exploration to exploitation, guided by the fitness function.
4. **Death and Spore Dispersion**: Infected ants die after a certain threshold, releasing spores that influence nearby ants to explore promising areas.
5. **Iteration End**: Poorly performing areas are eliminated, and new ants are introduced to unexplored regions to maintain diversity.

## Code Structure

### Classes
- **`Ant`**: Represents an ant in the solution space, with attributes for position, fitness, infection status, and life status.
- **`Spore`**: Represents a spore released by dead ants, with attributes for position and age.

### Functions
- **`dispatch_ants`**: Initializes ants with random positions within the solution space.
- **`borwnian_walk`**: Simulates random movement of ants.
- **`update_spores`**: Updates the age of spores and removes expired ones.
- **`ant_in_range`**: Checks if an ant is within the range of a spore.
- **`max_fit` / `min_fit`**: Finds the ant with the best fitness value.
- **Fitness Functions**: Includes benchmark functions like `sphere`, `rastrigin`, `ackley`, `griewank`, `levy`.
- **`estimate_gradient`**: Estimates the gradient of the fitness function for exploitation.

### Visualization
- **`plot`**: Visualizes the positions of ants and spores in a 2D solution space, along with the fitness landscape, showing:
- Healthy ants (blue)
- Infected ants (red)
- Dead ants (black)
- Spores (orange)

### Main Algorithm
- **`ophiocordyceps`**: Implements the optimization algorithm, iterating through exploration, infection, death, and spore dispersion phases.

### Execution
- The algorithm is executed with customizable parameters such as the number of ants, dimensions, bounds, infection probability, death probability, and fitness function.

## How to Run

1. Install the required Python libraries:
   ```bash
   pip install numpy matplotlib

2. Run the script:
   python Ophiocordyceps_Optimization_Algorithm.ipynb

3. Customize parameters in the if __name__ == "__main__": section to suit your optimization problem.

## Parameters

| Parameter              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `n_ants`              | Number of ants in the solution space.                                       |
| `n_dims`              | Dimensionality of the solution space.                                       |
| `lower_bound`         | Lower bounds of the solution space.                                         |
| `upper_bound`         | Upper bounds of the solution space.                                         |
| `fitness`             | Fitness function to optimize.                                               |
| `minimization`        | Whether to minimize (`True`) or maximize (`False`) the fitness function.    |
| `spore_age_limit`     | Maximum age of spores before they disappear.                                |
| `spore_radius`        | Radius of influence of spores.                                              |
| `base_infection_prob` | Base probability of ants becoming infected.                                 |
| `death_prob`          | Probability of infected ants dying.                                         |
| `max_iter`            | Maximum number of iterations.                                               |
| `learning_rate`       | Learning rate for gradient-based exploitation.                              |
| `ant_step_size`       | Step size for random movement of ants.                                      |

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it as needed.
```
