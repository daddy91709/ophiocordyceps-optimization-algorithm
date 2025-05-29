# Ophiocordyceps Optimization Algorithm

A bio-inspired optimization algorithm that simulates the fascinating relationship between **Ophiocordyceps fungi** and their ant hosts. This metaheuristic draws inspiration from one of nature's most remarkable examples of behavioral manipulation for computational optimization.

## Biological Inspiration

The Ophiocordyceps fungus exhibits extraordinary behavior: it infects ants, manipulates their behavior to climb to optimal positions for spore dispersal, and eventually kills them to release spores that can infect new hosts. This biological process creates a natural optimization cycle where:

- **Exploration** occurs through healthy ants randomly foraging
- **Exploitation** happens when infected ants are guided toward optimal locations
- **Information sharing** occurs via spore dispersal from deceased ants
- **Population renewal** maintains diversity through new ant generation

## Algorithm Overview

### Core Phases

1. **Initialization**: Ants are distributed randomly across the solution space
2. **Exploration**: Healthy ants perform random walk movements to explore new areas
3. **Infection**: Ants become infected with increasing probability over time
4. **Exploitation**: Infected ants follow gradient-based movement toward better solutions
5. **Death & Dispersal**: Infected ants eventually die, releasing "spores" that attract new ants to promising regions
6. **Renewal**: New ants are generated near productive areas to maintain population diversity

### Key Theoretical Concepts

- **Dual-phase search**: Combines stochastic exploration (healthy ants) with deterministic exploitation (infected ants)
- **Adaptive behavior**: Infection and death probabilities increase over time, shifting from exploration to exploitation
- **Spatial memory**: Dead ants leave traces (spores) that guide future search efforts
- **Population dynamics**: Maintains balance between exploration and exploitation through controlled population renewal

## Educational Value

This algorithm demonstrates several important optimization concepts:

- **Bio-inspired computing**: How natural processes can inspire computational methods
- **Multi-agent systems**: Emergent behavior from simple individual rules
- **Exploration vs. Exploitation**: The fundamental trade-off in optimization
- **Metaheuristic design**: Combining multiple search strategies
- **Population-based optimization**: Using multiple solution candidates

## Mathematical Foundation

The algorithm combines:
- **Brownian motion** for exploration
- **Gradient descent** for exploitation  
- **Probabilistic transitions** between behavioral states
- **Spatial influence models** for information sharing

## Repository Structure

```
├── notebooks/                    # Jupyter notebook with complete implementation
│   └── ophiocordyceps-optimization-algorithm.ipynb
├── src/                         # Source code modules
│   └── benchmark.py             # Benchmark functions for testing
├── results/                     # Experimental results
└── README.md                    # This file
```

## Benchmark Functions

The algorithm is tested on standard optimization benchmarks including:
- **Sphere**: Simple unimodal function
- **Rastrigin**: Highly multimodal with many local optima
- **Ackley**: Non-convex with global optimum in a narrow basin
- **Griewank**: Multimodal with interdependent variables
- **Levy**: Complex landscape with irregular structure

## Educational Applications

This implementation is designed for:
- Understanding bio-inspired optimization
- Learning metaheuristic algorithm design
- Studying population dynamics in optimization
- Exploring the balance between exploration and exploitation
- Comparing different optimization strategies

## License

MIT License - Free for educational and research purposes.

---

*This project demonstrates how biological systems can inspire computational solutions to complex optimization problems.*
