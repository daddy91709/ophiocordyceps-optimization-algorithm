# Ophiocordyceps Optimization Algorithm (OOA)

A bio-inspired metaheuristic optimization algorithm based on the biological behavior and life cycle of the **Ophiocordyceps** fungus (*zombie-ant fungus*).

---

## 🐜 Biological Inspiration

The algorithm models the natural interaction between the *Ophiocordyceps* fungus and ant hosts:
- **Exploration (Healthy Ants)**: Uninfected ants perform random stochastic walks (Brownian motion / Levy flights) to explore the solution space.
- **Exploitation (Infected Ants)**: Ants become infected over time; their movement is manipulated by gradient estimation towards promising fitness regions combined with social attraction towards the best-known solution.
- **Death & Spore Dispersal**: Infected ants that fail to find improvements eventually die and release spores in their area.
- **Population Renewal**: New ants spawn near productive regions to maintain diversity and prevent stagnation.

---

## 📂 Repository Structure

```
.
├── src/
│   ├── __init__.py                # Package exports (Ant, ophiocordyceps, benchmark)
│   ├── ophiocordyceps.py          # Core optimization algorithm implementation
│   ├── benchmark.py               # Benchmark test functions (Ackley, Sphere, Alpine1, etc.)
│   └── analysis.py                # Results loading, statistical analysis, and plotting utilities
│
├── notebooks/
│   ├── demo.ipynb                 # Interactive demo notebook explaining the algorithm
│   ├── ophiocordyceps-optimization-algorithm.ipynb # Original notebook
│   └── print-graphs.ipynb         # Graphs notebook
│
├── results/
│   └── risultati.csv              # Benchmark results output
│
├── main.py                        # Standalone demo & quick-test script
├── run_benchmark.py               # Configurable benchmark runner CLI
├── analyze_results.py             # Benchmark results analysis CLI
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
```

---

## 🚀 Getting Started

### 1. Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Quick Demo (Standard Python)

Run a fast optimization test across benchmark functions directly from the command line:
```bash
python main.py
```

### 3. Running Benchmarks

Execute the automated benchmark suite:
```bash
# Run benchmark on default functions (Ackley, Sphere, Alpine1)
python run_benchmark.py

# Custom benchmark execution
python run_benchmark.py --funcs Ackley Sphere Booth --dims 2 10 30 --runs 10
```

### 4. Analyzing Results

Inspect and summarize the benchmark results:
```bash
python analyze_results.py --csv results/risultati.csv
```

### 5. Interactive Demo Notebook

Open the demo notebook in Jupyter for visual step-by-step explanations:
```bash
jupyter notebook notebooks/demo.ipynb
```

---

## 💻 Python Usage Example

```python
import numpy as np
from src.ophiocordyceps import ophiocordyceps
import src.benchmark as bm

# Optimize a 2D Ackley function
best_ant = ophiocordyceps(
    n_ants=30,
    n_dims=2,
    lower_bound=[-5, -5],
    upper_bound=[5, 5],
    fitness=bm.ackley,
    minimization=True,
    use_best_guidance=True,
    verbose=True,
    max_iter=50
)

print(f"Best fitness: {best_ant.fitness:.6f}")
print(f"Best position: {best_ant.position}")
```

---

## 📜 License

MIT License - Free for educational and research purposes.
