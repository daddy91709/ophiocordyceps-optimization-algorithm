# Ophiocordyceps Optimization Algorithm (OOA)

A bio-inspired metaheuristic optimization algorithm based on the biological behavior and life cycle of the **Ophiocordyceps** fungus (*zombie-ant fungus*), featuring **multi-core CPU parallelization** and **GPU acceleration support**.

---

## 🐜 Biological Inspiration

The algorithm models the natural interaction between the *Ophiocordyceps* fungus and ant hosts:
- **Exploration (Healthy Ants)**: Uninfected ants perform random stochastic walks (Brownian motion / Levy flights) to explore the search space.
- **Exploitation (Infected Ants)**: Ants become infected over time; their movement is guided by numerical gradient estimation towards optimal fitness regions combined with social attraction towards the best-known position.
- **Death & Spore Dispersal**: Infected ants that stagnate without improvements eventually die and release spores in their area.
- **Population Renewal**: New ants spawn near productive spore regions to maintain population diversity and avoid local minima traps.

---

## ⚡ Performance & Acceleration Architecture

- **Multi-Core CPU Parallelism**:
  - `run_benchmark.py`: Distributes independent optimization runs across all available CPU cores (e.g. 11-12 parallel workers on 12-core systems) with linear speedups.
  - `src/ophiocordyceps.py`: Multi-threaded batch evaluation of fitness and gradients (`n_workers` parameter) for complex functions.
- **GPU & Tensor Acceleration (`src/gpu_backend.py`)**:
  - Hardware discovery engine (`src/device.py`) auto-detects NVIDIA CUDA GPUs, Intel/AMD DirectML, Apple Silicon MPS, or CPU vectorization.
  - Population-wide tensor operations (bounds clipping, random normal draws, gradient updates).

---

## 📂 Repository Structure

```
ophiocordyceps-optimization-algorithm/
│
├── src/                                  # Source modules
│   ├── __init__.py                       # Package exports
│   ├── ophiocordyceps.py                 # Core algorithm with multi-worker CPU evaluation
│   ├── gpu_backend.py                    # GPU tensor backend (PyTorch / CuPy / DirectML)
│   ├── device.py                         # Hardware discovery & device capability engine
│   ├── benchmark.py                      # Benchmark test functions (Ackley, Sphere, Alpine1, etc.)
│   └── analysis.py                       # Statistical analysis and visualization utilities
│
├── notebooks/                            # Interactive Notebooks
│   ├── demo.ipynb                        # Interactive tutorial explaining algorithm & parallelism
│   ├── ophiocordyceps-optimization-algorithm.ipynb # Original notebook
│   └── print-graphs.ipynb                # Results plotting notebook
│
├── tests/                                # Test Suite
│   └── test_ophiocordyceps.py            # Unit & integration tests
│
├── results/
│   └── risultati.csv                     # Benchmark results output
│
├── main.py                               # Quick demo & hardware check CLI
├── run_benchmark.py                      # Multi-core parallel benchmark runner CLI
├── analyze_results.py                    # Benchmark results analysis CLI
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Quick Demo & Hardware Check

Run the optimization demo with hardware auto-detection:
```bash
python main.py
```

### 3. Running Multi-Core Parallel Benchmarks

Run the benchmark suite in parallel across all CPU cores (`-j -1` or specify worker count):
```bash
# Auto-parallel benchmark on default functions across all CPU cores
python run_benchmark.py -j -1

# Custom benchmark execution
python run_benchmark.py --funcs Ackley Sphere Alpine1 Bohachevsky --dims 2 10 30 --runs 10 --jobs 8
```

### 4. Running the Test Suite

```bash
python -m pytest tests/
```

### 5. Analyzing Results

Inspect and summarize the benchmark results:
```bash
python analyze_results.py --csv results/risultati.csv
```

### 6. Interactive Demo Notebook

Launch Jupyter Notebook to explore the interactive visual guide:
```bash
jupyter notebook notebooks/demo.ipynb
```

---

## 💻 Python Usage Example

```python
import numpy as np
from src.ophiocordyceps import ophiocordyceps
import src.benchmark as bm

# Optimize a 10D Sphere function using 4 parallel CPU evaluation workers
best_ant = ophiocordyceps(
    n_ants=40,
    n_dims=10,
    lower_bound=[-5.12] * 10,
    upper_bound=[5.12] * 10,
    fitness=bm.sphere,
    minimization=True,
    use_best_guidance=True,
    max_iter=50,
    n_workers=4
)

print(f"Best fitness: {best_ant.fitness:.6f}")
print(f"Best position: {best_ant.position}")
```

---

## 📜 License

MIT License - Free for educational and research purposes.
