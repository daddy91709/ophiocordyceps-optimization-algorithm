"""
Unit and Integration Tests for Ophiocordyceps Optimization Algorithm
"""
import pytest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ophiocordyceps import (
    Ant, ophiocordyceps, dispatch_ants, borwnian_walk, 
    levy_flight, estimate_gradient, displace_new_ants
)
import src.benchmark as bm
from src.device import get_cpu_info, detect_gpu
from src.gpu_backend import GPUTensorBackend


def test_ant_initialization():
    pos = [1.0, 2.0]
    ant = Ant(position=pos, infection_probability=0.2, death_probability=0.05)
    assert np.allclose(ant.position, [1.0, 2.0])
    assert ant.infection_probability == 0.2
    assert ant.death_probability == 0.05
    assert ant.is_infected is False
    assert ant.is_alive is True
    assert ant.best_fitness is None


def test_dispatch_ants():
    ants = dispatch_ants(n=20, dims=3, lower_bound=[-5, -5, -5], upper_bound=[5, 5, 5], infection=0.1, death=0.05)
    assert len(ants) == 20
    for a in ants:
        assert len(a.position) == 3
        assert np.all(a.position >= -5)
        assert np.all(a.position <= 5)


def test_gradient_estimation():
    # Funzione quadratica f(x, y) = x^2 + y^2, gradiente esatto = [2x, 2y]
    def f(x):
        return x[0]**2 + x[1]**2
    
    pos = np.array([3.0, 4.0])
    grad = estimate_gradient(f, pos, epsilon=1e-6)
    assert np.allclose(grad, [6.0, 8.0], atol=1e-4)


def test_displace_new_ants():
    ants = []
    base_pos = np.array([0.0, 0.0])
    displace_new_ants(base_pos, ants, infection=0.1, death=0.05, radius=0.2)
    assert 1 <= len(ants) <= 3
    for a in ants:
        assert len(a.position) == 2


def test_optimization_sphere():
    # Test convergenza su Sphere Function 2D (minimo 0 in 0,0)
    np.random.seed(42)
    best = ophiocordyceps(
        n_ants=25,
        n_dims=2,
        lower_bound=[-5, -5],
        upper_bound=[5, 5],
        fitness=bm.sphere,
        minimization=True,
        max_iter=40,
        convergence_patience=10
    )
    assert best.fitness is not None
    assert best.fitness < 0.05
    assert np.allclose(best.position, [0, 0], atol=0.2)


def test_optimization_multi_worker():
    # Test con valutazione parallela multi-thread su CPU
    np.random.seed(42)
    best_single = ophiocordyceps(
        n_ants=20,
        n_dims=2,
        lower_bound=[-5, -5],
        upper_bound=[5, 5],
        fitness=bm.sphere,
        minimization=True,
        max_iter=20,
        n_workers=1
    )
    np.random.seed(42)
    best_multi = ophiocordyceps(
        n_ants=20,
        n_dims=2,
        lower_bound=[-5, -5],
        upper_bound=[5, 5],
        fitness=bm.sphere,
        minimization=True,
        max_iter=20,
        n_workers=4
    )
    assert best_single.fitness is not None
    assert best_multi.fitness is not None
    assert best_single.fitness < 0.5
    assert best_multi.fitness < 0.5


def test_device_detection():
    cpu = get_cpu_info()
    assert cpu["cores_logical"] >= 1
    assert cpu["cores_recommended"] >= 1
    
    gpu = detect_gpu()
    assert isinstance(gpu, dict)
    assert "available" in gpu


def test_gpu_backend_initialization():
    backend = GPUTensorBackend()
    info = backend.get_info()
    assert isinstance(info, dict)
    assert "torch_available" in info
