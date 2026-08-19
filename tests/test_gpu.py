"""
Unit tests for OOA GPU Acceleration Engine.
"""
import pytest
import numpy as np
from src.ophiocordyceps import ophiocordyceps, Ant
from src.ophiocordyceps_gpu import ophiocordyceps_gpu, get_torch_device
import src.benchmark as bm


def test_gpu_device_detection():
    dev = get_torch_device("auto")
    assert dev is not None
    assert dev.type in ["cuda", "mps", "privateuseone", "cpu"]


def test_gpu_execution_sphere_10d():
    res = ophiocordyceps_gpu(
        n_ants=30,
        n_dims=10,
        lower_bound=[-5.12] * 10,
        upper_bound=[5.12] * 10,
        fitness=bm.sphere,
        minimization=True,
        max_iter=30,
        device="auto"
    )
    assert "position" in res
    assert "fitness" in res
    assert len(res["position"]) == 10
    assert res["fitness"] < 1.0


def test_gpu_routing_via_main_api():
    best = ophiocordyceps(
        n_ants=25,
        n_dims=5,
        lower_bound=[-5.0] * 5,
        upper_bound=[5.0] * 5,
        fitness=bm.ackley,
        minimization=True,
        max_iter=25,
        device="gpu"
    )
    assert isinstance(best, Ant)
    assert best.fitness is not None
    assert len(best.position) == 5
