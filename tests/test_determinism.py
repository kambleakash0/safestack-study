import random

from safestack.determinism import set_seeds


def test_set_seeds_runs_without_torch():
    # Must not raise even when torch/numpy are absent (the base, torch-free env).
    set_seeds(0)


def test_set_seeds_makes_random_reproducible():
    set_seeds(123)
    a = [random.random() for _ in range(5)]
    set_seeds(123)
    b = [random.random() for _ in range(5)]
    assert a == b
