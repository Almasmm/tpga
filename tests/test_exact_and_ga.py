import numpy as np
import pytest

from conftest import SMALL_OPT
from tpga import GAConfig, TransportProblem, decode_permutation, run_ga, solve_lp
from tpga.ga import order_crossover, swap_mutation


def test_lp_known_optimum(small):
    x, z = solve_lp(small)
    assert small.is_feasible(x)
    assert z == pytest.approx(SMALL_OPT)


def test_lp_handles_unbalanced_instance():
    x, z = solve_lp(TransportProblem([10, 5], [8, 4], [[1, 2], [3, 4]]))
    assert x.shape == (2, 3) and z == pytest.approx(8 * 1 + 2 * 2 + 2 * 4)


def test_lp_is_a_lower_bound_for_every_decoded_plan():
    p = TransportProblem.random(5, 6, seed=3)
    _, z = solve_lp(p)
    rng = np.random.default_rng(0)
    assert all(p.total_cost(decode_permutation(p, rng.permutation(30))) >= z - 1e-6 for _ in range(300))


def test_ga_feasible_monotone_and_not_below_lp():
    p = TransportProblem.random(4, 5, seed=7)
    _, z = solve_lp(p)
    r = run_ga(p, GAConfig(population=30, generations=40, seed=1))
    assert p.is_feasible(r.x)
    assert r.cost >= z - 1e-6
    assert all(b <= a + 1e-9 for a, b in zip(r.history, r.history[1:], strict=False))
    assert len(r.history) == 41 and r.evaluations == 30 * 41 and r.runtime_s > 0


def test_ga_finds_optimum_of_small_instance(small):
    assert run_ga(small, GAConfig(population=40, generations=60, seed=0)).cost == pytest.approx(SMALL_OPT)


def test_ga_is_reproducible_with_seed(small):
    a = run_ga(small, GAConfig(population=20, generations=10, seed=5))
    b = run_ga(small, GAConfig(population=20, generations=10, seed=5))
    assert a.history == b.history


def test_ga_default_config_and_zero_generations(small):
    r = run_ga(small, GAConfig(generations=0, seed=1))
    assert len(r.history) == 1 and small.is_feasible(r.x)
    assert run_ga(TransportProblem([1], [1], [[3]])).cost == 3


@pytest.mark.parametrize(
    "kwargs",
    [{"population": 1}, {"generations": -1}, {"crossover_rate": 1.5}, {"tournament": 0}, {"elite": 60}],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        GAConfig(**kwargs)


def test_operators_preserve_permutations():
    rng = np.random.default_rng(0)
    for _ in range(100):
        p1, p2 = rng.permutation(12), rng.permutation(12)
        assert sorted(order_crossover(p1, p2, rng)) == list(range(12))
        assert sorted(swap_mutation(p1, rng)) == list(range(12))
    one = np.array([0])
    assert order_crossover(one, one, rng).tolist() == [0] and swap_mutation(one, rng).tolist() == [0]
