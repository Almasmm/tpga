import numpy as np
import pytest

from tpga import TransportProblem, decode_permutation


@pytest.mark.parametrize("seed", range(50))
def test_every_permutation_decodes_to_a_feasible_plan(seed):
    rng = np.random.default_rng(seed)
    p = TransportProblem.random(int(rng.integers(1, 9)), int(rng.integers(1, 9)), seed=seed)
    x = decode_permutation(p, rng.permutation(p.cost.size))
    assert p.is_feasible(x)


def test_identity_permutation_is_northwest_corner(small):
    # Visiting arcs row by row reproduces the north-west corner rule.
    x = decode_permutation(small, np.arange(9))
    assert np.array_equal(x, [[20, 0, 0], [5, 25, 0], [0, 0, 20]])


def test_rejects_unbalanced():
    with pytest.raises(ValueError, match="balanced"):
        decode_permutation(TransportProblem([10, 5], [8, 4], [[1, 2], [3, 4]]), np.arange(4))


@pytest.mark.parametrize("perm", [np.arange(8), np.array([0, 0, 1, 2, 3, 4, 5, 6, 7]), np.arange(1, 10)])
def test_rejects_non_permutations(small, perm):
    with pytest.raises(ValueError, match="permutation"):
        decode_permutation(small, perm)
