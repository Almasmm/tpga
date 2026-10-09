import json

import numpy as np
import pytest

from tpga import TransportProblem


def test_shape_and_balance(small):
    assert small.shape == (3, 3)
    assert small.is_balanced


@pytest.mark.parametrize(
    "supply, demand, cost, msg",
    [
        ([1, 2], [3], [[1, 2]], "does not match"),
        ([-1], [1], [[1]], "nonnegative"),
        ([], [1], [[]], "non-empty"),
        ([1], [1], [[np.inf]], "finite"),
    ],
)
def test_invalid_input(supply, demand, cost, msg):
    with pytest.raises(ValueError, match=msg):
        TransportProblem(supply, demand, cost)


def test_balancing_with_dummy_destination():
    p = TransportProblem([10, 5], [8, 4], [[1, 2], [3, 4]]).balanced()
    assert p.shape == (2, 3) and p.is_balanced and p.demand[-1] == 3


def test_balancing_with_dummy_source():
    p = TransportProblem([4], [3, 3], [[1, 1]]).balanced()
    assert p.shape == (2, 2) and p.is_balanced and p.supply[-1] == 2


def test_balanced_returns_self_when_balanced(small):
    assert small.balanced() is small


def test_feasibility_check(small):
    x = np.array([[0, 0, 20], [5, 25, 0], [20, 0, 0]], dtype=float)
    assert small.is_feasible(x)
    assert small.total_cost(x) == 225
    x[0, 2] = 19
    assert not small.is_feasible(x)
    assert not small.is_feasible(np.zeros((2, 2)))


def test_json_round_trip(tmp_path, small):
    path = tmp_path / "inst.json"
    small.to_json(path)
    again = TransportProblem.from_json(path)
    assert np.array_equal(again.cost, small.cost) and json.loads(path.read_text())["supply"] == [20, 30, 20]


def test_from_dict_missing_field():
    with pytest.raises(ValueError, match="cost"):
        TransportProblem.from_dict({"supply": [1], "demand": [1]})


def test_random_is_balanced_and_reproducible():
    p1, p2 = TransportProblem.random(4, 5, seed=3), TransportProblem.random(4, 5, seed=3)
    assert p1.is_balanced and np.array_equal(p1.cost, p2.cost)
    with pytest.raises(ValueError):
        TransportProblem.random(0, 3)
