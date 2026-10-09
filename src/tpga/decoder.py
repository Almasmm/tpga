"""Greedy decoder: permutation of arcs -> feasible shipment plan."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from tpga.model import FloatArray, TransportProblem


def decode_permutation(problem: TransportProblem, perm: ArrayLike) -> FloatArray:
    """Decode a permutation of the ``m * n`` arcs into a shipment plan.

    Arcs are visited in permutation order (arc ``k`` is ``(k // n, k % n)``) and each receives
    ``x_ij = min(remaining a_i, remaining b_j)``.

    Guarantee and its limits: for a *balanced* problem with nonnegative supplies and demands and
    a complete arc set, the result satisfies the supply, demand and nonnegativity constraints.
    The decoder does not handle arc capacities, fixed charges or time windows, and it makes no
    claim about optimality or about which feasible plans are reachable. Cost: ``O(m * n)``.

    Raises
    ------
    ValueError
        If the problem is unbalanced or ``perm`` is not a permutation of ``range(m * n)``.
    """
    if not problem.is_balanced:
        raise ValueError("the decoder requires a balanced problem; call problem.balanced() first")
    m, n = problem.shape
    p = np.asarray(perm, dtype=np.int64).ravel()
    if p.size != m * n or not np.array_equal(np.sort(p), np.arange(m * n)):
        raise ValueError(f"perm must be a permutation of range({m * n})")
    a = problem.supply.copy()
    b = problem.demand.copy()
    x = np.zeros((m, n))
    for k in p:
        i, j = divmod(int(k), n)
        q = min(a[i], b[j])
        if q > 0:
            x[i, j] = q
            a[i] -= q
            b[j] -= q
    return x
