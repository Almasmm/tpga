"""Exact baseline: linear programming with the HiGHS solver."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from tpga.model import FloatArray, TransportProblem


def solve_lp(problem: TransportProblem) -> tuple[FloatArray, float]:
    """Return the optimal plan and cost of the (balanced) classical model.

    Exact methods apply directly to the classical model, so this optimum is the reference
    against which any heuristic result on it should be reported.
    """
    p = problem.balanced()
    m, n = p.shape
    a_eq = np.zeros((m + n, m * n))
    for i in range(m):
        a_eq[i, i * n : (i + 1) * n] = 1.0
    for j in range(n):
        a_eq[m + j, j::n] = 1.0
    b_eq = np.concatenate([p.supply, p.demand])
    res = linprog(p.cost.ravel(), A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:  # pragma: no cover - a balanced classical problem is always feasible
        raise RuntimeError(f"LP solver failed: {res.message}")
    return res.x.reshape(m, n), float(res.fun)
