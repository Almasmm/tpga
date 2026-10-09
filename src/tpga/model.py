"""Data model of the classical transportation problem."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TransportProblem:
    """Classical transportation problem.

    Minimise ``sum_ij c_ij * x_ij`` subject to ``sum_j x_ij = a_i`` (supply),
    ``sum_i x_ij = b_j`` (demand) and ``x_ij >= 0``.

    Parameters
    ----------
    supply:
        Supplies ``a_i`` of the ``m`` sources, nonnegative.
    demand:
        Demands ``b_j`` of the ``n`` destinations, nonnegative.
    cost:
        Unit transport costs ``c_ij``, shape ``(m, n)``.
    """

    supply: FloatArray
    demand: FloatArray
    cost: FloatArray

    def __init__(self, supply: ArrayLike, demand: ArrayLike, cost: ArrayLike) -> None:
        a = np.asarray(supply, dtype=float).ravel()
        b = np.asarray(demand, dtype=float).ravel()
        c = np.asarray(cost, dtype=float)
        if a.size == 0 or b.size == 0:
            raise ValueError("supply and demand must be non-empty")
        if c.shape != (a.size, b.size):
            raise ValueError(f"cost shape {c.shape} does not match {a.size} sources x {b.size} destinations")
        if (a < 0).any() or (b < 0).any():
            raise ValueError("supply and demand must be nonnegative")
        if not np.isfinite(c).all():
            raise ValueError("costs must be finite")
        object.__setattr__(self, "supply", a)
        object.__setattr__(self, "demand", b)
        object.__setattr__(self, "cost", c)

    @property
    def shape(self) -> tuple[int, int]:
        """Number of sources and destinations, ``(m, n)``."""
        m, n = self.cost.shape
        return int(m), int(n)

    @property
    def is_balanced(self) -> bool:
        """True if total supply equals total demand."""
        return bool(np.isclose(self.supply.sum(), self.demand.sum()))

    def balanced(self) -> TransportProblem:
        """Return a balanced copy, adding a zero-cost dummy source or destination if needed."""
        s, d = float(self.supply.sum()), float(self.demand.sum())
        if np.isclose(s, d):
            return self
        m, n = self.shape
        if s > d:
            return TransportProblem(
                self.supply, np.append(self.demand, s - d), np.hstack([self.cost, np.zeros((m, 1))])
            )
        return TransportProblem(np.append(self.supply, d - s), self.demand, np.vstack([self.cost, np.zeros((1, n))]))

    def total_cost(self, x: ArrayLike) -> float:
        """Objective value of a shipment plan ``x``."""
        return float((self.cost * np.asarray(x, dtype=float)).sum())

    def is_feasible(self, x: ArrayLike, rtol: float = 1e-9) -> bool:
        """Check nonnegativity and the supply and demand equalities."""
        x = np.asarray(x, dtype=float)
        if x.shape != self.cost.shape:
            return False
        scale = max(1.0, float(np.max(self.supply)), float(np.max(self.demand)))
        tol = rtol * scale
        return bool(
            (x >= -tol).all()
            and np.allclose(x.sum(axis=1), self.supply, atol=tol)
            and np.allclose(x.sum(axis=0), self.demand, atol=tol)
        )

    # --- input / output -------------------------------------------------------------------
    def to_dict(self) -> dict[str, list]:
        return {"supply": self.supply.tolist(), "demand": self.demand.tolist(), "cost": self.cost.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> TransportProblem:
        try:
            return cls(data["supply"], data["demand"], data["cost"])
        except KeyError as exc:
            raise ValueError(f"instance is missing the field {exc.args[0]!r}") from None

    @classmethod
    def from_json(cls, path: str | Path) -> TransportProblem:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def random(
        cls,
        m: int,
        n: int,
        total: int = 1000,
        cost_range: tuple[int, int] = (1, 20),
        seed: int | None = None,
    ) -> TransportProblem:
        """Random balanced instance with integer supplies, demands and costs."""
        if m < 1 or n < 1:
            raise ValueError("m and n must be positive")
        rng = np.random.default_rng(seed)
        a = rng.multinomial(total, np.full(m, 1 / m)).astype(float)
        b = rng.multinomial(total, np.full(n, 1 / n)).astype(float)
        c = rng.integers(cost_range[0], cost_range[1] + 1, size=(m, n)).astype(float)
        return cls(a, b, c)
