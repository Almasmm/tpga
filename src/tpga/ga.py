"""Permutation-encoded genetic algorithm with the greedy decoder."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from tpga.decoder import decode_permutation
from tpga.model import FloatArray, TransportProblem

IntArray = np.ndarray


@dataclass(frozen=True)
class GAConfig:
    """Parameters of the genetic algorithm."""

    population: int = 60
    generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    tournament: int = 3
    elite: int = 2
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.population < 2:
            raise ValueError("population must be at least 2")
        if self.generations < 0:
            raise ValueError("generations must be nonnegative")
        if not (0.0 <= self.crossover_rate <= 1.0 and 0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("rates must lie in [0, 1]")
        if not (1 <= self.tournament <= self.population):
            raise ValueError("tournament size must lie in [1, population]")
        if not (0 <= self.elite < self.population):
            raise ValueError("elite must lie in [0, population)")


@dataclass
class GAResult:
    """Best plan found and the run record."""

    x: FloatArray
    cost: float
    history: list[float] = field(default_factory=list)  # best cost after each generation
    evaluations: int = 0
    runtime_s: float = 0.0


def order_crossover(p1: IntArray, p2: IntArray, rng: np.random.Generator) -> IntArray:
    """Order crossover (OX1): keep a slice of ``p1``, fill the rest in ``p2`` order."""
    n = p1.size
    if n < 2:
        return p1.copy()
    i, j = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1, dtype=p1.dtype)
    child[i : j + 1] = p1[i : j + 1]
    kept = set(child[i : j + 1].tolist())
    child[child < 0] = [g for g in p2 if g not in kept]
    return child


def swap_mutation(p: IntArray, rng: np.random.Generator) -> IntArray:
    """Swap two random positions."""
    q = p.copy()
    if q.size >= 2:
        i, j = rng.choice(q.size, size=2, replace=False)
        q[i], q[j] = q[j], q[i]
    return q


def run_ga(problem: TransportProblem, config: GAConfig | None = None) -> GAResult:
    """Run the GA. Every individual decodes to a feasible plan of the balanced classical model,
    so no penalty or repair is used. This is a reference implementation, not a tuned solver."""
    cfg = config or GAConfig()
    t0 = time.perf_counter()
    p = problem.balanced()
    m, n = p.shape
    length = m * n
    rng = np.random.default_rng(cfg.seed)

    def fitness(ind: IntArray) -> float:
        return p.total_cost(decode_permutation(p, ind))

    pop = [rng.permutation(length) for _ in range(cfg.population)]
    fit = np.array([fitness(ind) for ind in pop])
    evaluations = len(pop)
    history = [float(fit.min())]

    def tournament() -> IntArray:
        idx = rng.choice(cfg.population, size=cfg.tournament, replace=False)
        return pop[int(idx[np.argmin(fit[idx])])]

    for _ in range(cfg.generations):
        order = np.argsort(fit)
        new = [pop[int(k)].copy() for k in order[: cfg.elite]]
        while len(new) < cfg.population:
            a, b = tournament(), tournament()
            child = order_crossover(a, b, rng) if rng.random() < cfg.crossover_rate else a.copy()
            if rng.random() < cfg.mutation_rate:
                child = swap_mutation(child, rng)
            new.append(child)
        pop = new
        fit = np.array([fitness(ind) for ind in pop])
        evaluations += len(pop)
        history.append(float(min(fit.min(), history[-1])))

    best = pop[int(np.argmin(fit))]
    x = decode_permutation(p, best)
    return GAResult(
        x=x, cost=p.total_cost(x), history=history, evaluations=evaluations, runtime_s=time.perf_counter() - t0
    )
