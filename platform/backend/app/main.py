"""tp-ga platform API (skeleton).

Exposes the core library over HTTP.  Long GA runs will move to a job queue (Redis + RQ) in a
later milestone; for now /solve runs synchronously and is limited to small instances."""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tpga import GAConfig, TransportProblem, run_ga, solve_lp

MAX_ARCS = 400  # synchronous limit until the job queue exists

app = FastAPI(title="tp-ga", version="0.1.0")


class SolveRequest(BaseModel):
    supply: list[float] = Field(..., min_length=1)
    demand: list[float] = Field(..., min_length=1)
    cost: list[list[float]]
    method: Literal["lp", "ga"] = "lp"
    generations: int = Field(200, ge=1, le=5000)
    population: int = Field(60, ge=4, le=1000)
    seed: int | None = None


class SolveResponse(BaseModel):
    method: str
    cost: float
    plan: list[list[float]]
    balanced_with_dummy: bool
    lp_optimum: float | None = None
    gap_percent: float | None = None
    evaluations: int | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    try:
        p = TransportProblem(req.supply, req.demand, req.cost)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if p.cost.size > MAX_ARCS:
        raise HTTPException(413, f"instance has {p.cost.size} arcs; synchronous limit is {MAX_ARCS}")
    dummy = not p.is_balanced
    x_lp, z_lp = solve_lp(p)
    if req.method == "lp":
        return SolveResponse(method="lp", cost=z_lp, plan=x_lp.tolist(), balanced_with_dummy=dummy, lp_optimum=z_lp)
    r = run_ga(p, GAConfig(population=req.population, generations=req.generations, seed=req.seed))
    gap = None if z_lp == 0 else 100.0 * (r.cost - z_lp) / abs(z_lp)
    # The LP optimum is always returned alongside a GA result: on the classical model exact
    # methods apply, so a heuristic result is only interpretable relative to it.
    return SolveResponse(
        method="ga",
        cost=r.cost,
        plan=r.x.tolist(),
        balanced_with_dummy=dummy,
        lp_optimum=z_lp,
        gap_percent=gap,
        evaluations=r.evaluations,
    )
