"""Command-line interface: ``tpga solve`` and ``tpga generate``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from tpga import __version__
from tpga.exact import solve_lp
from tpga.ga import GAConfig, run_ga
from tpga.model import TransportProblem


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tpga", description="Transportation problem: GA with an exact LP baseline.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("solve", help="solve an instance given as JSON")
    s.add_argument("instance", help="JSON file with keys supply, demand, cost")
    s.add_argument("--method", choices=["lp", "ga", "both"], default="both")
    s.add_argument("--generations", type=int, default=200)
    s.add_argument("--population", type=int, default=60)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--plot", metavar="PNG", help="save the GA convergence plot")
    s.add_argument("--plan-plot", metavar="PNG", help="save a heat map of the best plan")
    s.add_argument("--json", action="store_true", help="print machine-readable output")

    g = sub.add_parser("generate", help="write a random balanced instance")
    g.add_argument("output")
    g.add_argument("-m", type=int, default=5)
    g.add_argument("-n", type=int, default=6)
    g.add_argument("--total", type=int, default=1000)
    g.add_argument("--seed", type=int, default=None)
    return ap


def _solve(args: argparse.Namespace) -> dict:
    p = TransportProblem.from_json(args.instance)
    out: dict = {"sources": p.shape[0], "destinations": p.shape[1], "balanced": p.is_balanced}
    x_lp, z_lp = solve_lp(p)
    out["lp"] = {"cost": z_lp}
    best_x = x_lp
    if args.method in ("ga", "both"):
        cfg = GAConfig(population=args.population, generations=args.generations, seed=args.seed)
        r = run_ga(p, cfg)
        gap = 0.0 if z_lp == 0 else 100.0 * (r.cost - z_lp) / abs(z_lp)
        out["ga"] = {"cost": r.cost, "gap_percent": gap, "evaluations": r.evaluations, "runtime_s": r.runtime_s}
        best_x = r.x
        if args.plot:
            from tpga.plot import plot_convergence, save

            save(plot_convergence(r, z_lp), args.plot)
    if args.method == "lp":
        out.pop("ga", None)
    if args.plan_plot:
        from tpga.plot import plot_plan, save

        save(plot_plan(best_x, "GA plan" if "ga" in out else "LP plan"), args.plan_plot)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            TransportProblem.random(args.m, args.n, total=args.total, seed=args.seed).to_json(args.output)
            print(f"wrote {args.output}")
            return 0
        out = _solve(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"instance: {out['sources']} sources x {out['destinations']} destinations, balanced={out['balanced']}")
        print(f"LP optimum: {out['lp']['cost']:.4f}")
        if "ga" in out:
            g = out["ga"]
            print(
                f"GA best:    {g['cost']:.4f}  (gap {g['gap_percent']:.2f}%, "
                f"{g['evaluations']} evaluations, {g['runtime_s']:.2f}s)"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
