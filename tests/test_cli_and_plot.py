import json

import numpy as np

from tpga import GAConfig, run_ga
from tpga.cli import main
from tpga.plot import plot_convergence, plot_plan, save


def test_cli_solve_both_json(tmp_path, capsys):
    inst = tmp_path / "i.json"
    inst.write_text(
        json.dumps({"supply": [20, 30, 20], "demand": [25, 25, 20], "cost": [[8, 6, 3], [5, 4, 9], [2, 7, 6]]})
    )
    conv, plan = tmp_path / "c.png", tmp_path / "p.png"
    rc = main(
        [
            "solve",
            str(inst),
            "--generations",
            "20",
            "--seed",
            "1",
            "--json",
            "--plot",
            str(conv),
            "--plan-plot",
            str(plan),
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["lp"]["cost"] == 225 and out["ga"]["gap_percent"] >= 0
    assert conv.stat().st_size > 0 and plan.stat().st_size > 0


def test_cli_solve_lp_text(tmp_path, capsys):
    inst = tmp_path / "i.json"
    inst.write_text(json.dumps({"supply": [10, 5], "demand": [8, 4], "cost": [[1, 2], [3, 4]]}))
    plan = tmp_path / "p.png"
    assert main(["solve", str(inst), "--method", "lp", "--plan-plot", str(plan)]) == 0
    out = capsys.readouterr().out
    assert "LP optimum: 20.0000" in out and "GA best" not in out and plan.exists()


def test_cli_ga_text_output(tmp_path, capsys):
    inst = tmp_path / "i.json"
    inst.write_text(json.dumps({"supply": [5], "demand": [5], "cost": [[0]]}))
    assert main(["solve", str(inst), "--method", "ga", "--generations", "2", "--population", "4"]) == 0
    assert "gap 0.00%" in capsys.readouterr().out


def test_cli_generate_and_errors(tmp_path, capsys):
    out = tmp_path / "g.json"
    assert main(["generate", str(out), "-m", "3", "-n", "4", "--seed", "1"]) == 0
    assert set(json.loads(out.read_text())) == {"supply", "demand", "cost"}
    assert main(["solve", str(tmp_path / "missing.json")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert main(["solve", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err


def test_plots_render(tmp_path):
    from tpga import TransportProblem

    p = TransportProblem.random(3, 3, seed=0)
    r = run_ga(p, GAConfig(population=10, generations=5, seed=0))
    save(plot_convergence(r), tmp_path / "a.png")
    save(plot_plan(np.array([[1.0, 0.0], [0.0, 5.0]])), tmp_path / "b.png")
    assert (tmp_path / "a.png").exists() and (tmp_path / "b.png").exists()
