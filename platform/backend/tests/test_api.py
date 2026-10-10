from app.main import app
from fastapi.testclient import TestClient

c = TestClient(app)
INST = {"supply": [20, 30, 20], "demand": [25, 25, 20], "cost": [[8, 6, 3], [5, 4, 9], [2, 7, 6]]}


def test_health():
    assert c.get("/health").json() == {"status": "ok"}


def test_lp_known_optimum():
    r = c.post("/solve", json={**INST, "method": "lp"}).json()
    assert abs(r["cost"] - 225) < 1e-6 and r["balanced_with_dummy"] is False


def test_ga_reports_gap_against_lp():
    r = c.post("/solve", json={**INST, "method": "ga", "generations": 30, "population": 20, "seed": 0}).json()
    assert r["cost"] >= r["lp_optimum"] - 1e-6 and r["gap_percent"] >= -1e-9


def test_unbalanced_gets_dummy():
    r = c.post("/solve", json={"supply": [10, 5], "demand": [8, 4], "cost": [[1, 2], [3, 4]]}).json()
    assert r["balanced_with_dummy"] is True and len(r["plan"][0]) == 3


def test_bad_shape_rejected():
    assert c.post("/solve", json={"supply": [1], "demand": [1], "cost": [[1, 2]]}).status_code == 422


def test_too_large_instance_rejected():
    big = {"supply": [1] * 21, "demand": [1] * 21, "cost": [[1] * 21 for _ in range(21)]}
    assert c.post("/solve", json=big).status_code == 413
