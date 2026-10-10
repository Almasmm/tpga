# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/)
and the project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-10-10

### Added
- Web platform API (`platform/backend`, FastAPI): `GET /health`, `POST /solve` with LP and GA; GA responses include
  the LP optimum and the gap. Dockerfile and docker-compose.
- CI job **Platform API tests**; the package build now depends on it.

### Changed
- The repository is now the single code repository of the project and is named `tp-ga`.
- GitHub Actions updated by Dependabot: `actions/checkout` v7, `actions/setup-python` v7, `actions/upload-artifact` v6.

## [0.1.0] — 2026-10-09

### Added
- `TransportProblem`: classical model, validation, balancing with a dummy source or destination, JSON input/output.
- `decode_permutation`: greedy decoder from a permutation of the m·n arcs to a feasible plan.
- `run_ga`: permutation GA (order crossover, swap mutation, tournament selection, elitism) with a run record.
- `solve_lp`: exact optimum by linear programming (SciPy HiGHS) as the reference result.
- `tpga` command-line tool (`solve`, `generate`) and matplotlib plots of convergence and shipment plans.
- Test suite (pytest), lint and format (ruff), type checking (mypy), CI and release workflows.
