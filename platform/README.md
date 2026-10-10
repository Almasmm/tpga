# Web platform

Backend: FastAPI over the `tpga` library.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness check |
| `POST /solve` | solve an instance with `lp` or `ga`; a GA response always includes the LP optimum and the percentage gap |

Runs are synchronous and limited to 400 arcs until the job queue exists (see the *Platform MVP* milestone).

```bash
pip install -e . -r platform/backend/requirements.txt   # from the repository root
cd platform/backend && uvicorn app.main:app --reload      # http://127.0.0.1:8000/docs
pytest platform/backend/tests                             # from the root, with PYTHONPATH=platform/backend
docker compose -f platform/docker-compose.yml up --build
```

The frontend is not started yet.
