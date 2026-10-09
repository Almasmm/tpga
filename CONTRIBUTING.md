# Contributing

1. Open an issue describing the change.
2. Create a branch from `main`: `feature/<short-name>` or `fix/<short-name>`.
3. Install the development tools: `pip install -e .[dev]`.
4. Before pushing, run the same checks as CI:
   ```bash
   ruff check . && ruff format --check . && mypy && pytest --cov=tpga
   ```
5. Open a pull request that references the issue (`Closes #N`). `main` is protected: the CI checks must pass before merging.

Commit messages use the imperative mood ("Add capacity constraints", not "Added …").
