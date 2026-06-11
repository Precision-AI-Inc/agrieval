# Contributing

Thank you for your interest in contributing to the Precision AI Agricultural Embedding API.

## Setup

Requires Python 3.10+.

```bash
git clone <repo-url>
cd pai-ag-emb

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
pre-commit install
```

`pip install -e ".[dev]"` installs all dependencies including test, analysis, and pre-commit tooling. `pre-commit install` registers the hooks so they run automatically on every commit.

## Making changes

1. Create a branch from `main`.
2. Make your changes.
3. Add or update tests — coverage must remain at or above 90%.
4. Commit. Pre-commit hooks run automatically and will block the commit if any check fails.

If a hook auto-fixes files (ruff lint/format), stage the changes and commit again.

## Pre-commit hooks

| Hook | What it checks |
|---|---|
| File hygiene | Large files (>1 MB), trailing whitespace, merge conflicts, private keys, debug statements |
| `ruff` | Linting and import sorting (auto-fix) |
| `ruff-format` | Code formatting (auto-fix) |
| `mypy` | Static type checking |
| `pytest` | Full test suite with ≥ 90% coverage |

To run all hooks manually without committing:

```bash
pre-commit run --all-files
```

## Code style

- **Formatter / linter:** ruff (`line-length = 88`, Python 3.10 target).
- **Type hints:** all public functions should have type annotations.
- **Docstrings:** NumPy style for all public functions, classes, and modules.
- **Comments:** only where the _why_ is non-obvious. No inline narration of what the code does.

## Tests

```bash
pytest                          # run all tests with coverage report
pytest tests/test_metrics.py    # run a specific file
pytest -k "test_purity"         # run tests matching a pattern
```

Tests live in `tests/` and mirror the package structure. The coverage threshold (90%) is enforced both by `pytest` directly and by the pre-commit hook.

## Pull requests

- Keep PRs focused — one logical change per PR.
- Write a clear description of what changed and why.
- All pre-commit hooks must pass before requesting review.
