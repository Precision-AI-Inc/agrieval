# Precision AI — Python Project Standards

Apply these standards when writing, reviewing, or refactoring code in any PAI Python project.

---

## Environment

- Python 3.10+, managed with `python -m venv .venv` (never conda)
- Install: `pip install -e ".[dev]"` → installs all deps including pre-commit
- Register hooks once: `pre-commit install`

---

## Project layout

```
pai/<namespace>/
  api/
    routes/      # FastAPI route handlers (thin — delegate to services)
    config.py    # env-var config only
    app.py       # FastAPI app factory
  schemas/       # Pydantic v2 request/response models
  services/      # Business logic (evaluate.py, reporting.py, …)
  metrics/       # Pure computation modules
    __init__.py  # re-exports only — no logic
docs/            # Sphinx (HTML + LaTeX/PDF)
tests/           # mirrors package structure
examples/        # standalone runnable scripts
```

---

## pyproject.toml — canonical config

```toml
[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E","W","F","I","UP","B","SIM","N","C90","D","PT","RUF","PL"]
ignore = [
    "E501",    # enforced by ruff-format
    "B008",    # FastAPI default-arg pattern
    "SIM108",  # ternary readability
    "D100","D104",          # module/package docstrings optional
    "D203","D213",          # pydocstyle conflicts — always ignore these two
    "PLR0913","PLR2004",    # arg count + magic values common in metrics/tests
]
per-file-ignores = {
    "**/__init__.py" = ["F401"],
    "tests/**"       = ["D","PLR"],
    "docs/conf.py"   = ["E402","UP031"],
}

[tool.ruff.lint.pydocstyle]
convention = "numpy"          # enforces NumPy docstring style

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.isort]
known-first-party = ["pai"]   # adjust namespace per project

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true  # formats code blocks inside docstrings

[tool.pytest.ini_options]
addopts = "--cov=pai --cov-report=term-missing --cov-fail-under=90"

[tool.coverage.report]
fail_under = 90
exclude_lines = ["pragma: no cover","if __name__ == .__main__.:",
                 "raise ImportError","except ImportError"]
```

---

## pre-commit

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks: [check-added-large-files (--maxkb=1000), check-yaml, check-json,
            check-toml, end-of-file-fixer, trailing-whitespace,
            check-merge-conflict, detect-private-key, debug-statements]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.5
    hooks: [ruff (--fix), ruff-format]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports, --no-strict-optional]

  - repo: local
    hooks:
      - id: pytest
        entry: python -m pytest
        language: system
        args: [tests/, -q, --tb=short, --no-header]
        always_run: true
```

---

## Code style

### Imports
- All imports at the **top of the file** — never inside functions
- Optional/unavailable deps: module-level `try/except ImportError` with a flag variable
- No silent fallbacks unless mathematically equivalent (document why)
- `__init__.py` re-exports only — no logic, no comments between import blocks
- `__all__` must be **alphabetically sorted**

```python
# optional dep pattern
try:
    import plotly.graph_objects as go  # type: ignore[import]
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# inside function that needs it
if not _PLOTLY_AVAILABLE:
    raise ImportError("plotly is required: pip install plotly") from None
```

### Docstrings
- **NumPy style** on all public functions, classes, and modules
- Module docstrings: plain English, no prefixes (no P0/P1/P2, no "original", no labels)
- One-line summary in **imperative mood** ("Compute …", "Return …", "Save …")
- Blank line between summary and extended description (D205)

### Type hints
- Required on all public function signatures
- Use `X | Y` union syntax (Python 3.10+), not `Union[X, Y]` or `(X, Y)` in isinstance
- Use `Any` from `typing` for genuinely dynamic types — don't use `object` when methods will be called on it

### Comments
- Only when the **why** is non-obvious
- No section-divider comments that describe what the code already says
- No cryptic labels, no TODO/FIXME without a ticket reference

### Exception handling
- Always `raise X from err` or `raise X from None` inside `except` blocks (B904)

### General
- `len()` returns `int` — never `int(len(...))`
- Loop variables not used in the body → rename to `_`
- `assert a and b` in tests → split into separate asserts
- `@pytest.fixture` not `@pytest.fixture()`
- `pytest.raises` always includes `match=` parameter
- No `# type: ignore[attr-defined]` when `Any` already covers the attribute

---

## Docs (Sphinx)

```
docs/
  conf.py          # version from git tag, logo, LaTeX/fancyhdr, enumitem fix
  index.rst        # toctree: readme, modules
  modules.rst      # autodoc for all public submodules
  requirements.txt # sphinx, sphinx-rtd-theme, myst-parser, sphinx-autodoc-typehints
  Makefile         # make html | make latexpdf | make clean
  assets/logo.png
```

- `conf.py` copies root `README.md` → `docs/readme.md` at build time
- Version read from `git describe --tags --exact-match`, falls back to `0.0.0`

---

## Testing

- Mirror package structure: `tests/test_<module>.py`
- Shared fixtures in `tests/conftest.py`
- 90% coverage hard minimum — enforced by pytest and pre-commit
- No mocks for database/filesystem unless truly unavoidable
- Integration tests marked `@pytest.mark.integration` and excluded from default runs

---

## What to avoid

| Pattern | Instead |
|---|---|
| `int(len(x))` | `len(x)` |
| `isinstance(x, (A, B))` | `isinstance(x, A \| B)` |
| `assert a and b` (tests) | two separate asserts |
| `@pytest.fixture()` | `@pytest.fixture` |
| `pytest.raises(ValueError)` | `pytest.raises(ValueError, match="…")` |
| `raise X` inside except | `raise X from None` or `raise X from err` |
| Deferred imports inside functions | Module-level try/except with flag |
| Cryptic prefixes (P0–P4, "original") | Plain descriptive names |
| `# type: ignore[attr-defined]` on `Any` | Remove — redundant |
| `# noqa` suppression | Fix the underlying issue |
| Silent fallback for optional dep | Raise `ImportError` with install hint |
