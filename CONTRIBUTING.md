# Contributing

This is a private research repository. Keep changes focused, tested, and easy to
review.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Checks

```bash
python -m pytest
python -m ruff check .
```

Generated experiment outputs belong under `results/` and are ignored by git.
Commit only small, intentional fixtures that are required by tests or examples.
