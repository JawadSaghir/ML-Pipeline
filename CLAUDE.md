# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An end-to-end ML pipeline (in early scaffolding) for **stock price prediction** using
historical AAPL data pulled from Yahoo Finance via `yfinance`. The intended pipeline
(per `projectflow.txt`) is: data ingestion → preprocessing → validation → training →
serving, with each component built and runnable individually under `src/`.

## Current state (scaffold vs. implemented)

Most of the tree is intentional placeholders. Do not assume code exists in empty dirs.

- **Implemented:** `src/data/ingestion.py` — downloads AAPL data and writes a CSV.
- **Empty stubs:** `src/data/preprocessing.py`, `src/data/validation.py`.
- **Empty scaffold dirs:** `api/`, `deployment/`, `monitoring/`, `models_artifacts/`,
  `test/`, `docs/`, `configs/` (only an empty `configs/data/`).
- `main.py` is a placeholder ("Hello from ml-project!").

When adding a pipeline stage, follow the existing `src/data/` layout (one module per
component) rather than inventing a new structure.

## Environment & commands

The project uses **`uv`** with a local `.venv` and targets Python 3.13+
(`.python-version` pins 3.14.2; `pyproject.toml` says `>=3.13`).

```bash
uv run python src/data/ingestion.py   # run a component directly
uv run python main.py                 # placeholder entrypoint
```

**Dependency caveat:** `pyproject.toml` currently declares `dependencies = []`, but the
`.venv` already has `yfinance`, `pandas`, `numpy`, and `dvc` installed ad hoc. When you
add an import, also add the package to `pyproject.toml` (`uv add <pkg>`) so the
environment is reproducible — don't rely on the pre-populated `.venv`.

There is no test runner or linter configured yet. If you add tests, use `pytest` under
`test/`.

## Data & DVC (important)

Data is versioned with **DVC, not git**. The `data/` directory is gitignored; its
real pointer is `data.dvc`.

- `data/AAPL_data.csv` is the tracked dataset; the DVC hash lives in `data.dvc`.
- Default remote is `data` → `../S3` (a local directory outside the repo). A second
  remote `myremote` → `../data_bucket` also exists. `autostage = true` is set.

```bash
uv run dvc pull        # fetch data tracked by data.dvc from the remote
uv run dvc add data    # re-track data/ after regenerating it
uv run dvc push        # upload to the configured remote
```

After changing the contents of `data/`, run `dvc add data` and commit the updated
`data.dvc` (the data files themselves must never be committed to git).

## Logging

Components log to the `Logs/` directory (e.g. `ingestion.py` writes
`Logs/data_ingestion.log`) via a file handler (DEBUG) plus a console handler (INFO).
Reuse this pattern for new components.

## Conventions / gotchas

- `ingestion.py:load_data` hardcodes an absolute Windows path default
  (`D:\machine_learning_course\ml_project\data`). Prefer passing an explicit relative
  path and parameterizing the ticker/date range rather than copying the hardcoded value.
- Several untracked artifacts (`S3/`, `nul`, `yfinance.ipynb` at repo root, `.claude-flow/`)
  are scratch/output — don't treat them as source.
