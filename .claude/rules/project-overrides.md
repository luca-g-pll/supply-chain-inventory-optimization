# Project overrides for the ECC rules

The rules in `.claude/rules/ecc/` are general guidance. In this project, **this file and `CLAUDE.md` win** whenever
they disagree.

## Formatting and style
- **ruff only** (`make lint`), line length 120, rules in `pyproject.toml`. Do not run or add black or isort.
- Type annotations on public functions of `src/`; Streamlit page scripts in `app/views/` do not need them.
- Immutability: dataclasses for configuration stay `frozen=True`. The numerical engines (`src/simulation.py`,
  `src/constraints.py`, `src/forecast.py`, `src/features.py`) update NumPy arrays in place for speed — keep them
  vectorised, do not rewrite them in a copy-on-write style.
- Function and file length limits are guidance: pipeline stages and page scripts can be longer when splitting them
  would only move code around.
- No hard-coded business values: they belong in `config/params.yaml`.

## Tests
- pytest + hypothesis, synthetic data only (`make test`). New behaviour comes with tests in `tests/`.
- No coverage threshold and no `pytest-cov` dependency; do not add unregistered `pytest.mark` markers.
- Writing the test first is welcome, not mandatory.

## Security
- The project needs no secrets; never commit any. Do not add `python-dotenv` or `bandit` unless the user asks.
- Web-API items (CSRF, rate limiting, auth, response envelopes) do not apply: the app is a Streamlit dashboard
  without an API.

## Workflow
- References to ECC agents, commands or skills (`ecc:…`, `python-patterns`, `python-testing`, `tdd-guide`, …) point
  to components that are **not installed**: ignore them, do not install them.
- Commit only when the user asks; message in the imperative ("Add …", "Fix …") with a short body; the user pushes.
- No hooks, no changes to `~/.claude`, no global installs.
