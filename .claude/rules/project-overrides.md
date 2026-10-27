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
  without an API. `ecc/python/fastapi.md` matches `app/**/*.py` by path only: it does **not** apply here.

## Code review and test workflow (ECC command, agents, skills)
- `/python-review`, the `python-reviewer` and `tdd-guide` agents and the `tdd-workflow` / `python-testing` skills are
  used **when the user asks** for a review or a test-first change — not launched proactively on every edit.
- Tools available here: `ruff` and `pytest` only. Do not run or install mypy, pylint, black, isort, bandit or
  pytest-cov; when a review step needs one of them, say it is not set up and move on. Never reformat with black.
- Reviews are local and read-only: report findings, change code only when the user asks. Never approve, comment on
  or request changes on GitHub pull requests (`gh pr review`, `gh api … comments`).
- Coverage percentages are not measured in this project; the 80% targets in the ECC files do not apply.
- A TDD evidence report is written only if the user asks, in `.claude/tdd/`.
- `ecc/python/hooks.md`: do not configure hooks (no `~/.claude/settings.json` changes).

## Workflow
- Installed from ECC: `/python-review`, agents `python-reviewer` and `tdd-guide`, skills `tdd-workflow` and
  `python-testing`. Any other ECC reference (`ecc:…` agents, `python-patterns`, `/review-pr`, `/orch-review`,
  `/code-review`, …) points to a component that is **not installed**: ignore it, do not install it.
- Commit only when the user asks; message in the imperative ("Add …", "Fix …") with a short body; the user pushes.
- No hooks, no changes to `~/.claude`, no global installs.
