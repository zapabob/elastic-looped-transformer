# 2026-05-27 Dependabot lock refresh - Codex

## Goal

Resolve open Dependabot alerts in the Python lockfile without changing ELT training or evaluation behavior.

## Files touched

- `uv.lock`
- `_docs/2026-05-27-dependabot-lock-refresh-codex.md`

## Key decisions

- Refreshed only the alerted packages: `GitPython`, `idna`, and `urllib3`.
- Kept source code and experiment configs unchanged.

## Verification

- `uv lock --check`: passed.
- `uv lock --dry-run`: no lockfile changes detected.
- Python AST parse of repository Python files: passed.
- `uv tree --package GitPython/idna/urllib3 --invert`: resolved `GitPython v3.1.50`, `idna v3.16`, and `urllib3 v2.7.0`.
- Full pytest exceeded the local timeout and is recorded as inconclusive for this lock-only change.
