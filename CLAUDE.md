# Vex — Development Rules

## Git Workflow

- **ALWAYS create a branch** before making changes. Never commit directly to main.
- **ALWAYS create a PR** for code changes. The only exception is this file.
- **NEVER push to a merged/closed PR branch.** Check PR status before pushing.
- **NEVER hack around a bug** — fix it properly, commit, push, rebuild.
- **Clean up local branches** after PRs are merged. Keep the repo tidy.

## Code Quality

- **Run ruff check + ruff format** before every commit.
- **Test locally against real data** when possible, not just unit tests.
- **Don't leave dead code** — unused functions, duplicate definitions, stale constants.
- **Wire config properly** — don't hardcode values that exist in pydantic settings.

## Docker / Deployment

- **Test that the container actually builds** before raising deployment PRs.
- **Verify health checks pass** after compose up.
- **Don't hack around Docker issues** with inline python -c scripts. Fix the actual code.

## Pipeline

- Nix flake for local dev, Docker for deployment.
- `ruff` for linting and formatting (line-length 120, Python 3.12).
- `pytest` with testcontainers for DB tests, `--run-golden` for golden dataset tests.
