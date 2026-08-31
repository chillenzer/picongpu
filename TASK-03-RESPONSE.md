# Response to review - task 03 (jupyter pre-commit hooks)

Rework on `task-03-jupyter-precommit` on top of review tip `5bf12792a`.
All findings were accepted and fixed; none rejected.

## Findings

| ID | Disposition + evidence | Commit(s) |
|----|------------------------|-----------|
| m1 | fixed - corrected the "metadata is intentionally NOT stripped" claim in `.pre-commit-config.yaml` and the PR proposal: nbstripout 0.9.1 strips 8 metadata keys by default (`metadata.signature`, `metadata.widgets`; cell keys `collapsed`, `ExecuteTime`, `execution`, `heading_collapsed`, `hidden`, `scrolled`). Verified against the installed 0.9.1 source (`extra_keys` list) and by behaviour: a default run kept `orig_nbformat`/`papermill`/`metadata.vscode`/`cell.metadata.editable` on a copy of the example notebook, matching the review's evidence. Decision to keep the default key set recorded in the comment/proposal (the committed notebooks carry none of the residual keys; revisit with `--extra-keys` if papermill/nbval notebook CI lands; top-level `orig_nbformat` is rejected by `check_notebook_format`, verified rc=1 "Additional properties are not allowed"). | `60313692c` |
| m2 | fixed - the hook captures `MissingIDFieldWarning`/`DuplicateCellId` around both `nbformat.read()` and `nbformat.validate()` and exits 1. Verified: notebook with a missing cell id -> rc=1 `MissingIDFieldWarning`, duplicate ids -> rc=1 `DuplicateCellId`, no raw warning lines with site-packages paths; negative self-test added covering both cases. | `5a8098a97`, `a41c0e8a1`, `7ceb84afe` |
| m3 | fixed - error messages shortened to the first line truncated to 200 chars; schema failures ("invalid notebook") distinguished from unreadable files ("not a valid notebook"). Verified: 6 KB invalid cell -> single stderr line of 203 chars; non-JSON -> `not a valid notebook: Notebook does not appear to be JSON`. | `d97edbfa5` |
| n1 | fixed - removed the inert `# noqa: BLE001` comments. | `d97edbfa5` |
| n2 | fixed - pin style aligned with the repo convention (spaces around `==`, cf. `pre-commit == v4.3.0` in `requirements_pre-commit.txt`): `nbformat == 5.11.1`. | `60313692c` |

Additional (not a review finding): task artifact documents (`^TASK-.*\.md$` at
the repo root) are excluded from the `require-ascii` hook so the committed,
non-ASCII `TASK-03-REVIEW.md` passes unchanged; same pattern as the sibling
task branches (`9443ec892`).

## Gates (final tip)

- `python -m pytest quick/ -q` in `lib/python/test/picongpu`:
  `174 passed, 2 xfailed, 1 xpassed, 3499 subtests passed` (rc=0)
- `pre-commit run --all-files`: 23/23 hooks `Passed` (rc=0), worktree clean;
  both example notebooks hook-clean
- `share/ci/check_notebook_format_selftest.py`: 8/8 PASS (valid 4.5/4.4,
  missing id, duplicate ids, bad cell type, bad id pattern, large invalid
  cell, non-JSON) - not part of the pinned pytest quick/ gate
