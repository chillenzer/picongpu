# PR Proposal - task 03: Add jupyter-related pre-commit hooks

## Proposed title

`Add jupyter-related pre-commit hooks (nbstripout + nbformat validation)`

## Merge order

This PR touches `.pre-commit-config.yaml` and **should be merged before
`task-08-ruff-all`**. Both modify the same file; this PR only *appends* new
hook entries (ruff entries are left byte-for-byte unchanged) so the merge is
expected to be clean, but landing this first keeps task 08's diff minimal.

## What

Two new pre-commit hooks for Jupyter notebooks (`.ipynb`), appended to the end
of `.pre-commit-config.yaml`:

1. **`nbstripout`** (repo `kynan/nbstripout`, rev `0.9.1`) - strips outputs,
   execution counts and normalizes cell ids on commit, so no execution state
   ever lands in git.
2. **`check_notebook_format`** (local hook, new script
   `share/ci/check_notebook_format.py`, pinned `nbformat==5.11.1` in
   `additional_dependencies`) - validates every notebook against the nbformat
   schema of the version it declares.

Plus the one-time, hook-driven reformatting of
`lib/python/picongpu/extra/input/preparingInsightData_example.ipynb` that the
hooks produce (see below).

## Why

The two example notebooks under `lib/python/picongpu/extra/input/` were not
protected by any pre-commit hook. Without these hooks, a contributor who
executes a notebook and commits it would push outputs, execution counts and
regenerated cell ids, producing noisy diffs. Linting of notebook code cells
was already covered by the existing ruff hooks (`types_or: [python, pyi,
jupyter]`), so no nbqa hook is added.

### Hook choices - deviating from the task file's suggested recipes

The task file suggested `nbstripout` with `id: strip-notebook` +
`--strip-execution` and `check-jsonschema` with `jsonschema_store:
jupyter-notebook` (or `pre-commit-ci/hooks` `nbformat`). All of those were
verified against the current hook repos and are **no longer valid**:

- The nbstripout hook repo is `kynan/nbstripout` (the `nbdev/nbstripout`
  reference 404s). The current hook id is **`nbstripout`**, and the
  `--strip-execution` flag no longer exists - stripping execution counts is
  the *default* behavior (disable via `--keep-count`). Hence no `args` are
  passed.
- **Metadata is intentionally not stripped.** The example notebooks only
  carry the standard `kernelspec` and `language_info` metadata, which tools
  need to open the notebooks with the right kernel. Modern nbstripout does not
  strip metadata by default anyway (the old `--strip-metadata` flag is gone).
- `check-jsonschema` no longer supports a `jsonschema_store` option, and the
  JSON Schema Store no longer hosts a `jupyter-notebook` schema. The
  `pre-commit-ci/hooks` repo referenced for an `nbformat` hook does not exist.
  The `nbformat` CLI (`nbformat --validate`) was also removed from the
  `nbformat` package.
- Therefore validation is implemented as a small local hook that calls the
  still-supported `nbformat.validate` Python API. This follows the repo's
  existing pattern for local check hooks (`share/ci/check_cpp_code_style.sh`)
  and pins `nbformat==5.11.1` for reproducibility. It validates each notebook
  against the schema of its *declared* `nbformat`/`nbformat_minor` (the repo
  contains both a 4.4 and a 4.5 notebook), which a single fixed schema file
  could not do.

## Verification

- `pre-commit run --all-files` exits 0 with every hook `Passed`, including
  the new `nbstripout` and `check_notebook_format`.
- Dirty-notebook check: adding an output + execution count + `scrolled`
  metadata to a notebook and running the hooks on it - `nbstripout`
  auto-strips it (hook reports "files were modified", i.e. the commit would be
  blocked/autofixed) and `check_notebook_format` accepts the cleaned result.
  A structurally invalid notebook (bad cell id / non-JSON) is rejected with a
  non-zero exit.
- Notebooks remain valid: `nbformat.validate` passes on both; code cells still
  pass the existing ruff lint + format hooks.
- Test gate unchanged: `lib/python/test/picongpu` quick suite reports
  `174 passed, 2 xfailed, 1 xpassed`.

## Hook-driven notebook reformatting (committed)

Running the hooks normalized `preparingInsightData_example.ipynb`:

- cell ids changed from UUIDs to stable sequential ids (`0`..`27`) - this is
  nbstripout's intended behavior to keep ids stable across executions;
- a transient `scrolled: true` cell-metadata entry was dropped;
- one source line that violated the nbformat spec (a non-final line without a
  trailing newline) was rejoined with the following line - exactly how
  `nbformat.read` (and Jupyter itself) already interprets the file, so the
  rendered output is unchanged.

`createBunch_example.ipynb` (nbformat 4.4) was already clean and is untouched.

## Possible follow-up (out of scope)

Executing the example notebooks in CI (e.g. via `nbval`) would guarantee the
examples stay runnable, but requires Jupyter kernels and extra dependencies
and was not requested by the task.
