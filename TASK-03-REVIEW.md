# Review — Task 03: Add jupyter-related pre-commit hooks

- **Branch:** `task-03-jupyter-precommit` (tip `0eef04606`, base `dev`/`b4e4ca5b2`)
- **Reviewed:** 2026-08-31 · **Scope:** 3 commits, 4 files, +178/-33
- **Verdict:** APPROVE
  (Gates re-verified green, hook choices fact-checked and sound, deviations documented; remaining findings are minor wording/edge-case issues.)

## 1. Summary

The branch appends two pre-commit hooks for `.ipynb` files — `nbstripout` (rev `0.9.1`, latest tag, verified) and a small local `check_notebook_format` hook wrapping `nbformat.validate` — plus a one-time, hook-driven re-normalization of `preparingInsightData_example.ipynb` (cell ids UUID→`0..27`, `scrolled` metadata dropped, one non-newline-terminated source line rejoined). I re-ran both test gates: the quick suite reports exactly `174 passed, 2 xfailed, 1 xpassed` and `pre-commit run --all-files` exits 0 with 23/23 hooks `Passed`. I independently reproduced the committed notebook byte-for-byte from the base file via the hook entries and confirmed it is a fixed point of `ruff format`/`ruff check`/`nbstripout`, so the "hook-driven reformatting" claim holds. I also verified the author's fact-check of the (stale) task file's suggested recipes — all four of its hook references are indeed dead, and the chosen alternatives are the right ones. The most important issues, in one line each: (1) the "metadata is intentionally NOT stripped" / "no execution state ever lands in git" statements are factually wrong — nbstripout's default *does* strip 8 metadata keys, while top-level editor/run metadata (`vscode`, `papermill`, `orig_nbformat`) and non-default cell keys (`editable`) do land in git, so the DoD's "strip … editor metadata" is only partially met; (2) the local check passes notebooks with missing or duplicate cell ids (nbformat soft-warns, rc=0), which is weaker than the artifact's "rejected with a non-zero exit" claim; (3) validation-error messages can dump a whole cell into CI logs.

## 2. Findings

### 2.1 Critical

None found.

### 2.2 Major

None found.

### 2.3 Minor

- **m1** — **`.pre-commit-config.yaml:98-100`, `TASK-03-PR-PROPOSAL.md:20-21,54-55`** — The justification that "Metadata is intentionally NOT stripped" / "Modern nbstripout does not strip metadata by default anyway" is factually wrong, and the DoD clause "stripped of … editor metadata" is only partially enforced. nbstripout 0.9.1 strips 8 metadata keys by default (`metadata.signature`, `metadata.widgets`, `cell.metadata.{collapsed,ExecuteTime,execution,heading_collapsed,hidden,scrolled}` — see `_nbstripout.py:569-578` of the installed 0.9.1), and the committed diff itself proves it (the `scrolled: true` entry in `preparingInsightData_example.ipynb` was dropped by the default run). Conversely, top-level editor/run state is *kept*: I added `metadata.vscode`, `metadata.papermill`, top-level `orig_nbformat`, and `cell.metadata.editable` to a copy and ran `nbstripout` — all survived (top-level `vscode`/`papermill`/`orig_nbformat` retained, `editable` retained). So "no execution state ever lands in git" (PR proposal "What" section) is an overclaim.
  - *Evidence:* `nbstripout case_toplevel.ipynb` → `top-level keys: ['cells', 'metadata', 'nbformat', 'nbformat_minor', 'orig_nbformat', 'papermill']`; `cell1 metadata: {'editable': True}`; default extra-keys list read from the installed 0.9.1 source; `scrolled` removal visible in the branch's own notebook diff.
  - *Suggested fix:* (a) correct the wording in the config comment and the proposal (default behavior strips a fixed set of *cell* metadata keys; top-level metadata is untouched); (b) decide explicitly whether the DoD's "editor metadata" needs more: if yes, add `args: [--extra-keys, metadata.papermill, metadata.vscode, cell.metadata.editable]` (note: `--extra-keys` only supports `metadata.*` / `cell.metadata.*` paths, so top-level `orig_nbformat` cannot be handled by nbstripout at all — either accept it or drop it in the local hook). If no, record the decision explicitly against the DoD wording.
  - *Alternative:* keep the conservative default and document the residual keys in a comment; the two example notebooks carry none of them, so the practical risk today is low.

- **m2** — **`share/ci/check_notebook_format.py:19-22`** — Notebooks with a *missing* or *duplicated* cell id pass the hook (rc=0) with a raw warning on stderr; this is weaker than the artifact's claim "A structurally invalid notebook (bad cell id / non-JSON) is rejected with a non-zero exit". Pattern-invalid ids (`"a b"`, `""`, 10k chars) *are* rejected (rc=1), non-JSON is rejected (rc=1) — only the two soft-warning cases leak through. In the full hook chain this is mitigated because `nbstripout` (running first) rewrites all ids to sequential ones, so the gap only bites in standalone invocations (`pre-commit run check_notebook_format`), where the hook also prints the raw `warnings` line including the hook-env site-packages path.
  - *Evidence:* `python share/ci/check_notebook_format.py case_no_id.ipynb` → rc=0, stderr `.../nbformat/validator.py:434: MissingIDFieldWarning: Cell is missing an id field...`; `case_dup_ids.ipynb` → rc=0, `DuplicateCellId: Non-unique cell id '0' detected. Corrected to 'eb2c2e35'.`
  - *Suggested fix:* promote validation warnings to failures:
    ```python
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nbformat.validate(notebook)
    for w in caught:
        print(f"{filename}: {w.category.__name__}: {w.message}", file=sys.stderr)
        exit_code = 1
    ```

- **m3** — **`share/ci/check_notebook_format.py:21`** — Error reporting quality: for structural failures the nbformat message embeds the *entire offending cell* (a multi-KB single line containing the full cell source), and non-schema failures surface as bare Python exception text.
  - *Evidence:* `case_bad_celltype.ipynb` (cell_type `"banana"`) → message begins `{'cell_type': 'banana', 'id': '0', 'metadata': {}, 'source': 'This file is part of PIConGPU. \\...` (whole cell inlined); `case_bad_source.ipynb` (int in source list) → `sequence item 1: expected str instance, int found` (no context that it is a notebook parse failure).
  - *Suggested fix:* print the first line of `str(error)` as the headline and the remainder truncated (e.g. 200 chars) after `...`, or catch `nbformat.ValidationError` separately and emit `f"{filename}: invalid notebook: {error.message}"` (first line of jsonschema's message is already the human-readable part).

### 2.4 Nits

- **n1** — **`share/ci/check_notebook_format.py:19`** — `# noqa: BLE001` is inert: the repo has no ruff configuration (no `ruff.toml`, no `[tool.ruff]` in `lib/python/pyproject.toml`), so ruff's default rule selection (E4/E7/E9/F) does not include BLE001 and RUF100 (unused-noqa) is off. Remove the comment or keep the bare `except Exception` without it.
- **n2** — **`.pre-commit-config.yaml:110`** — Pin style `nbformat==5.11.1` vs the repo convention `pre-commit == v4.3.0` in `requirements_pre-commit.txt` (spaces around `==`). Purely cosmetic; `additional_dependencies` is the correct mechanism for a `language: python` local hook, and the exact pin is good for reproducibility.

## 3. Requirement traceability

| # | Requirement (from task file) | Status | Where / note |
|---|---|---|---|
| 1 | Add jupyter-related pre-commit hooks | met | `.pre-commit-config.yaml:93-111` (nbstripout + local check) |
| 2 | Strip outputs/execution counts on commit | met | nbstripout default behavior; verified on a dirty notebook (count→`null`, outputs→`[]`) |
| 3 | Strip editor metadata on commit | partial | Only nbstripout's 8 default keys; top-level `vscode`/`papermill`/`orig_nbformat`, cell `editable` survive (m1); decision documented but mis-worded |
| 4 | Validate against Jupyter Notebook schema / nbformat | met | local hook uses `nbformat.validate` against the declared version's schema (4.4 and 4.5 both validate); soft-warning caveats in m2 |
| 5 | Linting already covered by existing ruff hooks | met | verified: injected unused import into a code cell → `ruff check` reports F401 on the `.ipynb`; no nbqa added (correct, documented) |
| 6 | `pre-commit run --all-files` green on the two existing notebooks | met | re-run: rc=0, 23/23 `Passed`, worktree left clean |
| 7 | No manual cleanup beyond hooks; commit hook-driven reformatting | met | committed notebook reproduced byte-for-byte from base via `ruff format`+`nbstripout`; verified fixed point of all three hook entries; `createBunch_example.ipynb` correctly untouched |
| 8 | (Suggested) nbstripout at a current stable rev, verify hook ids | met | `0.9.1` is the latest tag of `kynan/nbstripout` (verified via `git ls-remote`); id `nbstripout` correct |
| 9 | (Suggested) nbformat validation via check-jsonschema or a nbformat hook | met (documented deviation) | suggested repos/options verified stale (see §4); local `nbformat.validate` hook is the right substitute, follows the `share/ci/check_cpp_code_style` pattern |
| 10 | (Suggested) sensible hook ordering | met | ruff → … → nbstripout → check (strip before validate); chain behaves correctly |
| 11 | Verification: dirty notebook → commit blocked/autofixed | met | dirty copy auto-stripped by nbstripout (pre-commit would report "files were modified" → blocks commit); check passes on the cleaned result |
| 12 | Verification: notebooks remain executable examples, cells pass ruff | met | ruff + ruff-format `Passed` on both notebooks |
| 13 | Note nbval/CI-execution as out-of-scope follow-up | met | `TASK-03-PR-PROPOSAL.md:98-101` |

## 4. Claim verification (author artifact)

| Claim (from TASK-03-PR-PROPOSAL.md / report) | Re-verified? | Result / delta |
|---|---|---|
| Quick suite `174 passed, 2 xfailed, 1 xpassed` | yes | exact match: `174 passed, 2 xfailed, 1 xpassed, 3499 subtests passed` (rc=0) — equals baseline |
| `pre-commit run --all-files` exits 0, every hook `Passed` (23/23) | yes | rc=0, 23 hooks `Passed` (check_cpp_code_style excluded: `stages: [manual, pre-push]`); worktree left clean |
| `nbdev/nbstripout` 404s, real repo is `kynan/nbstripout` | yes | `git ls-remote` fails for `nbdev/nbstripout`, succeeds for `kynan/nbstripout` (tag `0.9.1` present and is the newest tag) |
| Hook id is `nbstripout`; `--strip-execution` no longer exists, count-stripping is default | yes | `nbstripout --help` (0.9.1): no such flag; `--keep-count` disables the default |
| "Modern nbstripout does not strip metadata by default (the old `--strip-metadata` flag is gone)" | yes | **inaccurate** — 8 metadata keys are stripped by default (see m1); the config-comment rationale is wrong even though the net decision is defensible |
| `check-jsonschema` store option gone; JSON Schema Store no longer hosts a `jupyter-notebook` schema | yes | schemastore catalog: 0 entries matching `jupyter`; check-jsonschema hooks are now autogenerated per-schema (`--builtin-schema vendor.…`) |
| `pre-commit-ci/hooks` repo does not exist | yes | `git ls-remote` → 404 (auth prompt) |
| `nbformat` CLI removed from the package | yes | no `nbformat` entry point installed by nbformat 5.11.1 |
| Dirty notebook: nbstripout auto-strips; check accepts cleaned result | yes | execution_count→`null`, outputs→`[]`, `scrolled` dropped; check rc=0 on cleaned file |
| "A structurally invalid notebook (bad cell id / non-JSON) is rejected with a non-zero exit" | yes | true for pattern-invalid ids (`rc=1`, e.g. `'a b' does not match '^[a-zA-Z0-9-_]+$'`) and non-JSON (`rc=1`); **not true** for missing/duplicate ids (rc=0, stderr warning only) — see m2 |
| Reformatting was hook-driven; "rendered output is unchanged" for the rejoined source line | yes | base → `ruff format` + `nbstripout` reproduces the committed file; nbstripout (not ruff) rejoins the non-newline-terminated line; joined string identical to how the reader always parsed it |
| Cell ids → sequential `0..27` is nbstripout's intended behavior | yes | `--keep-id` help text confirms default id-renormalization |
| `createBunch_example.ipynb` (4.4) already clean, untouched | yes | `nbstripout` on base copy: no change; not in the diff |
| Merge-order claim (append-only change, ruff entries byte-for-byte unchanged) | yes | diff hunk `@@ -90,3 +90,22 @@` is append-only |

## 5. Design discussion

**Local `nbformat.validate` hook vs. the task's suggested third-party hooks.** The task file's two suggested validation recipes were stale, and the author's fact-check is accurate (all verified above). Given that, the local hook was the right call: it is ~27 lines, follows the repo's existing `share/ci/check_*` local-hook pattern, pins `nbformat==5.11.1` for reproducibility, and — importantly — validates against the schema of each notebook's *declared* version, which matters here because the repo contains both a 4.4 and a 4.5 notebook. A fixed local JSON schema file (check-jsonschema style) could not do that. Residual trade-offs a maintainer should weigh: (a) the pin means a future nbformat-5 notebook is hard-rejected with `Unsupported nbformat version 5` until the pin is bumped — fail-closed, which is the right default for a format gate; (b) conversely, unknown *future* 4.x minors (e.g. 4.6) are accepted leniently — `nbformat.validate(nb, strict=True)` would reject those instead, at the cost of false failures whenever Jupyter ships a new minor before the pin is updated; the lenient default is the pragmatic choice. (c) The hook environment is built per-machine/CI (nbformat + jupyter_core/traitlets/jsonschema/fastjsonschema) — a bit heavier than the "lighter" check-jsonschema the task anticipated, but a one-time cost and negligible at runtime (500 valid notebooks validate in ~0.5 s in my probe).

**nbstripout configuration.** Default args (no `args:`) is the conservative, correct choice; stripping outputs/counts is the tool's primary contract and the repo currently has zero notebooks with outputs. The one judgment call is metadata: the default set covers the noisy JupyterLab keys (`collapsed`, `execution`, `scrolled`, …) but leaves `papermill`, `vscode`, `editable`, `orig_nbformat` in git. For a repo where notebooks are hand-maintained examples this is fine; if papermill-based notebook CI ever lands (the suggested nbval follow-up), `--extra-keys metadata.papermill` should be added then. Top-level `orig_nbformat` is not reachable by nbstripout at all (`--extra-keys` only supports `metadata.*`/`cell.metadata.*` paths), so only a mutator (the local hook, renamed/extended) could strip it — not worth it today.

**Interaction with existing hooks.** Ordering (ruff → nbstripout → check) is sound: linting code cells is unaffected by outputs, and validation runs on stripped state. `ruff format` and nbstripout both serialize notebook JSON with 1-space indent, so there is no reformat ping-pong (I verified the committed file is a joint fixed point). `pretty-format-json`, `trailing-whitespace`, `require-ascii` etc. run on `.ipynb` harmlessly; none of the new hooks affect non-jupyter files (`types: [jupyter]`).

**CI impact.** `.gitlab-ci.yml` runs `pre-commit run --all-files --hook-stage manual`; in pre-commit 4.3.0 the root default is *all* stages, so the new hooks run in CI (as do all other hooks), plus their env is pip-built there. No breakage for non-jupyter workflows: both hooks are `types: [jupyter]`-gated and the config change is append-only.

## 6. Prioritized next steps

1. Fix the misleading metadata wording in `.pre-commit-config.yaml:96-100` and the PR proposal ("does not strip metadata by default" → describe the actual 8-key default; "no execution state ever lands in git" → qualify it), and explicitly record the decision on the DoD's "editor metadata" clause (m1).
2. Make the local hook fail on `MissingIDFieldWarning`/`DuplicateCellId` (capture warnings, rc=1) and format error messages so a bad cell doesn't dump its whole source into CI logs (m2, m3).
3. Remove the inert `# noqa: BLE001`; optionally align the pin style (`nbformat == 5.11.1`) (n1, n2).
4. (Optional, content) the reformat exposed a pre-existing defect in the same markdown cell — "…in case of complex input.The 3D field data…" (missing newline, words glued) plus "The the" — consider a one-line content fix now that the file is being touched (FYI below).

## FYI (inherited from base, not scored here)

- `preparingInsightData_example.ipynb` (base) had a markdown source line missing its trailing newline, so the rendered text reads "…in case of complex input.The 3D field data…" (glued sentences); the same paragraph also has "The the" and "ready te be used". The hook-driven reformat preserved the rendered text exactly (verified) but made the defect a single source line — an easy moment to fix the content.
- The task file's "Suggested approach" section contained four stale hook references (wrong nbstripout repo, wrong id/flags, dead check-jsonschema store option, nonexistent `pre-commit-ci/hooks` repo). The author correctly detected, verified, and documented all of them — worth a line in the PR description so reviewers don't re-derive it.
- Baseline quick-suite numbers on `dev` (174 passed / 2 xfailed / 1 xpassed) match the branch exactly — no test regression.
