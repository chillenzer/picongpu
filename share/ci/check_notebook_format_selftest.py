"""Self-test for the check_notebook_format pre-commit hook.

Runs the actual hook script (share/ci/check_notebook_format.py) on
deliberately malformed notebooks and checks the expected exit code and
output for each case. This is the negative test for the hook; it is
not part of the pytest suite.

Usage (from the repo root, with an interpreter that has nbformat
installed, e.g. the pre-commit hook environment of
check_notebook_format):

    python share/ci/check_notebook_format_selftest.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HOOK = Path(__file__).resolve().with_name("check_notebook_format.py")

VALID_4_5 = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["print('hi')\n"],
            "id": "0",
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Title\n"],
            "id": "1",
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def copy():
    return json.loads(json.dumps(VALID_4_5))


def make_valid_4_4():
    nb = copy()
    nb["nbformat_minor"] = 4
    for cell in nb["cells"]:
        del cell["id"]
    return nb


def make_missing_id():
    nb = copy()
    del nb["cells"][0]["id"]
    return nb


def make_duplicate_ids():
    nb = copy()
    nb["cells"].append(json.loads(json.dumps(nb["cells"][0])))
    return nb


def make_bad_cell_type():
    nb = copy()
    nb["cells"][0]["cell_type"] = "banana"
    return nb


def make_bad_id_pattern():
    nb = copy()
    nb["cells"][0]["id"] = "a b"
    return nb


def make_big_bad_cell():
    nb = copy()
    nb["cells"][0]["cell_type"] = "banana"
    nb["cells"][0]["source"] = ["x = " + "y" * 6000 + "\n"]
    return nb


# (name, notebook builder or None for a raw file, raw content, expected rc, stderr must contain)
CASES = [
    ("valid 4.5 notebook", lambda: copy(), None, 0, None),
    ("valid 4.4 notebook", make_valid_4_4, None, 0, None),
    ("missing cell id", make_missing_id, None, 1, "MissingIDFieldWarning"),
    ("duplicate cell ids", make_duplicate_ids, None, 1, "DuplicateCellId"),
    ("bad cell type", make_bad_cell_type, None, 1, "invalid notebook"),
    ("bad cell id pattern", make_bad_id_pattern, None, 1, "invalid notebook"),
    ("large invalid cell", make_big_bad_cell, None, 1, "invalid notebook"),
    ("non-JSON file", None, "this is not json", 1, "not a valid notebook"),
]


def run_hook(path):
    return subprocess.run([sys.executable, str(HOOK), str(path)], capture_output=True, text=True)


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for name, builder, raw, expected_rc, expected_substring in CASES:
            path = tmpdir / f"case_{name.replace(' ', '_')}.ipynb"
            if raw is not None:
                path.write_text(raw, encoding="utf8")
            else:
                path.write_text(json.dumps(builder()), encoding="utf8")

            result = run_hook(path)
            problems = []
            if result.returncode != expected_rc:
                problems.append(f"rc={result.returncode}, expected {expected_rc}")
            if expected_substring is not None and expected_substring not in result.stderr:
                problems.append(f"missing {expected_substring!r} in stderr")
            # A large invalid cell must not dump its full source: every
            # stderr line stays bounded.
            if name == "large invalid cell":
                longest = max((len(line) for line in result.stderr.splitlines()), default=0)
                if longest > 300:
                    problems.append(f"stderr line of {longest} chars (expected bounded output)")
            # No raw Python warning lines (they carry the hook env path).
            if "site-packages" in result.stderr:
                problems.append("raw warning output leaked to stderr")

            if problems:
                failures += 1
                print(f"FAIL {name}: {'; '.join(problems)}")
                print(f"     stderr: {result.stderr.strip()[:200]}")
            else:
                print(f"PASS {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
