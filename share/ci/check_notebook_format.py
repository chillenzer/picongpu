"""Validate Jupyter notebooks against the nbformat schema.

Used as the entry point of the `check_notebook_format` pre-commit hook
(see .pre-commit-config.yaml). Each notebook is validated against the
schema of the nbformat version it declares. `nbformat` is provided by
the hook environment (`additional_dependencies`).
"""

import sys
import warnings

import nbformat
from nbformat.warnings import DuplicateCellId, MissingIDFieldWarning

# nbformat only emits warnings (and auto-repairs in memory) for these
# issues; they will become hard errors in future nbformat versions, so
# reject them here.
SOFT_WARNINGS = (MissingIDFieldWarning, DuplicateCellId)


def main():
    exit_code = 0
    for filename in sys.argv[1:]:
        try:
            notebook = nbformat.read(filename, as_version=nbformat.NO_CONVERT)
        except Exception as error:  # noqa: BLE001
            print(f"{filename}: {error}", file=sys.stderr)
            exit_code = 1
            continue

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                nbformat.validate(notebook)
            except Exception as error:  # noqa: BLE001
                print(f"{filename}: {error}", file=sys.stderr)
                exit_code = 1
        for warning in caught:
            if issubclass(warning.category, SOFT_WARNINGS):
                print(f"{filename}: {warning.category.__name__}: {warning.message}", file=sys.stderr)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
