"""Validate Jupyter notebooks against the nbformat schema.

Used as the entry point of the `check_notebook_format` pre-commit hook
(see .pre-commit-config.yaml). Each notebook is validated against the
schema of the nbformat version it declares. `nbformat` is provided by
the hook environment (`additional_dependencies`).
"""

import sys

import nbformat


def main():
    exit_code = 0
    for filename in sys.argv[1:]:
        try:
            notebook = nbformat.read(filename, as_version=nbformat.NO_CONVERT)
            nbformat.validate(notebook)
        except Exception as error:  # noqa: BLE001
            print(f"{filename}: {error}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
