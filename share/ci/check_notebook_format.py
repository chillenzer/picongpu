"""Validate Jupyter notebooks against the nbformat schema.

Used as the entry point of the `check_notebook_format` pre-commit hook
(see .pre-commit-config.yaml). Each notebook is validated against the
schema of the nbformat version it declares. `nbformat` is provided by
the hook environment (`additional_dependencies`).
"""

import logging
import sys
import warnings

import nbformat
from nbformat.warnings import DuplicateCellId, MissingIDFieldWarning

# nbformat only emits warnings (and auto-repairs in memory) for these
# issues; they will become hard errors in future nbformat versions, so
# reject them here.
SOFT_WARNINGS = (MissingIDFieldWarning, DuplicateCellId)

# jsonschema error messages embed the whole offending instance, which
# can be many KB for a cell with a long source; keep the first (bounded)
# line only so bad cells do not dump their full source into CI logs.
MAX_MESSAGE_LEN = 200


def shorten(text):
    line = text.strip().splitlines()[0] if text.strip() else str(text)
    if len(line) > MAX_MESSAGE_LEN:
        line = line[:MAX_MESSAGE_LEN].rstrip() + "..."
    return line


def check_notebook(filename):
    # nbformat.read() validates internally as well: hard errors are
    # swallowed and reported through the `nbformat` logger, while soft
    # ones (missing/duplicate cell ids) are emitted as raw warnings.
    # Wrap both calls so the warnings are captured here, and quiet the
    # logger so hard errors are reported exactly once (by validate()
    # below, with a bounded message).
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            notebook = nbformat.read(filename, as_version=nbformat.NO_CONVERT)
            nbformat.validate(notebook)
    except nbformat.ValidationError as error:
        print(f"{filename}: invalid notebook: {shorten(str(error))}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"{filename}: not a valid notebook: {error}", file=sys.stderr)
        return 1
    for warning in caught:
        if issubclass(warning.category, SOFT_WARNINGS):
            print(f"{filename}: {warning.category.__name__}: {warning.message}", file=sys.stderr)
            return 1
    return 0


def main():
    nbformat_logger = logging.getLogger("nbformat")
    saved_level = nbformat_logger.level
    nbformat_logger.setLevel(logging.CRITICAL)
    exit_code = 0
    try:
        for filename in sys.argv[1:]:
            exit_code |= check_notebook(filename)
        return exit_code
    finally:
        nbformat_logger.setLevel(saved_level)


if __name__ == "__main__":
    sys.exit(main())
