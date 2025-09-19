"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Brian Edward Marre, Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from os import PathLike
from pathlib import Path
from typing import Any

import typeguard

from ..rendering import SelfRegisteringRenderedObject


@typeguard.typechecked
class Plugin(SelfRegisteringRenderedObject):
    """general interface for all plugins"""

    def _absolute_path(
        self,
        filename: PathLike | str,
        working_directory: PathLike | str | None = None,
        sub_directory: PathLike | str | None = None,
    ):
        """
        Implement convention how filenames are handled.
        """
        if Path(filename).is_absolute():
            return filename

        directory = Path(sub_directory or ".")
        if not directory.is_absolute():
            directory = Path(working_directory or ".").absolute() / directory

        return directory / filename

    def _fill_openPMD_path(self, filename: PathLike | str, openPMD_options: dict[str, str] | None = None):
        openPMD_options = openPMD_options or {}
        return Path(
            f"{filename or 'checkpoint'}{openPMD_options.get('infix', '_%06T')}.{openPMD_options.get('ext', 'bp5')}"
        )

    def result_info(self, result_directory: PathLike) -> dict[str, Any]:
        """
        Return a list of dictionaries where to find the results

        The result_directory might be used in case the path is configured to be relative.
        """
        return {}
