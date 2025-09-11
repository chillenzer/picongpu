"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Brian Edward Marre, Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from os import PathLike
from typing import Any

import typeguard

from ..rendering import SelfRegisteringRenderedObject


@typeguard.typechecked
class Plugin(SelfRegisteringRenderedObject):
    """general interface for all plugins"""

    def result_info(self, result_directory: PathLike) -> list[dict[str, Any]]:
        """
        Return a list of dictionaries where to find the results

        The result_directory might be used in case the path is configured to be relative.
        """
        return []
