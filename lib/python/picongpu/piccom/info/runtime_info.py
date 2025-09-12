"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path

from picongpu.piccom.schema.info import RuntimeInfo
from .platform_info import platform_information


def gather_runtime_info(self) -> RuntimeInfo:
    return RuntimeInfo(
        platform=platform_information(),
        expected_results={
            f"{plugin._name}": content
            for plugin in self.sim.plugins
            if (content := plugin.result_info(Path(self.run_dir) / "simOutput"))
        },
    )
