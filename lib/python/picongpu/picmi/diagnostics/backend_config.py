"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from picongpu.pypicongpu.output.openpmd_plugin import OpenPMDConfig as PyPIConGPUOpenPMDConfig


class BackendConfig:
    def result_path(self, prefix_path):
        raise NotImplementedError()


class OpenPMDConfig(PyPIConGPUOpenPMDConfig, BackendConfig):
    def __init__(self, *args, **kwargs):
        super(PyPIConGPUOpenPMDConfig, self).__init__(*args, **kwargs)
