"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import platform

from picongpu.piccom.schema.info import PlatformInfo


def platform_information() -> PlatformInfo:
    return PlatformInfo(platform=platform.platform(), **platform.uname()._asdict())
