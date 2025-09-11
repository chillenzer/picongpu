"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Literal


AVAILABLE_METADATA_FORMAT_VERSIONS = ["0.1.0"]
METADATA_FORMAT_VERSION = "0.1.0"

AnyMetadataFormatVersion = Literal[*AVAILABLE_METADATA_FORMAT_VERSIONS]
