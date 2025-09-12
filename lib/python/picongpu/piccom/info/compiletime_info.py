"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import subprocess
from pathlib import Path
from .platform_info import platform_information
from picongpu.piccom.schema.info import CompiletimeInfo


def _run_or_error_message(*args):
    try:
        return subprocess.run(*args, capture_output=True, text=True).stdout
    except Exception:
        return "failed"


def _get_version_control_info():
    my_path = str(Path(__file__).parent)
    return {
        "log": _run_or_error_message(["git", "-C", my_path, "log", "-n", "1"]),
        "status": _run_or_error_message(["git", "-C", my_path, "status"]),
        "diff": _run_or_error_message(["git", "-C", my_path, "diff"]),
    }


def gather_compiletime_info(_) -> CompiletimeInfo:
    return CompiletimeInfo(git=_get_version_control_info(), platform=platform_information())
