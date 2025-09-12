"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Any

from pydantic import BaseModel


class PlatformInfo(BaseModel):
    platform: str
    system: str
    node: str
    release: str
    version: str
    machine: str
    processor: str


class RuntimeInfo(BaseModel):
    platform: PlatformInfo
    expected_results: dict[str, Any]


class GitInfo(BaseModel):
    log: str
    status: str
    diff: str


class CompiletimeInfo(BaseModel):
    platform: PlatformInfo
    git: GitInfo
