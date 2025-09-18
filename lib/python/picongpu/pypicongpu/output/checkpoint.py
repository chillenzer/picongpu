"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from os import PathLike
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .plugin import Plugin
from .timestepspec import TimeStepSpec


class Checkpoint(Plugin, BaseModel):
    period: TimeStepSpec | None
    timePeriod: Annotated[int, Field(..., ge=0)] | None
    directory: Path | None
    file: str | None
    restart: bool | None
    tryRestart: bool | None
    restartStep: Annotated[int, Field(..., ge=0)] | None
    restartDirectory: str | None
    restartFile: str | None
    restartChunkSize: Annotated[int, Field(..., gt=0)] | None
    restartLoop: Annotated[int, Field(..., ge=0)] | None
    openPMD: dict | None
    _name: str = PrivateAttr("checkpoint")

    def result_info(self, result_directory: PathLike) -> dict[str, Any]:
        directory = Path(self.directory or "checkpoints")
        if not directory.is_absolute():
            directory = result_directory / directory

        openPMD_options = self.openPMD or {}
        file = Path(
            f"{self.file or 'checkpoint'}{openPMD_options.get('infix', '_%06T')}.{openPMD_options.get('ext', 'bp5')}"
        )

        return [{"checkpoint": {"type": "local disk", "path": file if file.is_absolute() else directory / file}}]

    @model_validator(mode="after")
    def check(self):
        if self.period is None and self.timePeriod is None:
            raise ValueError("At least one of period or timePeriod must be provided")
        return self
