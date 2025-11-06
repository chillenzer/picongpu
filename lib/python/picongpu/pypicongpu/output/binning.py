"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import json
from os import PathLike
from typing import Any, Optional

import typeguard
from pydantic import BaseModel, Field

from .. import util
from ..rendering.renderedobject import RenderedObject
from ..species import Species
from .particle_functor import ParticleFunctor as BinningFunctor
from .plugin import Plugin
from .timestepspec import TimeStepSpec


class BinSpec(RenderedObject, BaseModel):
    kind: str
    start: float | int
    stop: float | int
    nsteps: float | int


class BinningAxis(RenderedObject, BaseModel):
    axis_name: str = Field(alias="name")
    bin_spec: BinSpec
    axis_functor: BinningFunctor = Field(alias="functor")
    use_overflow_bins: bool


@typeguard.typechecked
class Binning(Plugin):
    name = util.build_typesafe_property(str)
    species = util.build_typesafe_property(list[Species])
    period = util.build_typesafe_property(TimeStepSpec)
    openPMD = util.build_typesafe_property(Optional[dict])
    openPMDExt = util.build_typesafe_property(Optional[str])
    openPMDInfix = util.build_typesafe_property(Optional[str])
    dumpPeriod = util.build_typesafe_property(int)

    _name = "binning"

    def __init__(
        self, name, deposition_functor, axes, species, period, openPMD, openPMDInfix, dumpPeriod, openPMDExt="bp5"
    ):
        self.name = name
        self.deposition_functor = deposition_functor
        self.axes = axes
        self.species = species
        self.period = period
        self.openPMD = openPMD
        self.openPMDInfix = openPMDInfix
        self.dumpPeriod = dumpPeriod
        self.openPMDExt = openPMDExt

    def _get_serialized(self) -> dict:
        return {
            "binner_name": self.name,
            "deposition_functor": self.deposition_functor.get_rendering_context(),
            "axes": list(map(BinningAxis.get_rendering_context, self.axes)),
            "species": list(map(Species.get_rendering_context, self.species)),
            "period": self.period.get_rendering_context(),
            "openPMD": json.dumps(self.openPMD) if self.openPMD else self.openPMD,
            "openPMDExtension": self.openPMDExt,
            "openPMDInfix": self.openPMDInfix,
            "dumpPeriod": self.dumpPeriod,
        }

    def result_info(self, result_directory: PathLike) -> dict[str, Any]:
        openPMD_options = {
            key: value
            for key, value in zip(["infix", "ext"], [self.openPMDInfix, self.openPMDExt])
            if value is not None
        }
        return {
            "binning": {
                self.name: {
                    "type": "local disk",
                    "path": self._absolute_path(
                        self._fill_openPMD_path(self.name, openPMD_options),
                        working_directory=result_directory,
                        sub_directory="binningOpenPMD",
                    ),
                    "meshes": ["Binning"],
                }
            }
        }
