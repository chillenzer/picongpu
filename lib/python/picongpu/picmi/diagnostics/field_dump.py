"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from os import PathLike
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.openpmd_plugin import (
    NATIVE_FIELDS,
    PREDEFINED_DERIVED_ATTRIBUTES,
    ParticleFunctor as PyPIConGPUParticleFunctor,
)

from .backend_config import BackendConfig, OpenPMDConfig
from .timestepspec import TimeStepSpec


class FieldDump(BaseModel):
    fieldname: str
    period: TimeStepSpec = TimeStepSpec[:]("steps")
    options: BackendConfig = OpenPMDConfig(file="simData")

    class Config:
        arbitrary_types_allowed = True

    def result_path(self, prefix_path: PathLike):
        return self.options.result_path(prefix_path=Path(prefix_path) / "simOutput" / "openPMD")


class NativeFieldDump(BaseModel):
    fieldname: Literal[*NATIVE_FIELDS]


class ParticleFunctor(PyPIConGPUParticleFunctor):
    pass


class DerivedFieldDump(FieldDump):
    species: Species
    functor: ParticleFunctor

    def __init__(self, *args, **kwargs):
        if "fieldname" in kwargs:
            raise ValueError("fieldname gets internally computed in a DerivedFieldDump. Please don't try to set it.")
        kwargs["fieldname"] = f"{kwargs['species'].name}_all_{PREDEFINED_DERIVED_ATTRIBUTES[kwargs['functor'].name]}"
        return super().__init__(*args, **kwargs)


for attribute in PREDEFINED_DERIVED_ATTRIBUTES:
    globals()[attribute] = lambda *args, **kwargs: DerivedFieldDump(
        *args, **kwargs, functor=ParticleFunctor(name=attribute, functor=None)
    )
