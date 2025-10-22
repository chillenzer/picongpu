"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from os import PathLike
from pathlib import Path
from typing import Literal

from picongpu.picmi.species import Shape

import numpy as np
from pydantic import BaseModel

from picongpu.picmi.diagnostics.particle_functor import (
    Particle,
    ParticleFunctor,
    make_particle,
)
from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.openpmd_plugin import (
    NATIVE_FIELDS,
    PREDEFINED_DERIVED_ATTRIBUTES,
)

from .backend_config import BackendConfig, OpenPMDConfig
from .timestepspec import TimeStepSpec


class FieldDump(BaseModel):
    fieldname: str
    period: TimeStepSpec = TimeStepSpec[:]("steps")
    options: BackendConfig = OpenPMDConfig(file="simData")
    is_predefined: bool = True

    class Config:
        arbitrary_types_allowed = True

    def result_path(self, prefix_path: PathLike):
        return self.options.result_path(prefix_path=Path(prefix_path) / "simOutput" / "openPMD")


class NativeFieldDump(BaseModel):
    fieldname: Literal[*NATIVE_FIELDS]


class DerivedFieldDump(FieldDump):
    species: Species
    functor: ParticleFunctor
    is_predefined: bool = False

    def __init__(self, *args, **kwargs):
        if "fieldname" in kwargs:
            raise ValueError("fieldname gets internally computed in a DerivedFieldDump. Please don't try to set it.")
        name = kwargs["functor"].name
        kwargs["fieldname"] = f"{kwargs['species'].name}_all_{PREDEFINED_DERIVED_ATTRIBUTES.get(name, name)}"
        return super().__init__(*args, **kwargs)

    def __call__(self, grid, particle):
        if not isinstance(particle, Particle):
            return self(grid, make_particle(particle))
        return self._distribute_to_grid(grid, self.functor(particle))

    def _distribute_to_grid(self, grid, particles):
        if Shape[self.species.particle_shape.upper()] != Shape.COUNTER:
            raise NotImplementedError(
                f"Currently only naive distribution to cells is supported. Your species has {self.species.particle_shape=}. Only COUNTER is allowed."
            )
        data = (
            particles.set_index(
                ["position_x", "position_y", "position_z", "positionOffset_x", "positionOffset_y", "positionOffset_z"]
            )[self.functor.name]
            .groupby(by=lambda pos: grid.compute_cell_index(np.reshape(pos, [2, 3]).sum(axis=0)))
            .sum()
        )
        result = np.zeros(grid.number_of_cells).reshape(-1)
        result[data.index] = data.to_numpy()
        return result.reshape(grid.number_of_cells)


def generate_predefined_attribute(attribute):
    return lambda *args, **kwargs: DerivedFieldDump(
        *args,
        **kwargs,
        functor=ParticleFunctor(name=attribute, functor=lambda particle: particle.get("position"), return_type=float),
        is_predefined=True,
    )


for attribute in PREDEFINED_DERIVED_ATTRIBUTES:
    globals()[attribute] = generate_predefined_attribute(attribute)
