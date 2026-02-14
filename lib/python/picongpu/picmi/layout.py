"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from functools import partial
from typing import Annotated, Self
import numpy as np
import picmistandard
from pydantic import AfterValidator, BaseModel, Field, computed_field, model_validator
from picmistandard.base import broadcast_validation

from ..pypicongpu.species.operation.layout import OnePosition as PyPIConGPU_OnePosition
from ..pypicongpu.species.operation.layout import Quiet, Random


class PseudoRandomLayout(picmistandard.PICMI_PseudoRandomLayout):
    n_macroparticles_per_cell: int = Field(gt=0)
    # PIConGPU can't handle the following separately:
    n_macroparticles: None = None
    seed: None = None
    grid: None = None

    def get_as_pypicongpu(self):
        return Random(ppc=self.n_macroparticles_per_cell)


class GriddedLayout(picmistandard.PICMI_GriddedLayout):
    n_macroparticles_per_cell: list[int] = Field([0], init_var=False)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        self.n_macroparticles_per_cell = self.n_macroparticle_per_cell
        return self

    def get_as_pypicongpu(self):
        return Quiet(ppc=np.prod(self.n_macroparticle_per_cell), n_points=self.n_macroparticle_per_cell)

    @computed_field
    def in_cell_offsets(self) -> np.ndarray:
        return (np.mgrid[*map(slice, self.n_macroparticles_per_cell)] + 0.5).reshape(
            len(self.n_macroparticles_per_cell), -1
        ).T / self.n_macroparticles_per_cell


class OnePosition(BaseModel):
    n_macroparticles_per_cell: int = Field(gt=0, description="Number of particles per cell")
    in_cell_offset: Annotated[
        list[float],
        AfterValidator(
            partial(
                broadcast_validation,
                condition=lambda v: v >= 0.0 and v < 1.0,
                message="In-cell offset is relative to cell size and must be between 0 and 1.",
            )
        ),
    ] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
        description="Offset to cell origin where the particles are placed.",
    )
    grid: None = None

    def get_as_pypicongpu(self):
        return PyPIConGPU_OnePosition(ppc=self.n_macroparticles_per_cell, in_cell_offset=self.in_cell_offset)


AnyLayout = PseudoRandomLayout | GriddedLayout | OnePosition
