"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import numpy as np
import picmistandard
from pydantic import BaseModel, Field

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
    def get_as_pypicongpu(self):
        return Quiet(ppc=np.prod(self.n_macroparticles_per_cell), n_points=self.n_macroparticle_per_cell)


class OnePosition(BaseModel):
    n_macroparticles_per_cell: int = Field(gt=0, description="Number of particles per cell")
    in_cell_offset: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        ge=0,
        lt=1.0,
        min_length=3,
        max_length=3,
        description="Offset to cell origin where the particles are placed.",
    )
    grid: None = None

    def get_as_pypicongpu(self):
        return PyPIConGPU_OnePosition(ppc=self.n_macroparticles_per_cell, in_cell_offset=self.in_cell_offset)


AnyLayout = PseudoRandomLayout | GriddedLayout | OnePosition
