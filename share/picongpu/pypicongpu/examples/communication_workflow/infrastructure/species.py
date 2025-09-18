"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from .parameters import DENSITY, foil
from picongpu.picmi import Species, PseudoRandomLayout
from picongpu.picmi.distribution import AnalyticDistribution


def generate_species(width, duration):
    return [
        Species(
            particle_type="electron",
            name="electron",
            initial_distribution=AnalyticDistribution(foil(DENSITY, width)),
        )
    ]


LAYOUT = PseudoRandomLayout(n_macroparticles_per_cell=2)
