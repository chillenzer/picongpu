"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging

from infrastructure.communicator import COMMUNICATOR
from infrastructure.diagnostics import generate_diagnostics
from infrastructure.parameters import (
    BOX_SIZE,
    DENSITY,
    DIRECTORIES,
    DURATIONS,
    MAX_STEPS,
    NUM_CELLS,
    WIDTHS,
)
from infrastructure.postprocessing import postprocessing
from infrastructure.species import LAYOUT, generate_species

from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    GaussianLaser,
    Simulation,
)
from picongpu.picmi.lasers import PolarizationType

logging.basicConfig(level=logging.DEBUG)

"""
@file PICMI user script reproducing the PIConGPU LWFA example

This Python script is example PICMI user script reproducing the LaserWakefield example setup, based on 8.cfg.

"""


def generate_simulation(communicator, width, duration):
    species = generate_species(width, duration)
    return Simulation(
        solver=ElectromagneticSolver(
            grid=Cartesian3DGrid(
                picongpu_n_gpus=[1, 1, 1],
                number_of_cells=NUM_CELLS,
                lower_bound=[0, 0, 0],
                upper_bound=BOX_SIZE,
                lower_boundary_conditions=["open", "open", "open"],
                upper_boundary_conditions=["open", "open", "open"],
            ),
            method="Yee",
            cfl=0.95,
        ),
        max_steps=MAX_STEPS,
        picongpu_species=[(s, LAYOUT) for s in species],
        picongpu_diagnostics=generate_diagnostics(species),
        picongpu_laser=GaussianLaser(
            wavelength=0.8e-6,
            waist=BOX_SIZE[0] / 4,
            duration=duration,
            propagation_direction=[0.0, 1.0, 0.0],
            polarization_direction=[1.0, 0.0, 0.0],
            focal_position=BOX_SIZE / 2,
            centroid_position=BOX_SIZE / 2 * [1, -1, 1],
            picongpu_polarization_type=PolarizationType.LINEAR,
            a0=8.0,
            phi0=0.0,
        ),
        picongpu_communicator=communicator,
    )


def run(communicator, widths, durations):
    for width in widths:
        for duration in durations:
            communicator.record_additionally({"density": DENSITY, "width": width})
            sim = generate_simulation(communicator, width, duration)
            sim.picongpu_run(
                setup_dir=str(DIRECTORIES["setup"](width, duration)),
                run_dir=str(DIRECTORIES["run"](width, duration)),
            )


def main():
    communicator = COMMUNICATOR
    run(communicator, WIDTHS, DURATIONS)
    postprocessing(communicator)


if __name__ == "__main__":
    main()
