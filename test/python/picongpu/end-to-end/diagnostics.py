"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
from pathlib import Path
from unittest import TestCase, main


import numpy as np
from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    GriddedLayout,
    Simulation,
    Species,
)
from picongpu.picmi.diagnostics import (
    Checkpoint,
    DerivedFieldDump,
    FieldDump,
    OpenPMDConfig,
    ParticleDump,
    TimeStepSpec,
)
from picongpu.picmi.diagnostics.particle_functor import ParticleFunctor
from sympy import Piecewise
from scipy.constants import c, epsilon_0
from sympy.vector import CoordSys3D, cross

from .arbitrary_parameters import (
    CELL_SIZE,
    NUMBER_OF_CELLS,
    UPPER_BOUNDARY,
)
from .compare_particles import (
    load_diagnostic_result,
    read_fields,
    read_particles,
    sort_particles,
)
from .distributions import Gaussian, SphereFlanks, _make_vector

logging.basicConfig(level=logging.INFO)

LAYOUT = GriddedLayout(n_macroparticles_per_cell=2)
# This is a debugging shape.
# It arranges for particles
# to be counted exactly in the cell they belong to
# and nowhere else.
PARTICLE_SHAPE = "Counter"
SPECIES = [
    Species(
        name="Gaussian_predefined",
        particle_type="electron",
        particle_shape=PARTICLE_SHAPE,
        initial_distribution=Gaussian().distributions["predefined"],
    ),
    Species(
        name="SphereFlanks_free_form",
        particle_type="electron",
        particle_shape=PARTICLE_SHAPE,
        initial_distribution=SphereFlanks().distributions["free_form"],
    ),
]


def basic_simulation():
    return Simulation(
        max_steps=0,
        solver=ElectromagneticSolver(
            method="Yee",
            cfl=1.0,
            grid=Cartesian3DGrid(
                number_of_cells=NUMBER_OF_CELLS,
                lower_bound=[0, 0, 0],
                # cell size is slightly different from 1
                upper_bound=UPPER_BOUNDARY,
                lower_boundary_conditions=["open", "open", "open"],
                upper_boundary_conditions=["open", "open", "open"],
            ),
        ),
    )


CUTOFF_ENERGY = 10.0


def larmor_power(particle):
    charge = particle.get("charge")
    mass = particle.get("mass")
    gamma = particle.get("gamma")
    dt = particle.get("timestep_size")

    e = CoordSys3D("default")
    momentum = _make_vector(particle.get("momentum"), e)
    previous_momentum = _make_vector(particle.get("momentumPrev1"), e)

    mom_dt = (momentum - previous_momentum) / dt
    el_factor = charge**2 / (6 * np.pi * epsilon_0 * c**2 * mass**2)
    momentumToBetaConvert = 1 / (mass * c * gamma)

    return el_factor * (mom_dt.magnitude() ** 2 - momentumToBetaConvert**2 * cross(momentum, mom_dt).magnitude() ** 2)


def generate_diagnostics(species):
    options = OpenPMDConfig(file="other_name", ext=".h5", infix="", data_preparation_strategy="doubleBuffer")
    particles = [ParticleDump(species=s) for s in species] + [ParticleDump(species=species[0], options=options)]
    native_fields = [FieldDump(fieldname=fieldname) for fieldname in ["E", "B"]]
    functors = [
        # ParticleFunctor(
        #     name="bound_electrons", functor=lambda particle: particle.get("boundElectrons"), return_type=float
        # ),
        # ParticleFunctor(
        #     name="charge_density",
        #     functor=lambda particle: particle.get("charge") / np.prod(CELL_SIZE),
        #     return_type=float,
        # ),
        ParticleFunctor(name="particle_counter", functor=lambda particle: particle.get("weighting"), return_type=float),
        ParticleFunctor(
            name="density", functor=lambda particle: particle.get("weighting") / np.prod(CELL_SIZE), return_type=float
        ),
        ParticleFunctor(
            name="kinetic_energy", functor=lambda particle: particle.get("kinetic energy"), return_type=float
        ),
        ParticleFunctor(
            name="kinetic_energy_density",
            functor=lambda particle: particle.get("kinetic energy") / np.prod(CELL_SIZE),
            return_type=float,
        ),
        ParticleFunctor(
            name="kinetic_energy_density_cutoff",
            functor=lambda particle: Piecewise(
                (
                    particle.get("kinetic energy") / np.prod(CELL_SIZE),
                    particle.get("kinetic energy") < CUTOFF_ENERGY * particle.get("weighting"),
                ),
                (0.0, True),
            ),
            return_type=float,
        ),
        # ParticleFunctor(name="larmor_power", functor=larmor_power, return_type=float),
        ParticleFunctor(name="macroparticle_counter", functor=lambda _: 1, return_type=int),
        # ParticleFunctor(
        #     name="mid_current_density_x",
        #     functor=lambda particle: particle.get("charge")
        #     / np.prod(CELL_SIZE)
        #     * particle.get("momentum")[0]
        #     / (particle.get("gamma") * particle.get("mass")),
        #     return_type=int,
        # ),
        ParticleFunctor(name="momentum_y", functor=lambda particle: particle.get("momentum")[1], return_type=float),
        ParticleFunctor(
            name="momentum_density_z",
            functor=lambda particle: particle.get("momentum")[2] / np.prod(CELL_SIZE),
            return_type=float,
        ),
        ParticleFunctor(
            name="weighted_velocity_x",
            functor=lambda particle: particle.get("velocity")[0] * particle.get("weighting"),
            return_type=float,
        ),
    ]
    derived_fields = [DerivedFieldDump(species=s, functor=f) for s in species for f in functors]
    return particles + native_fields + derived_fields


RUN_DIR = ""


def setup_sim():
    sim = basic_simulation()
    for species in SPECIES:
        sim.add_species(species, LAYOUT)
    sim.diagnostics = [Checkpoint(TimeStepSpec[:])] + generate_diagnostics(SPECIES)
    if RUN_DIR:
        sim.picongpu_get_runner().run_dir = RUN_DIR
    else:
        sim.step(0)
    return sim


SIM = None


class TestDiagnostics(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
        self.sim = SIM

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim.picongpu_get_runner().run_dir)
        return self._result_path

    def test_particle_dump(self):
        for diag in self.sim.diagnostics:
            if isinstance(diag, ParticleDump):
                from_checkpoint = sort_particles(
                    read_particles(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5")
                ).loc(axis=0)[*diag.species.name.split("_", maxsplit=1)]
                from_diagnostics = sort_particles(load_diagnostic_result(diag, self.result_path))
                np.testing.assert_allclose(from_checkpoint, from_diagnostics)

    def test_field_dump(self):
        for diag in self.sim.diagnostics:
            if isinstance(diag, FieldDump) and not isinstance(diag, DerivedFieldDump):
                np.testing.assert_allclose(
                    load_diagnostic_result(diag, self.result_path),
                    read_fields(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5")[
                        diag.fieldname
                    ],
                )

    def test_derived_fields(self):
        particles = read_particles(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5")
        for diag in self.sim.diagnostics:
            if isinstance(diag, DerivedFieldDump):
                my_particles = particles.loc(axis=0)[*diag.species.name.split("_", maxsplit=1)]
                expected = diag(self.sim.solver.grid, my_particles)
                result = load_diagnostic_result(diag, self.result_path).transpose((2, 1, 0))
                # The openPMD data apparently has NaNs where there wasn't a particle at all.
                result[np.isnan(result)] = 0.0
                np.testing.assert_allclose(result, expected, rtol=1.0e-5, atol=1.0e-5)


if __name__ == "__main__":
    main()
