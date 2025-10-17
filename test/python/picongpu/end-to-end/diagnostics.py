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
from .arbitrary_parameters import CELL_SIZE, NUMBER_OF_CELLS, UPPER_BOUNDARY, MACRO_PARTICLES_PER_CELL
from .compare_particles import (
    load_diagnostic_result,
    read_densities_into_mesh,
    read_particles,
    sort_particles,
    read_fields,
)
from .distributions import Gaussian, SphereFlanks
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
from picongpu.picmi.diagnostics.field_dump import PREDEFINED_DERIVED_ATTRIBUTES
from picongpu.picmi import diagnostics

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


def generate_diagnostics(species):
    options = OpenPMDConfig(file="other_name", ext=".h5", infix="", data_preparation_strategy="doubleBuffer")
    particles = [ParticleDump(species=s) for s in species] + [
        ParticleDump(species=species[0], options=options),
    ]
    native_fields = [FieldDump(fieldname=fieldname) for fieldname in ["E", "B"]]
    derived_fields = [
        d
        for s in species
        for DerivedAttribute in map(lambda x: diagnostics.__dict__[x], PREDEFINED_DERIVED_ATTRIBUTES)
        # LarmorPower and BoundElectronDensity need special attributes and those are not yet implemented
        if (d := DerivedAttribute(species=s)).functor.name not in ["LarmorPower", "BoundElectronDensity"]
    ]
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

DERIVED_FIELD_CONVERSIONS_FROM_COUNTER = {
    "MacroCounter": lambda counter: (counter > 0) * MACRO_PARTICLES_PER_CELL,
    # "Energy": lambda counter: np.zeros_like(counter),
    # Looks like it works up to a pre-factor
    # but getting the details right turned out to be non-trivial:
    # 'Density': lambda counter: counter,
}


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

    def test_counter_equals_density(self):
        for diag in self.sim.diagnostics:
            if isinstance(diag, DerivedFieldDump) and diag.functor.name == "Counter":
                from_checkpoint = read_densities_into_mesh(
                    self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5", NUMBER_OF_CELLS, CELL_SIZE
                ).loc(axis=0)[*diag.species.name.split("_", maxsplit=1)]
                # the particle counter does not include the factor of base density:
                from_checkpoint /= 1.0e25
                # not quite sure about the factor 1/2 yet, could be MACRO_PARTICLES_PER_CELL?
                from_diagnostics = load_diagnostic_result(diag, self.result_path).transpose((2, 1, 0)) / 2
                np.testing.assert_allclose(from_checkpoint, from_diagnostics)

    def test_diagnostics_consistent_with_counter(self):
        for diag in self.sim.diagnostics:
            if isinstance(diag, DerivedFieldDump) and diag.functor.name in DERIVED_FIELD_CONVERSIONS_FROM_COUNTER:
                from_diag = load_diagnostic_result(diag, self.result_path)
                counter = next(
                    filter(
                        lambda d: isinstance(d, DerivedFieldDump)
                        and d.functor.name == "Counter"
                        and d.species == diag.species,
                        self.sim.diagnostics,
                    )
                )
                from_counter = load_diagnostic_result(counter, self.result_path)
                np.testing.assert_allclose(
                    from_diag, DERIVED_FIELD_CONVERSIONS_FROM_COUNTER[diag.functor.name](from_counter)
                )

    def test_densities_consistent_with_non_normalised_values(self):
        for diag in self.sim.diagnostics:
            if isinstance(diag, DerivedFieldDump) and "Density" in diag.functor.name:
                density_name = diag.functor.name
                name = density_name.replace("Density", "")
                try:
                    nonnormalised_diag = next(
                        filter(
                            lambda d: isinstance(d, DerivedFieldDump)
                            and d.functor.name == name
                            and d.species == diag.species,
                            self.sim.diagnostics,
                        )
                    )
                except StopIteration:
                    # We don't have non-normalised values available.
                    pass
                else:
                    density = load_diagnostic_result(diag, self.result_path)
                    nonnormalised = load_diagnostic_result(nonnormalised_diag, self.result_path)
                    np.testing.assert_allclose(density, nonnormalised / np.prod(CELL_SIZE))


if __name__ == "__main__":
    main()
