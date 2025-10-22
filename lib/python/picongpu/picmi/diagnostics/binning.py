"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path

import numpy as np
import typeguard

from picongpu.picmi.diagnostics.backend_config import OpenPMDConfig

from ...pypicongpu.output.binning import Binning as PyPIConGPUBinning
from ...pypicongpu.output.binning import BinningAxis as PyPIConGPUBinningAxis
from ...pypicongpu.output.binning import BinSpec as PyPIConGPUBinSpec
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies
from ..species import Species as PICMISpecies
from .particle_functor import Particle, make_particle
from .particle_functor import ParticleFunctor as BinningFunctor
from .timestepspec import TimeStepSpec


@typeguard.typechecked
class BinSpec:
    def __init__(self, kind, start, stop, nsteps):
        self.kind = kind
        self.start = start
        self.stop = stop
        self.nsteps = nsteps

    def get_as_pypicongpu(self):
        return PyPIConGPUBinSpec(self.kind.lower().capitalize(), self.start, self.stop, self.nsteps)

    @property
    def bins(self):
        if self.kind.lower() == "linear":
            return np.linspace(self.start, self.stop, self.nsteps + 1, endpoint=True)
        raise NotImplementedError("Computing bins for other than linear BinSpecs is not implemented.")


@typeguard.typechecked
class BinningAxis:
    def __init__(
        self,
        functor: BinningFunctor,
        bin_spec: BinSpec,
        name: str | None = None,
        use_overflow_bins: bool = True,
    ):
        self.functor = functor
        self.bin_spec = bin_spec
        self.name = name or functor.name
        self.use_overflow_bins = use_overflow_bins

    def get_as_pypicongpu(self) -> PyPIConGPUBinningAxis:
        return PyPIConGPUBinningAxis(
            name=self.name,
            functor=self.functor.get_as_pypicongpu(),
            bin_spec=self.bin_spec.get_as_pypicongpu(),
            use_overflow_bins=self.use_overflow_bins,
        )

    def __call__(self, particle):
        return np.digitize(self.functor(particle)[self.functor.name].to_numpy(), self.bin_spec.bins)


@typeguard.typechecked
class Binning:
    def __init__(
        self,
        name: str,
        deposition_functor: BinningFunctor,
        axes: list[BinningAxis],
        species: PICMISpecies | list[PICMISpecies],
        period: TimeStepSpec | None = None,
        openPMD: dict | None = None,
        openPMDExt: str | None = None,
        openPMDInfix: str | None = None,
        dumpPeriod: int = 1,
    ):
        self.name = name
        self.deposition_functor = deposition_functor
        self.axes = axes
        if isinstance(species, PICMISpecies):
            species = [species]
        self.species = species
        self.period = period or TimeStepSpec[:]
        self.openPMD = openPMD
        self.openPMDExt = openPMDExt
        self.openPMDInfix = openPMDInfix
        self.dumpPeriod = dumpPeriod

    def result_path(self, prefix_path):
        return OpenPMDConfig(
            file=self.name, ext=self.openPMDExt or ".bp5", infix=self.openPMDInfix or "_%06T"
        ).result_path(prefix_path=Path(prefix_path) / "simOutput" / "binningOpenPMD")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size,
        num_steps,
    ) -> PyPIConGPUBinning:
        if len(not_found := [s for s in self.species if s not in dict_species_picmi_to_pypicongpu.keys()]) > 0:
            raise ValueError(f"Species {not_found} are not known to Simulation")
        pypic_species = list(map(dict_species_picmi_to_pypicongpu.get, self.species))

        return PyPIConGPUBinning(
            name=self.name,
            deposition_functor=self.deposition_functor.get_as_pypicongpu(),
            axes=list(map(BinningAxis.get_as_pypicongpu, self.axes)),
            species=pypic_species,
            period=self.period.get_as_pypicongpu(time_step_size, num_steps),
            openPMD=self.openPMD,
            openPMDExt=self.openPMDExt,
            openPMDInfix=self.openPMDInfix,
            dumpPeriod=self.dumpPeriod,
        )

    def __call__(self, particle):
        if not isinstance(particle, Particle):
            return self(make_particle(particle))
        result = np.zeros([a.bin_spec.nsteps for a in self.axes])
        result[np.transpose([axis(particle) for axis in self.axes])] += self.deposition_functor(particle)[
            self.deposition_functor.name
        ]
        return result
