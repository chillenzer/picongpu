"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from picongpu.picmi.diagnostics import Binning, Checkpoint, TimeStepSpec
from picongpu.picmi.diagnostics.binning import BinningAxis, BinningFunctor, BinSpec
import numpy as np
from scipy.constants import eV
from sympy import atan2


def generate_diagnostics(species):
    electron_spectrum = Binning(
        "electron_spectrum",
        axes=[
            BinningAxis(
                BinningFunctor("Energy", lambda particle: particle.get("kinetic energy"), float),
                BinSpec("linear", 0.0, 20.0 * eV, 800),
            ),
            BinningAxis(
                BinningFunctor("pointingXY", lambda particle: atan2(*particle.get("momentum")[:2]), float),
                BinSpec("linear", -np.pi, np.pi, 256),
            ),
        ],
        deposition_functor=BinningFunctor("Charge", lambda particle: particle.get("charge"), float),
        species=species,
        period=TimeStepSpec[::100],
    )
    checkpoint = Checkpoint(period=TimeStepSpec[::100])
    return [checkpoint, electron_spectrum]
