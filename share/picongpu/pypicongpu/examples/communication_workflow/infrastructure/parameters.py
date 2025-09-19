"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path

import numpy as np
from sympy import And, Piecewise

WORKING_DIRECTORY = Path("lwfa").absolute()
DIRECTORIES = {
    "setup": lambda duration, width: (WORKING_DIRECTORY / "setups" / f"{duration}_{width}").absolute(),
    "run": lambda duration, width: (WORKING_DIRECTORY / "runs" / f"{duration}_{width}").absolute(),
    "database": lambda *_: (WORKING_DIRECTORY / "database").absolute(),
    "plot": lambda *_: (WORKING_DIRECTORY / "plots").absolute(),
}

NUM_CELLS = np.array([64, 256, 64])
BOX_SIZE = np.array([3.40224e-05, 3.0e-05, 3.40224e-05])
CELL_SIZE = BOX_SIZE / NUM_CELLS

DENSITY = 1.0e25
MAX_STEPS = 1000

WIDTHS = np.linspace(0.4, 0.6, 3) * BOX_SIZE[1]
DURATIONS = np.linspace(5.0, 15.0, 4) * 1.0e-15


def foil(density, width):
    return lambda x, y, z: density * Piecewise(
        (1, And(y > BOX_SIZE[1] / 2 - width / 2, y < BOX_SIZE[1] / 2 + width / 2)),
        (0, True),
    )
