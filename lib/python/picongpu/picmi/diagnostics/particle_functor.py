"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Callable, Any
from ...pypicongpu.output.binning import (
    BinningFunctor as PyPIConGPUParticleFunctor,
)
from typeguard import typechecked
import sympy

_COORDINATE_SYSTEM = {
    (
        origin.lower(),
        precision.lower(),
        unit.lower(),
    ): tuple(sympy.Symbol(f"{c}_{precision.lower()}_{unit.lower()}") for c in coords)
    for (origin, coords) in (
        ("TOTAL", ("xt", "yt", "zt")),
        ("GLOBAL", ("xg", "yg", "zg")),
        ("LOCAL", ("xl", "yl", "zl")),
        ("MOVING_WINDOW", ("xmw", "ymw", "zmw")),
        ("LOCAL_WITH_GUARDS", ("xlg", "ylg", "zlg")),
    )
    for precision in ("CELL", "SUB_CELL")
    for unit in ("CELL", "PIC", "SI")
}


@typechecked
class Particle:
    def __init__(self):
        self.used_attributes = {}

    def get_attribute_map(self):
        return self.used_attributes

    def get(self, attribute, **kwargs):
        if attribute == "position":
            origin = kwargs.get("origin", "total")
            precision = kwargs.get("precision", "cell")
            unit = kwargs.get("unit", "cell")
            symbols = _COORDINATE_SYSTEM[(origin, precision, unit)]
            self.used_attributes |= {symbols: ("position", origin, precision, unit)}

        elif attribute == "momentum":
            symbols = sympy.symbols("px,py,pz")
            self.used_attributes |= {symbols: "momentum"}

        elif attribute in ["gamma", "kinetic energy", "velocity"]:
            # This relies on python dictionaries having a stable ordering.
            # We first add mass and momentum
            # and later use their symbols inside of the same preamble.
            self.get("mass")
            self.get("momentum")
            if attribute == "gamma":
                symbols = sympy.Symbol("gamma")
            elif attribute == "kinetic energy":
                symbols = sympy.Symbol("Ekin")
            elif attribute == "velocity":
                symbols = sympy.symbols("vx,vy,vz")
            else:
                raise ValueError("Reached impossible path.")
            self.used_attributes |= {symbols: attribute}

        else:
            symbols = sympy.Symbol(attribute)
            self.used_attributes |= {symbols: attribute}

        return symbols


@typechecked
class ParticleFunctor:
    def check(self):
        pass

    def __init__(
        self,
        name: str,
        functor: Callable[[Particle], Any],
        return_type: type | str,
    ):
        self.name = name
        self.functor = functor
        self.return_type = return_type

    def get_as_pypicongpu(self) -> PyPIConGPUParticleFunctor:
        self.check()
        particle = Particle()
        functor_expression = self.functor(particle)
        return PyPIConGPUParticleFunctor(
            name=self.name,
            functor_expression=functor_expression,
            attribute_mapping=particle.get_attribute_map(),
            return_type=self.return_type,
        )
