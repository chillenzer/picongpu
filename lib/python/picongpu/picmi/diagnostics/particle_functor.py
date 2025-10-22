"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Any, Callable, Iterable

import pandas as pd
from sympy import Expr, sqrt, symbols, Symbol, lambdify
from scipy.constants import c
from typeguard import typechecked

from ...pypicongpu.output.binning import (
    BinningFunctor as PyPIConGPUParticleFunctor,
)

_COORDINATE_SYSTEM = {
    (
        origin.lower(),
        precision.lower(),
        unit.lower(),
    ): tuple(Symbol(f"{c}_{precision.lower()}_{unit.lower()}") for c in coords)
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


class Particle:
    def get(self, attribute, **kwargs) -> Expr | Iterable[Expr]:
        NotImplementedError()


@typechecked
class AbstractParticle(Particle):
    def __init__(self):
        self.used_attributes = {}

    def get_attribute_map(self):
        return self.used_attributes

    def get(self, attribute, **kwargs) -> Expr | Iterable[Expr]:
        if attribute == "position":
            origin = kwargs.get("origin", "total")
            precision = kwargs.get("precision", "cell")
            unit = kwargs.get("unit", "cell")
            my_symbols = _COORDINATE_SYSTEM[(origin, precision, unit)]
            self.used_attributes |= {my_symbols: ("position", origin, precision, unit)}

        elif attribute == "momentum":
            my_symbols = symbols("px,py,pz")
            self.used_attributes |= {my_symbols: "momentum"}

        elif attribute in ["gamma", "kinetic energy", "velocity"]:
            # This relies on python dictionaries having a stable ordering.
            # We first add mass and momentum
            # and later use their symbols inside of the same preamble.
            self.get("mass")
            self.get("momentum")
            if attribute == "gamma":
                my_symbols = Symbol("gamma")
            elif attribute == "kinetic energy":
                my_symbols = Symbol("Ekin")
            elif attribute == "velocity":
                my_symbols = symbols("vx,vy,vz")
            else:
                raise ValueError("Reached impossible path.")
            self.used_attributes |= {my_symbols: attribute}

        else:
            my_symbols = Symbol(attribute)
            self.used_attributes |= {my_symbols: attribute}

        return my_symbols

    def finalize(self, expression, name=None):
        return expression


def attribute_lookup_information(attribute, **kwargs):
    if attribute == "kinetic energy":
        return (
            symbols(["mass", "momentum_x", "momentum_y", "momentum_z"]),
            (Symbol("momentum_x") ** 2 + Symbol("momentum_y") ** 2 + Symbol("momentum_z") ** 2) / (2 * Symbol("mass")),
        )
    if attribute == "momentum":
        return (
            symbols(["momentum_x", "momentum_y", "momentum_z"]),
            [Symbol("momentum_x"), Symbol("momentum_y"), Symbol("momentum_z")],
        )
    if attribute == "gamma":
        return (
            symbols(["mass", "momentum_x", "momentum_y", "momentum_z"]),
            sqrt(
                1
                + (Symbol("momentum_x") ** 2 + Symbol("momentum_y") ** 2 + Symbol("momentum_z") ** 2)
                / (Symbol("mass") ** 2 * c**2)
            ),
        )
    if attribute == "velocity":
        return (
            symbols(["mass", "momentum_x", "momentum_y", "momentum_z"]),
            [
                Symbol("momentum_x") / Symbol("mass"),
                Symbol("momentum_y") / Symbol("mass"),
                Symbol("momentum_z") / Symbol("mass"),
            ],
        )
    if attribute == "charge":
        return (symbols(["charge", "weighting"]), Symbol("charge") * Symbol("weighting"))

    return [Symbol(attribute)], Symbol(attribute)


class ParticleFromDataFrame(Particle):
    def __init__(self, df):
        self.df = df
        self.symbols = set()

    def get(self, attribute, **kwargs) -> Expr | Iterable[Expr]:
        my_symbols, expression = attribute_lookup_information(attribute, **kwargs)
        if missing := set(map(str, my_symbols)) - set(self.df.keys()):
            raise ValueError(f"Required information is missing from the given particle data frame: {missing=}")
        self.symbols |= set(my_symbols)
        return expression

    def finalize(self, expression, name=None):
        name = name or "result"
        result = lambdify(list(self.symbols), expression, "numpy")(*self.df[list(map(str, self.symbols))].T.to_numpy())
        return self.df.assign(**{name: result})


def make_particle(particle_like):
    if isinstance(particle_like, pd.DataFrame):
        return ParticleFromDataFrame(particle_like)
    return AbstractParticle()


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
        particle = AbstractParticle()
        functor_expression = self.functor(particle)
        return PyPIConGPUParticleFunctor(
            name=self.name,
            functor_expression=functor_expression,
            attribute_mapping=particle.get_attribute_map(),
            return_type=self.return_type,
        )

    def __call__(self, particle):
        expression = self.functor(particle)
        return particle.finalize(expression, self.name)
