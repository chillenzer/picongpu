"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from functools import reduce
from inspect import _empty, signature
from operator import methodcaller, or_
from typing import Any, Callable, Iterable

from sympy import Expr, Symbol, symbols
from typeguard import typechecked

from picongpu.picmi.particle_functor.rng_arg import RNGArg
from picongpu.picmi.particle_functor.unit_dimension import UnitDimension
from picongpu.pypicongpu.particle_functor import ParticleFunctor as PyPIConGPUParticleFunctor
from picongpu.pypicongpu.particle_functor import UnitDimension as PyPIConGPUUnitDimension
from picongpu.pypicongpu.particle_functor import generate_preamble
from picongpu.pypicongpu.util import UnpackChain, alt, is_iterable

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
        ("CELL", ("xc", "yc", "zc")),
    )
    for precision in ("CELL", "SUB_CELL")
    for unit in ("CELL", "PIC", "SI")
}


class Particle:
    def get(self, attribute, **kwargs) -> Expr | Iterable[Expr]:
        NotImplementedError()

    def finalize(self, expression):
        return expression


class MacroParticle(Particle):
    needs_total_position = False

    def __init__(self):
        self.used_attributes = {}

    def get_attribute_map(self):
        return self.used_attributes

    def get(self, attribute, **kwargs) -> Expr | Iterable[Expr]:
        if attribute == "position":
            origin = kwargs.get("origin", "total")
            precision = kwargs.get("precision", "cell")
            unit = kwargs.get("unit", "cell")
            self.needs_total_position = self.needs_total_position or (origin.lower() not in ["cell", "local"])
            my_symbols = _COORDINATE_SYSTEM[(origin, precision, unit)]
            self.used_attributes |= {my_symbols: ("position", origin, precision, unit)}

        elif attribute == "momentum":
            my_symbols = symbols("px,py,pz")
            self.used_attributes |= {my_symbols: "momentum"}

        elif attribute == "momentumPrev1":
            my_symbols = symbols("p1x,p1y,p1z")
            self.used_attributes |= {my_symbols: "momentumPrev1"}

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


_SCALING = {Symbol("mass"): 1, Symbol("Ekin"): 1, Symbol("charge"): 1}


class PhysicalParticle(MacroParticle):
    def __init__(self, scales_with_weighting=None):
        self.scales_with_weighting = scales_with_weighting
        super().__init__()

    def get(self, *args, **kwargs):
        my_symbols = super().get(*args, **kwargs)
        if self.scales_with_weighting is None:
            w = super().get("weighting")
            rescaled = tuple(s * (w ** (-_SCALING[s])) for s in alt(lambda: iter(my_symbols), [my_symbols]))
            my_symbols = rescaled if is_iterable(my_symbols) else rescaled[0]
        return my_symbols

    def finalize(self, expression):
        if self.scales_with_weighting is not None:
            expression *= super().get("weighting") ** (-self.scales_with_weighting)
        return expression


@typechecked
class ParticleFunctor:
    def __new__(cls, functor=None, **kwargs):
        if functor is None:
            return lambda f: ParticleFunctor(functor=f, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        functor: Callable[[Particle], Any] | Callable[[Particle, RNGArg], Any],
        *,
        name: str | None = None,
        return_type: type | str | None = None,
        unit_dimension: UnitDimension | None = None,
        scales_with_weighting: int | None = None,
    ):
        self.functor = functor
        self.name = name or functor.__name__
        self.return_type = return_type or (
            v if not issubclass(v := signature(functor).return_annotation, _empty) else float
        )
        self.unit_dimension = unit_dimension or UnitDimension()
        self.scales_with_weighting = scales_with_weighting
        self.argument_types = [
            cls
            for cls in UnpackChain(signature(self.functor)).parameters.values().annotation
            if any(issubclass(cls, expected) for expected in [Particle, RNGArg])
        ]
        if not issubclass(self.argument_types[0], Particle):
            raise ValueError(
                f"ParticleFunctor takes exactly one particle as first argument. You have requested {self.argument_types=} in your signature."
            )
        if issubclass(particle_type := self.argument_types[0], PhysicalParticle):
            self.argument_types[0] = lambda: particle_type(self.scales_with_weighting)
        else:
            if self.scales_with_weighting is not None:
                raise TypeError(f"Can't apply scaling to {particle_type=}. You gave: {self.scales_with_weighting=}.")
        if self.argument_types.count(RNGArg) > 1:
            raise ValueError(
                f"ParticleFunctor can take at most one RNG. You have requested {self.argument_types=} in your signature."
            )

    def get_as_pypicongpu(self, mode) -> PyPIConGPUParticleFunctor:
        args = [cls() for cls in self.argument_types]
        return PyPIConGPUParticleFunctor(
            name=self.name,
            functor_expression=self(*args),
            functor_preamble=generate_preamble(reduce(or_, map(methodcaller("get_attribute_map"), args)), mode=mode),
            return_type=self.return_type,
            unit_dimension=PyPIConGPUUnitDimension(unit_dimension=self.unit_dimension.unit_vector.tolist()),
            needs_total_position=args[0].needs_total_position,
            rng_info=alt(lambda: args[1].model_dump(mode="python"), None),
        )

    def __call__(self, particle, *args):
        return particle.finalize(self.functor(particle, *args))
