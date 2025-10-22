"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import numbers

from pydantic import BaseModel

from ..rendering.pmaccprinter import PMAccPrinter
from ..rendering.renderedobject import RenderedObject


def by_bracket(attribute):
    return f"particle[{attribute}_]"


ACCESSORS = {
    (
        "position",
        origin.lower(),
        precision.lower(),
        unit.lower(),
    ): f"getParticlePosition<DomainOrigin::{origin}, PositionPrecision::{precision}, PositionUnits::{unit}>(domainInfo, particle)"
    for origin in ("TOTAL", "GLOBAL", "LOCAL", "MOVING_WINDOW", "LOCAL_WITH_GUARDS")
    for precision in ("CELL", "SUB_CELL")
    for unit in ("CELL", "PIC", "SI")
} | {
    "mass": "picongpu::traits::attribute::getMass(particle[weighting_], particle)",
    # CAUTION: The names in the gamma formula are currently hardcoded.
    # We'll certainly trip over this, should we ever dare to change the internal names.
    "gamma": "picongpu::Gamma()(momentum::type{px, py, pz}, mass)",
    "kinetic energy": "picongpu::KinEnergy()(momentum::type{px, py, pz}, mass)",
    "velocity": "picongpu::Velocity()(momentum::type{px, py, pz}, mass)",
    "charge": "picongpu::traits::attribute::getCharge(particle[weighting_], particle)",
    "charge_state": "picongpu::traits::attribute::getChargeState(particle)",
    "damped_weighting": "picongpu::traits::attribute::getDampedWeighting(particle)",
    "timestep": "domainInfo.currentStep",
    "timestep_size": "sim.pic.getDt()",
}


def symbol_to_string(symbol):
    return str(symbol) if not isinstance(symbol, tuple) else "[" + ",".join(map(str, symbol)) + "]"


def generate_preamble(attribute_mapping):
    return [
        {"statement": f"auto const {symbol_to_string(symbol)} = {ACCESSORS.get(attribute, by_bracket(attribute))};"}
        for symbol, attribute in attribute_mapping.items()
    ]


def translate_to_cpp_type(return_type):
    try:
        if issubclass(return_type, numbers.Integral):
            return "int"
        if issubclass(return_type, numbers.Real):
            return "float_X"
        if issubclass(return_type, bool):
            return "bool"
    except TypeError:
        pass
    if isinstance(return_type, str):
        return return_type
    raise ValueError(f"Cannot translate {return_type=} to a C++ type.")


class ParticleFunctor(RenderedObject, BaseModel):
    name: str
    functor_expression: str
    functor_preamble: list[dict[str, str]]
    return_type: str

    def __init__(self, name, functor_expression, attribute_mapping, return_type):
        name = name
        functor_expression = PMAccPrinter().doprint(functor_expression)
        functor_preamble = generate_preamble(attribute_mapping)
        return_type = translate_to_cpp_type(return_type)
        super().__init__(
            name=name, functor_expression=functor_expression, functor_preamble=functor_preamble, return_type=return_type
        )

    def _get_serialized(self):
        return self.model_dump(mode="json")
