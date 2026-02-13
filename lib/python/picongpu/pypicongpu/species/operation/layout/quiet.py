"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from functools import partial
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, PlainSerializer, PrivateAttr

from .layout import Layout


def serialise_vec(value) -> dict:
    return dict(zip("xyz", value))


def broadcast_validation(values, condition, message="Condition not met."):
    if not all(condition(value) for value in values):
        raise ValueError(f"{message} You gave: {values}.")
    return values


Vec3_int = Annotated[
    tuple[int, int, int],
    PlainSerializer(serialise_vec),
    AfterValidator(
        partial(
            broadcast_validation,
            condition=lambda v: v >= 0 and v < 1,
            message="All of in_cell_offset must be between 0 and 1.",
        )
    ),
]


class Quiet(Layout, BaseModel):
    _name: str = PrivateAttr("quiet")
    n_points: Vec3_int = Field(default=(0, 0, 0))
    ppc: int = Field(gt=0)
    """particles per cell, >0"""
