"""
SPDX-FileCopyrightText: 2024-2024 PIConGPU contributors, Brian Edward Marre
SPDX-License-Identifier: GPL-3.0-or-later
"""

from ..groundstateionizationmodel import GroundStateIonizationModel
from .ionizationcurrent import IonizationCurrent


class FieldIonization(GroundStateIonizationModel):
    """common interface of all field ionization models"""

    ionization_current: IonizationCurrent | None
    """ionization current for energy conservation of field ionization"""
