"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Richard Pausch
License: GPLv3+
"""

import math
import typing

import picmistandard
import typeguard

from ...pypicongpu import laser, util
from .base_laser import BaseLaser
from .polarization_type import PolarizationType


@typeguard.typechecked
class GaussianLaser(picmistandard.PICMI_GaussianLaser, BaseLaser):
    """PICMI object for Gaussian Laser"""

    def __init__(
        self,
        wavelength,
        waist,
        duration,
        propagation_direction,
        polarization_direction,
        focal_position,
        centroid_position,
        a0=None,
        E0=None,
        picongpu_polarization_type=(PolarizationType.LINEAR),
        picongpu_laguerre_modes: typing.Optional[typing.List[float]] = None,
        picongpu_laguerre_phases: typing.Optional[typing.List[float]] = None,
        # make sure to always place Huygens-surface inside PML-boundaries,
        # default is valid for standard PMLs
        # @todo create check for insufficient dimension
        # @todo create check in simulation for conflict between PMLs and
        # Huygens-surfaces
        picongpu_huygens_surface_positions: typing.List[typing.List[int]] = [
            [16, -16],
            [16, -16],
            [16, -16],
        ],
        **kw,
    ):
        if waist <= 0:
            raise ValueError(f"waist must be > 0. You gave {waist=}.")
        if wavelength <= 0:
            raise ValueError(f"wavelength must be > 0. You gave {wavelength=}.")
        if duration <= 0:
            raise ValueError(f"laser pulse duration must be > 0. You gave {duration=}.")

        assert (picongpu_laguerre_modes is None and picongpu_laguerre_phases is None) or (
            picongpu_laguerre_modes is not None and picongpu_laguerre_phases is not None
        ), (
            "laguerre_modes and laguerre_phases MUST BE both set or both \
            unset"
        )

        self.picongpu_polarization_type = picongpu_polarization_type
        self.picongpu_laguerre_modes = picongpu_laguerre_modes
        self.picongpu_laguerre_phases = picongpu_laguerre_phases
        self.picongpu_huygens_surface_positions = picongpu_huygens_surface_positions

        # Calculate a0 and E0 using our base laser, as the PICMI standard does not provide consistency checks.
        self.k0 = 2.0 * math.pi / wavelength
        self.a0, self.E0 = self._compute_E0_and_a0(self.k0, E0, a0)
        kw["E0"] = self.E0
        kw["a0"] = self.a0

        super().__init__(
            wavelength,
            waist,
            duration,
            propagation_direction,
            polarization_direction,
            focal_position,
            centroid_position,
            **kw,
        )

        self.phi0 = self.phi0 or 0.0

    def get_as_pypicongpu(self) -> laser.GaussianLaser:
        util.unsupported("laser name", self.name)
        util.unsupported("laser zeta", self.zeta)
        util.unsupported("laser beta", self.beta)
        util.unsupported("laser phi2", self.phi2)
        # unsupported: fill_in (do not warn, b/c we don't know if it has been
        # set explicitly, and always warning is bad)

        self._validate_common_properties()
        assert self._propagation_connects_centroid_and_focus(), (
            "propagation_direction must connect centroid_position and focus_position"
        )

        pypicongpu_laser = laser.GaussianLaser()
        pypicongpu_laser.wavelength = self.wavelength
        pypicongpu_laser.waist = self.waist
        pypicongpu_laser.duration = float(self.duration)
        pypicongpu_laser.focus_pos = list(self.focal_position)
        pypicongpu_laser.phase = self.phi0
        pypicongpu_laser.E0 = self.E0
        pypicongpu_laser.pulse_init = self._compute_pulse_init()
        pypicongpu_laser.polarization_type = self.picongpu_polarization_type.get_as_pypicongpu()
        pypicongpu_laser.polarization_direction = self.polarization_direction

        pypicongpu_laser.propagation_direction = self.propagation_direction

        if self.picongpu_laguerre_modes is None:
            pypicongpu_laser.laguerre_modes = [1.0]
        else:
            pypicongpu_laser.laguerre_modes = self.picongpu_laguerre_modes

        if self.picongpu_laguerre_phases is None:
            pypicongpu_laser.laguerre_phases = [0.0]
        else:
            pypicongpu_laser.laguerre_phases = self.picongpu_laguerre_phases

        pypicongpu_laser.huygens_surface_positions = self.picongpu_huygens_surface_positions

        return pypicongpu_laser
