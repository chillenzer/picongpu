"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Richard Pausch,
         Masoud Afshari
License: GPLv3+
"""

import math
import typing

import numpy as np
import picmistandard
import typeguard

from scipy.spatial.transform import Rotation

from ...pypicongpu import laser, util
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser
from .polarization_type import PolarizationType


@default_converts_to(laser.GaussianLaser)
@typeguard.typechecked
class GaussianLaser(picmistandard.PICMI_GaussianLaser, BaseLaser):
    """
    PICMI object for Gaussian Laser.

    Standard Gaussian laser pulse parameters are:

    - wavelength : float
        Central wavelength of the laser [m].

    - waist : float
        Spot size (1/e^2 radius of the intensity) of the laser at focus [m].

    - duration : float
        Full-width-half-maximum (FWHM of intensity) duration of the pulse [s].

    - propagation_direction : list[float]
        Normalized vector of propagation direction.

    - polarization_direction : list[float]
        Normalized vector of polarization direction.

    - focal_position : list[float]
        3D coordinates of the laser focus [m].

    - centroid_position : list[float]
        3D coordinates of the laser centroid [m].

    - a0 : float, optional
        Normalized vector potential (dimensionless).

    - E0 : float, optional
        Peak electric field amplitude [V/m].

    - picongpu_polarization_type: Polarization type in PIConGPU (LINEAR or CIRCULAR)

    - picongpu_laguerre_modes: Optional magnitudes of Laguerre modes (only relevant for structured beams)

    - picongpu_laguerre_phases: Optional phases of Laguerre modes (only relevant for structured beams)

    - picongpu_huygens_surface_positions : list[list[int]], default=[[16, -16],[16, -16],[16, -16]]
        Positions of the Huygens surface inside the PML. Each entry is a
        pair [min, max] indices along x, y, z.

    - phi0 : float, optional
    Initial phase offset [rad].

    Notes:
    - Exactly one of ``a0`` or ``E0`` must be provided, the other is
      calculated automatically.
    """

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
        picongpu_polarization_type=PolarizationType.LINEAR,
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
        self.picongpu_laguerre_modes = picongpu_laguerre_modes or [1.0]
        self.picongpu_laguerre_phases = picongpu_laguerre_phases or [0.0]
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
        self.check()
        self.pulse_init = self._compute_pulse_init()

    def _Omega0(self):
        from picongpu.picmi.constants import c

        return 2 * np.pi * c / self.wavelength

    def _inverse_curvature_radius(self, z, w0):
        from picongpu.picmi.constants import c

        return z / (z**2 + (0.5 * self._Omega0() * w0**2 / c) ** 2)

    def check(self):
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

    def _complex_amplitude_standard_conditions(self, x, y, z, t):
        """
        This is
        - focus is in the origin at t=0
        - propagation in z direction
        - polarization in x direction
        """

        from picongpu.picmi.constants import c

        w0x = self.waist
        w0y = self.waist
        tau0 = self.duration / np.sqrt(2 * np.log(2))
        zRx = 0.5 * self._Omega0() * w0x**2 / c
        zRy = 0.5 * self._Omega0() * w0y**2 / c
        Rx_inv = self._inverse_curvature_radius(z, w0x)
        Ry_inv = self._inverse_curvature_radius(z, w0y)
        wx = w0x * np.sqrt(1 + z**2 / zRx**2)
        wy = w0y * np.sqrt(1 + z**2 / zRy**2)
        gamma4 = (t - z / c - 0.5 / c * (x**2 * Rx_inv + y**2 * Ry_inv)) / tau0

        return (
            self.E0
            / (tau0 * np.sqrt(np.pi))
            * (1 + z**2 / zRx**2) ** (-1 / 4)
            * (1 + z**2 / zRy**2) ** (-1 / 4)
            * np.exp(1.0j * self._Omega0() * gamma4 * tau0)
            * np.exp(0.5j * (np.arctan(z / zRx) + np.arctan(z / zRy)))
            * np.exp(-(x**2 / wx**2 + y**2 / wy**2))
            * np.exp(-(gamma4**2))
        )

    def _polarization_vector_standard_conditions_at(self, x, y, z, t):
        """
        This is
        - focus is in the origin at t=0
        - propagation in z direction
        - polarization in x direction
        """
        if self.picongpu_polarization_type == PolarizationType.LINEAR:
            w0x = self.waist
            w0y = self.waist
            angle_x = np.arcsin(x * self._inverse_curvature_radius(z, w0x))
            angle_y = np.arcsin(y * self._inverse_curvature_radius(z, w0y))
            r = Rotation.from_euler("xy", np.moveaxis(np.asarray([angle_x, angle_y]), 0, -1))
            return np.moveaxis(r.apply(self.polarization_direction), -1, 0)
        elif self.picongpu_polarization_type == PolarizationType.CIRCULAR:
            raise NotImplementedError("Circular polarization is not yet implemented.")
        else:
            raise ValueError("Unknown {self.picongpu_polarization_type=}.")

    def polarization_vector_at(self, x, y, z, t=0.0):
        return self._polarization_vector_standard_conditions_at(*self._to_standard_coordinates(x, y, z, t))

    def _standard_rotation(self):
        return Rotation.align_vectors(
            [[1, 0, 0], [0, 0, 1]], [self.polarization_direction, self.propagation_direction]
        )[0].apply

    def _to_standard_coordinates(self, x, y, z, t):
        from picongpu.picmi.constants import c

        coords = np.asarray([x, y, z]) - as_broadcastable_factor_in_front_of(self.focal_position, of=(x, y, z))
        r = self._standard_rotation()
        x, y, z = np.moveaxis(r(np.moveaxis(coords, 0, -1)), -1, 0)
        t -= np.dot(np.asarray(self.focal_position) - self.centroid_position, self.propagation_direction) / c
        return x, y, z, t

    def complex_amplitude(self, x, y, z, t=0.0):
        return self._complex_amplitude_standard_conditions(*self._to_standard_coordinates(x, y, z, t))

    def E(self, x, y, z, t=0.0):
        return (
            self.polarization_vector_at(x, y, z, t)
            * np.real(self.complex_amplitude(x, y, z, t)).astype(float)[np.newaxis, ...]
        )

    def Ex(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[0]

    def Ey(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[1]

    def Ez(self, x, y, z, t=0.0):
        return self.E(x, y, z, t)[2]

    def envelope(self, x, y, z, t=0.0):
        return np.abs(self.complex_amplitude(x, y, z, t))


def as_broadcastable_factor_in_front_of(arr, of=tuple()):
    if of:
        return np.reshape(arr, (-1, *(1 for _ in np.broadcast_shapes(*map(np.shape, of)))))
    return np.asarray(arr)
