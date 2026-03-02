"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Richard Pausch,
         Masoud Afshari
License: GPLv3+
"""

from picmistandard import PICMI_GaussianLaser
from pydantic import Field, computed_field, model_validator

from ...pypicongpu import laser, util
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser
from .polarization_type import PolarizationType


@default_converts_to(laser.GaussianLaser, conversions={"pulse_init": "pulse_init"})
class GaussianLaser(PICMI_GaussianLaser, BaseLaser):
    """
    PICMI object for Gaussian Laser.

    Standard Gaussian laser pulse parameters are:

    - wavelength : float
        Central wavelength of the laser [m].

    - waist : float
        Spot size (1/e^2 radius) of the laser at focus [m].

    - duration : float
        Full-width-half-maximum (FWHM) duration of the pulse [s].

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

    picongpu_polarization_type: PolarizationType = PolarizationType.LINEAR
    picongpu_laguerre_modes: list[float] = Field(default_factory=lambda: [1.0])
    picongpu_laguerre_phases: list[float] = Field(default_factory=lambda: [0.0])
    # make sure to always place Huygens-surface inside PML-boundaries,
    # default is valid for standard PMLs
    # @todo create check for insufficient dimension
    # @todo create check in simulation for conflict between PMLs and
    # Huygens-surfaces
    picongpu_huygens_surface_positions: list[list[int]] = [
        [16, -16],
        [16, -16],
        [16, -16],
    ]
    phi0: float = 0.0

    @computed_field
    def pulse_init(self) -> int:
        return self._compute_pulse_init()

    @model_validator(mode="after")
    def _validate(self):
        util.unsupported("laser name", self.name)
        util.unsupported("laser zeta", self.zeta)
        util.unsupported("laser beta", self.beta)
        util.unsupported("laser phi2", self.phi2)
        # unsupported: fill_in (do not warn, b/c we don't know if it has been
        # set explicitly, and always warning is bad)

        if len(self.picongpu_laguerre_modes) != len(self.picongpu_laguerre_phases):
            raise ValueError(
                "Your setup specifies a different number of Laguerre modes and phases. "
                "Please be explicit about both and use the same length. "
                f"You gave: {self.picongpu_laguerre_modes=} and {self.picongpu_laguerre_phases=}."
            )
        self._validate_common_properties()

        assert self._propagation_connects_centroid_and_focus(), (
            "propagation_direction must connect centroid_position and focus_position"
        )
        return self
