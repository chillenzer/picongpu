"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Julian Lenz
License: GPLv3+
"""

from .rendering import SelfRegisteringRenderedObject
from . import util

import enum
import typing
import typeguard
import logging


class PolarizationType(enum.Enum):
    """represents a polarization of a laser (for PIConGPU)"""

    LINEAR = 1
    CIRCULAR = 2

    def get_cpp_str(self) -> str:
        """retrieve name as used in c++ param files"""
        cpp_by_ptype = {
            PolarizationType.LINEAR: "Linear",
            PolarizationType.CIRCULAR: "Circular",
        }
        return cpp_by_ptype[self]


def _get_huygens_surface_serialized(huygens_surface_positions) -> dict:
    """Serialize huygens surface positions for all laser types"""
    return {
        "row_x": {
            "negative": huygens_surface_positions[0][0],
            "positive": huygens_surface_positions[0][1],
        },
        "row_y": {
            "negative": huygens_surface_positions[1][0],
            "positive": huygens_surface_positions[1][1],
        },
        "row_z": {
            "negative": huygens_surface_positions[2][0],
            "positive": huygens_surface_positions[2][1],
        },
    }


class Laser(SelfRegisteringRenderedObject):
    pass


class _BaseLaser(Laser):
    """Base class for all laser types with common properties and serialization logic"""

    # Common properties for all lasers
    propagation_direction = util.build_typesafe_property(typing.List[float])
    """propagation direction (normalized vector)"""
    polarization_direction = util.build_typesafe_property(typing.List[float])
    """direction of polarization (normalized vector)"""
    polarization_type = util.build_typesafe_property(PolarizationType)
    """laser polarization"""
    wavelength = util.build_typesafe_property(float)
    """wave length in m"""
    duration = util.build_typesafe_property(float)
    """duration in s (1 sigma)"""
    focus_pos = util.build_typesafe_property(typing.List[float])
    """focus position vector in m"""
    phase = util.build_typesafe_property(float)
    """phase in rad, periodic in 2*pi"""
    E0 = util.build_typesafe_property(float)
    """E0 in V/m"""
    pulse_init = util.build_typesafe_property(float)
    """laser will be initialized pulse_init times of duration (unitless)"""

    # Huygens surface position (common to all lasers)
    huygens_surface_positions = util.build_typesafe_property(typing.List[typing.List[int]])
    """Position in cells of the Huygens surface relative to start/
       edge(negative numbers) of the total domain"""

    def _get_common_serialized_fields(self) -> dict:
        """Get all common serialized fields for lasers"""
        return {
            "wave_length_si": float(self.wavelength),
            "pulse_duration_si": float(self.duration),
            "focus_pos_si": list(map(lambda x: {"component": float(x)}, self.focus_pos)),
            "phase": self.phase,
            "E0_si": self.E0,
            "pulse_init": float(self.pulse_init),
            "propagation_direction": list(map(lambda x: {"component": float(x)}, self.propagation_direction)),
            "polarization_type": self.polarization_type.get_cpp_str(),
            "polarization_direction": list(map(lambda x: {"component": float(x)}, self.polarization_direction)),
            "huygens_surface_positions": _get_huygens_surface_serialized(self.huygens_surface_positions),
        }


@typeguard.typechecked
class GaussianLaser(_BaseLaser):
    """
    PIConGPU Gaussian Laser

    Holds Parameters to specify a gaussian laser
    """

    _name = "gaussian"

    waist = util.build_typesafe_property(float)
    """beam waist in m"""
    laguerre_modes = util.build_typesafe_property(typing.List[float])
    """array containing the magnitudes of radial Laguerre-modes"""
    laguerre_phases = util.build_typesafe_property(typing.List[float])
    """array containing the phases of radial Laguerre-modes"""

    def _get_serialized(self) -> dict:
        if [] == self.laguerre_modes:
            raise ValueError("Laguerre modes MUST NOT be empty.")
        if [] == self.laguerre_phases:
            raise ValueError("Laguerre phases MUST NOT be empty.")
        if len(self.laguerre_phases) != len(self.laguerre_modes):
            raise ValueError("Laguerre modes and Laguerre phases MUST BE arrays of equal length.")
        if len(list(filter(lambda x: x < 0, self.laguerre_modes))) > 0:
            logging.warning("Laguerre mode magnitudes SHOULD BE positive definite.")

        # Build on the common fields
        return self._get_common_serialized_fields() | {
            "waist_si": float(self.waist),
            "laguerre_modes": list(map(lambda x: {"single_laguerre_mode": float(x)}, self.laguerre_modes)),
            "laguerre_phases": list(map(lambda x: {"single_laguerre_phase": float(x)}, self.laguerre_phases)),
            "modenumber": len(self.laguerre_modes) - 1,
        }


@typeguard.typechecked
class PlaneWaveLaser(_BaseLaser):
    """
    PIConGPU Plane Wave Laser

    Holds Parameters to specify a plane wave laser
    """

    _name = "planewave"

    laser_nofocus_constant_si = util.build_typesafe_property(float)
    """constant for plane wave laser without focus (unitless)"""

    def _get_serialized(self) -> dict:
        return self._get_common_serialized_fields() | {
            "laser_nofocus_constant_si": float(self.laser_nofocus_constant_si),
        }


@typeguard.typechecked
class DispersivePulseLaser(_BaseLaser):
    """
    PIConGPU Dispersive Pulse Laser

    Holds Parameters to specify a dispersive Gaussian laser pulse with dispersion parameters
    """

    _name = "dispersive"

    waist = util.build_typesafe_property(float)
    """beam waist in m"""
    spectral_support = util.build_typesafe_property(float)
    """width of the spectral support for the discrete Fourier transform [none]"""
    sd_si = util.build_typesafe_property(float)
    """spatial dispersion in focus [m*s]"""
    ad_si = util.build_typesafe_property(float)
    """angular dispersion in focus [rad*s]"""
    gdd_si = util.build_typesafe_property(float)
    """group velocity dispersion in focus [s^2]"""
    tod_si = util.build_typesafe_property(float)
    """third order dispersion in focus [s^3]"""

    def _get_serialized(self) -> dict:
        return self._get_common_serialized_fields() | {
            "waist_si": self.waist,
            "spectral_support": self.spectral_support,
            "sd_si": self.sd_si,
            "ad_si": self.ad_si,
            "gdd_si": self.gdd_si,
            "tod_si": self.tod_si,
        }


@typeguard.typechecked
class FromOpenPMDPulseLaser(Laser):
    """
    PIConGPU FromOpenPMDPulseLaser

    Holds Parameters to specify a laser pulse from an OpenPMD file
    """

    _name = "fromOpenPMDPulse"

    propagation_direction = util.build_typesafe_property(typing.List[float])
    """propagation direction (normalized vector)"""
    polarization_direction = util.build_typesafe_property(typing.List[float])
    """direction of polarization (normalized vector)"""
    file_path = util.build_typesafe_property(str)
    """File path to the OpenPMD file containing the pulse data"""
    iteration = util.build_typesafe_property(int)
    """Iteration in the OpenPMD file to use"""
    dataset_name = util.build_typesafe_property(str)
    """Name of the dataset in the OpenPMD file containing the pulse data"""
    datatype = util.build_typesafe_property(str)
    """Data type of the pulse data"""
    time_offset_si = util.build_typesafe_property(float)
    """Time offset in seconds to apply to the pulse data [s]"""
    polarisationAxisOpenPMD = util.build_typesafe_property(str)
    """Polarization axis name in the OpenPMD file"""
    propagationAxisOpenPMD = util.build_typesafe_property(str)
    """Propagation axis name in the OpenPMD file"""
    huygens_surface_positions = util.build_typesafe_property(typing.List[typing.List[int]])
    """Position in cells of the Huygens surface relative to start/
       edge(negative numbers) of the total domain"""

    def _get_serialized(self) -> dict:
        return {
            "propagation_direction": list(map(lambda x: {"component": float(x)}, self.propagation_direction)),
            "polarization_direction": list(map(lambda x: {"component": float(x)}, self.polarization_direction)),
            "file_path": self.file_path,
            "iteration": self.iteration,
            "dataset_name": self.dataset_name,
            "datatype": self.datatype,
            "time_offset_si": self.time_offset_si,
            "polarisationAxisOpenPMD": self.polarisationAxisOpenPMD,
            "propagationAxisOpenPMD": self.propagationAxisOpenPMD,
            "huygens_surface_positions": _get_huygens_surface_serialized(self.huygens_surface_positions),
        }
