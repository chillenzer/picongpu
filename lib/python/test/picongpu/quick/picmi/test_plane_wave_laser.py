"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import os
import re
import tempfile
from unittest import TestCase

import pytest
from picongpu import picmi
from pydantic import ValidationError


class TestPicmiPlaneWaveLaser(TestCase):
    def _make_laser(self, **kwargs):
        args = {
            "wavelength": 800e-9,
            "duration": 30e-15,
            "propagation_direction": [0, 1, 0],
            "polarization_direction": [1, 0, 0],
            "centroid_position": [0.0, -5e-6, 0.0],
            "a0": 1.0,
        }
        args.update(kwargs)
        return picmi.PlaneWaveLaser(**args)

    def test_basic(self):
        """PlaneWaveLaser accepts no focus-position argument and converts successfully"""
        picmi_laser = self._make_laser()
        pypic_laser = picmi_laser.get_as_pypicongpu()
        dumped = pypic_laser.model_dump()
        assert dumped["type_planewave"] is True
        assert dumped["laser_nofocus_constant_si"] == 0.0
        assert all(component.component == 0.0 for component in pypic_laser.focus_pos_si)

    def test_model_dump_is_not_a_roundtrip_constructor(self):
        """model_dump() output is serialization-only; it is not valid PlaneWaveLaser constructor input"""
        picmi_laser = self._make_laser()
        with pytest.raises(ValidationError):
            picmi.PlaneWaveLaser(**picmi_laser.model_dump())

    def test_focus_position_not_user_facing(self):
        """The focus position must not be a user-facing constructor argument"""
        with pytest.raises(ValidationError, match="focus position"):
            self._make_laser(focal_position=[1, 2, 3])
        with pytest.raises(ValidationError, match="focus position"):
            self._make_laser(focus_pos=[1, 2, 3])
        assert "focal_position" not in picmi.PlaneWaveLaser.model_fields
        assert "focus_pos" not in picmi.PlaneWaveLaser.model_fields
        assert "focal_position" not in picmi.PlaneWaveLaser.model_json_schema().get("properties", {})
        assert "focus_pos" not in picmi.PlaneWaveLaser.model_json_schema().get("properties", {})

    def test_renders_incident_field(self):
        """PlaneWaveLaser renders incidentField.param with LASER_NOFOCUS_CONSTANT_SI and a focal field"""
        laser = self._make_laser()
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[128, 512, 256],
            lower_bound=[0, 0, 0],
            upper_bound=[17, 192, 42],
            lower_boundary_conditions=["periodic", "periodic", "open"],
            upper_boundary_conditions=["periodic", "periodic", "open"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(time_step_size=1, max_steps=2, solver=solver)
        sim.add_laser(laser, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "input")
            sim.write_input_file(output_dir)
            rendered_path = os.path.join(output_dir, "include", "picongpu", "param", "incidentField.param")
            with open(rendered_path) as rendered_file:
                rendered = rendered_file.read()

        assert re.search(r"PyPIConGPUPlaneWaveParam", rendered) is not None, "PlaneWave profile not rendered"
        match = re.search(r"LASER_NOFOCUS_CONSTANT_SI = ([0-9eE.+-]+);", rendered)
        assert match is not None, "LASER_NOFOCUS_CONSTANT_SI not found in rendered incidentField.param"
        assert float(match.group(1)) == 0.0
        assert re.search(r"FOCUS_POSITION_X_SI = focus_position\[0\];", rendered) is not None
        assert re.search(r"FOCUS_POSITION_Y_SI = focus_position\[1\];", rendered) is not None
        assert re.search(r"FOCUS_POSITION_Z_SI = focus_position\[2\];", rendered) is not None
