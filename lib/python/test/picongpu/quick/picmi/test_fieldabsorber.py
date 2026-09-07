"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: PIConGPU AI Agent (TT-19)
License: GPLv3+
"""

import tempfile
from pathlib import Path

import pytest

from picongpu import core, picmi, pypicongpu


def _grid(**kwargs) -> picmi.Cartesian3DGrid:
    base = dict(
        number_of_cells=[192, 2048, 12],
        lower_bound=[0, 0, 0],
        upper_bound=[3.40992e-5, 9.07264e-5, 2.1312e-6],
        lower_boundary_conditions=["open", "open", "periodic"],
        upper_boundary_conditions=["open", "open", "periodic"],
    )
    base.update(kwargs)
    return picmi.Cartesian3DGrid(**base)


def _sim(grid: picmi.Cartesian3DGrid) -> picmi.Simulation:
    solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
    return picmi.Simulation(time_step_size=1.39e-16, max_steps=32, solver=solver)


def test_default_keeps_cpp_defaults():
    """without configuration the pypicongpu default absorber (== static C++ file) is used"""
    absorber = _sim(_grid()).get_as_pypicongpu().field_absorber
    assert absorber.kind == "pml"
    assert absorber.thickness == ((12, 12), (12, 12), (12, 12))
    assert absorber.strength == ((1e-3, 1e-3), (1e-3, 1e-3), (1e-3, 1e-3))


def test_pml_cells_symmetric():
    """standard pml_cells maps to the per-axis symmetric NUM_CELLS"""
    absorber = _sim(_grid(pml_cells=[12, 12, 12])).get_as_pypicongpu().field_absorber
    assert absorber.thickness == ((12, 12), (12, 12), (12, 12))


def test_pml_cells_disables_axis_with_zero():
    """pml_cells=0 on an axis disables absorption there (thickness 0)"""
    absorber = _sim(_grid(pml_cells=[13, 0, 11])).get_as_pypicongpu().field_absorber
    assert absorber.thickness == ((13, 13), (0, 0), (11, 11))


def test_pml_cells_per_axis_values():
    absorber = _sim(_grid(pml_cells=[13, 12, 11])).get_as_pypicongpu().field_absorber
    assert absorber.thickness == ((13, 13), (12, 12), (11, 11))


def test_field_absorber_asymmetric():
    """the fully faithful picongpu_field_absorber carries per-boundary thickness and kind"""
    field_absorber = pypicongpu.fieldabsorber.FieldAbsorber(kind="exponential", thickness=((13, 0), (4, 12), (32, 32)))
    absorber = _sim(_grid(picongpu_field_absorber=field_absorber)).get_as_pypicongpu().field_absorber
    assert absorber.kind == "exponential"
    assert absorber.thickness == ((13, 0), (4, 12), (32, 32))


def test_field_absorber_conflicts_with_pml_cells():
    """providing both sources of truth for the same number is rejected (neither wins)"""
    field_absorber = pypicongpu.fieldabsorber.FieldAbsorber(kind="exponential")
    with pytest.raises(ValueError, match=".*only one of them.*"):
        _sim(_grid(pml_cells=[12, 12, 12], picongpu_field_absorber=field_absorber)).get_as_pypicongpu()


def test_invalid_pml_cells():
    with pytest.raises(ValueError, match=".*pml_cells must be a list of 3.*"):
        _sim(_grid(pml_cells=[12, 12])).get_as_pypicongpu()
    with pytest.raises(ValueError, match=".*pml_cells must be a list of 3.*"):
        _sim(_grid(pml_cells=[12, -2, 12])).get_as_pypicongpu()


def test_default_write_input_file_byte_equal():
    """full setup generation renders fieldAbsorber.param byte-equal to the static C++ file"""
    static = (core.path("include") / "picongpu/param/fieldAbsorber.param").read_bytes()
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "setup"
        _sim(_grid()).write_input_file(outdir)
        rendered = (outdir / "include/picongpu/param/fieldAbsorber.param").read_bytes()
        assert rendered == static
        cfg = (outdir / "etc/picongpu/N.cfg").read_text()
        assert 'TBG_fieldAbsorber="--fieldAbsorber pml"' in cfg


def test_exponential_kind_wired_into_cfg():
    """kind selection ends up in the --fieldAbsorber command line option"""
    field_absorber = pypicongpu.fieldabsorber.FieldAbsorber(kind="exponential")
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "setup"
        _sim(_grid(picongpu_field_absorber=field_absorber)).write_input_file(outdir)
        cfg = (outdir / "etc/picongpu/N.cfg").read_text()
        assert 'TBG_fieldAbsorber="--fieldAbsorber exponential"' in cfg
        rendered = (outdir / "include/picongpu/param/fieldAbsorber.param").read_text()
        assert "namespace exponential" in rendered
        assert "constexpr float_X STRENGTH[3][2]" in rendered
