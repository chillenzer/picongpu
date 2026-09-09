"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: PIConGPU AI Agent (TT-78)
License: GPLv3+
"""

import tempfile
from pathlib import Path

import pytest

from picongpu import picmi, pypicongpu


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


def _sim(**kwargs) -> picmi.Simulation:
    solver = picmi.ElectromagneticSolver(method="Yee", grid=_grid())
    sim = picmi.Simulation(time_step_size=1.39e-16, max_steps=32, solver=solver, **kwargs)
    return sim


def test_defaults():
    """constructing without arguments yields the C++ default parameters of the solver"""
    solver = picmi.ElectrostaticSolver()
    assert solver.method == "BICGStab"
    assert solver.required_precision == 1e-8
    assert solver.maximum_iterations == 2000
    assert solver.preconditioner == "default"
    assert solver.preconditioner_maximum_iterations == 20


def test_unsupported_method_rejected():
    """only the actual solver method (BiCGStab) is supported"""
    with pytest.raises(AssertionError):
        picmi.ElectrostaticSolver(method="FFT")
    with pytest.raises(AssertionError):
        picmi.ElectrostaticSolver(method="Multigrid")


def test_custom_parameters_round_trip():
    """custom parameters are forwarded to the pypicongpu PoissonSolver"""
    solver = picmi.ElectrostaticSolver(
        required_precision=1e-6,
        maximum_iterations=100,
        preconditioner="none",
        preconditioner_maximum_iterations=5,
    )
    poisson = solver.get_as_pypicongpu()
    assert isinstance(poisson, pypicongpu.PoissonSolver)
    assert poisson.tolerance == 1e-6
    assert poisson.max_steps == 100
    assert poisson.preconditioner == "none"
    assert poisson.preconditioner_max_steps == 5
    assert poisson.preconditioner_disabled is True


def test_defaults_round_trip():
    """an unconfigured ElectrostaticSolver is equivalent to the pypicongpu defaults"""
    assert picmi.ElectrostaticSolver().get_as_pypicongpu() == pypicongpu.PoissonSolver()


def test_pypicongpu_defaults():
    """the pypicongpu PoissonSolver is fully optional (cf. upstream review finding)"""
    poisson = pypicongpu.PoissonSolver()
    assert poisson.max_steps == 2000
    assert poisson.tolerance == 1e-8
    assert poisson.preconditioner == "default"
    assert poisson.preconditioner_max_steps == 20
    assert poisson.preconditioner_disabled is False


def test_pypicongpu_preconditioner_disabled_computed():
    """preconditioner_disabled mirrors the preconditioner choice"""
    assert pypicongpu.PoissonSolver(preconditioner="default").preconditioner_disabled is False
    assert pypicongpu.PoissonSolver(preconditioner="none").preconditioner_disabled is True


def test_pypicongpu_validation():
    """solver parameters must be positive"""
    with pytest.raises(Exception):
        pypicongpu.PoissonSolver(max_steps=0)
    with pytest.raises(Exception):
        pypicongpu.PoissonSolver(tolerance=-1)
    with pytest.raises(Exception):
        pypicongpu.PoissonSolver(preconditioner_max_steps=0)


def test_simulation_default_disabled():
    """an electrostatic solver is not attached by default"""
    assert _sim().get_as_pypicongpu().poisson_solver is None


def test_simulation_attached():
    """a configured electrostatic solver ends up as the pypicongpu poisson solver"""
    sim = _sim(picongpu_electrostatic_solver=picmi.ElectrostaticSolver())
    assert sim.get_as_pypicongpu().poisson_solver == pypicongpu.PoissonSolver()


def test_write_input_file_activates_solver():
    """writing the input files activates the solver via the --poisson.activate flag"""
    sim = _sim(picongpu_electrostatic_solver=picmi.ElectrostaticSolver())
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "out"
        sim.write_input_file(outdir)
        n_cfg = (outdir / "etc/picongpu/N.cfg").read_text()
        assert "--poisson.activate" in n_cfg
        assert "--poisson.maxSteps 2000" in n_cfg
        assert "--poisson.tolerance 1e-08" in n_cfg
        assert "--poisson.tolerance 1.0e-8" not in n_cfg


def test_write_input_file_preconditioner_disable_flag():
    """disabling the preconditioner emits only the disable flag, not maxSteps"""
    sim = _sim(
        picongpu_electrostatic_solver=picmi.ElectrostaticSolver(
            preconditioner="none", preconditioner_maximum_iterations=5
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "out"
        sim.write_input_file(outdir)
        n_cfg = (outdir / "etc/picongpu/N.cfg").read_text()
        assert "--poisson.preconditioner.disable" in n_cfg
        assert "--poisson.preconditioner.maxSteps" not in n_cfg


def test_write_input_file_non_default_tolerance_is_clean():
    """a non-default tolerance renders as a clean decimal literal, not binary noise"""
    sim = _sim(picongpu_electrostatic_solver=picmi.ElectrostaticSolver(required_precision=1e-6))
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "out"
        sim.write_input_file(outdir)
        n_cfg = (outdir / "etc/picongpu/N.cfg").read_text()
        assert "--poisson.tolerance 1e-06" in n_cfg
        assert "9.9999999999999995e-7" not in n_cfg


def test_write_input_file_default_preconditioner_max_steps():
    """with the default preconditioner, maxSteps is emitted"""
    sim = _sim(picongpu_electrostatic_solver=picmi.ElectrostaticSolver())
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "out"
        sim.write_input_file(outdir)
        n_cfg = (outdir / "etc/picongpu/N.cfg").read_text()
        assert "--poisson.preconditioner.disable" not in n_cfg
        assert "--poisson.preconditioner.maxSteps 20" in n_cfg


def test_write_input_file_disabled_has_no_flag():
    """without an electrostatic solver no --poisson.* options are emitted"""
    sim = _sim()
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "out"
        sim.write_input_file(outdir)
        n_cfg = (outdir / "etc/picongpu/N.cfg").read_text()
        assert "--poisson" not in n_cfg


def test_electrostatic_solver_rejected_as_em_solver():
    """the PICMI-standard solver= slot rejects ElectrostaticSolver with a clear error"""
    with pytest.raises(AttributeError, match="picongpu_electrostatic_solver"):
        picmi.Simulation(
            time_step_size=1.39e-16,
            max_steps=32,
            solver=picmi.ElectrostaticSolver(),
        )


def test_electrostatic_solver_grid_must_match_em_solver_grid():
    """a mismatched grid on the electrostatic solver is rejected"""
    mismatched = picmi.ElectrostaticSolver(grid=_grid(number_of_cells=[1, 2, 3]))
    with pytest.raises(ValueError, match="must match the grid"):
        _sim(picongpu_electrostatic_solver=mismatched)


def test_electrostatic_solver_same_grid_accepted():
    """passing the same grid as the electromagnetic solver is accepted"""
    grid = _grid()
    sim = picmi.Simulation(
        time_step_size=1.39e-16,
        max_steps=32,
        solver=picmi.ElectromagneticSolver(method="Yee", grid=grid),
        picongpu_electrostatic_solver=picmi.ElectrostaticSolver(grid=grid),
    )
    assert sim.get_as_pypicongpu().poisson_solver == pypicongpu.PoissonSolver()


def test_electrostatic_solver_default_grid_accepted():
    """omitting the grid on the electrostatic solver is accepted (uses EM solver grid)"""
    sim = _sim(picongpu_electrostatic_solver=picmi.ElectrostaticSolver())
    assert sim.get_as_pypicongpu().poisson_solver == pypicongpu.PoissonSolver()
