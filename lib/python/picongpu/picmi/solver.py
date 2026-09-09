"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from collections.abc import Sequence
from typing import Annotated, Literal

from picmistandard import PICMI_BinomialSmoother, PICMI_ElectromagneticSolver, PICMI_ElectrostaticSolver

from picongpu.pypicongpu import util
from picongpu.pypicongpu.field_solver import AnySolver, LeheSolver, YeeSolver
from picongpu.pypicongpu.poissonsolver import PoissonSolver


class BinomialSmoother(PICMI_BinomialSmoother):
    """
    PICMI Binomial Smoother

    PIConGPU's binomial current deposition uses fixed parameters, so all
    standard parameters except `n_pass` (which must be given by the standard
    but is not used) are rejected.
    """

    compensation: Annotated[Sequence[bool] | None, util.rejects_unsupported("binomial smoother parameters")] = None
    stride: Annotated[Sequence[int] | None, util.rejects_unsupported("binomial smoother parameters")] = None
    alpha: Annotated[Sequence[float] | None, util.rejects_unsupported("binomial smoother parameters")] = None


class ElectromagneticSolver(PICMI_ElectromagneticSolver):
    """
    PICMI Electromagnic Solver

    See PICMI spec for full documentation.

    Only the Yee and Lehe solvers are supported; solver options that PIConGPU
    does not implement are rejected at construction time.
    """

    field_smoother: Annotated[PICMI_BinomialSmoother | None, util.rejects_unsupported("field smoothers")] = None
    method: Literal["Yee", "Lehe"]
    stencil_order: Annotated[Sequence[int] | None, util.rejects_unsupported("higher order solver stencils")] = None
    subcycling: Annotated[int | None, util.rejects_unsupported("subcycling")] = None
    galilean_velocity: Annotated[Sequence[float] | None, util.rejects_unsupported("galilean velocity")] = None
    divE_cleaning: Annotated[bool | None, util.rejects_unsupported("divE cleaning")] = None
    divB_cleaning: Annotated[bool | None, util.rejects_unsupported("divB cleaning")] = None
    pml_divE_cleaning: Annotated[bool | None, util.rejects_unsupported("pml divE cleaning")] = None
    pml_divB_cleaning: Annotated[bool | None, util.rejects_unsupported("pml divB cleaning")] = None

    def get_as_pypicongpu(self) -> AnySolver:
        return YeeSolver() if self.method == "Yee" else LeheSolver()


class ElectrostaticSolver(PICMI_ElectrostaticSolver):
    """
    Electrostatic solver computing the initial electric field of the simulation.

    Semantics
    ---------
    PICMI and the PIConGPU extension maintain the purely electromagnetic nature
    of the simulation: the electric field is propagated by the electromagnetic
    :class:`ElectromagneticSolver` given as ``Simulation.solver``. This class
    instead describes a *starting condition*: the electric field is not
    initialised to zero but to the solution of the static (Poisson) equation

        divergence E = rho / epsilon_0

    for the charge density of the initially specified species. The solver runs
    once before the time loop starts and is then handed over to the regular
    electromagnetic solver.

    Implementation
    --------------
    In contrast to the methods listed by the PICMI standard (``FFT``,
    ``Multigrid``), PIConGPU solves the discretised Poisson equation
    iteratively with the BiCGStab (biconjugate gradient stabilized) Krylov
    method, optionally accelerated by a preconditioner. ``method`` therefore
    only supports ``"BICGStab"``.

    Attach an instance to a simulation via ``Simulation.picongpu_electrostatic_solver``.

    Note on the grid
    ----------------
    The solver operates on the grid of the electromagnetic solver (i.e. on
    ``Simulation.solver.grid``); the ``grid`` argument of the PICMI base class
    is therefore accepted but must be consistent with it and is not used to
    steer the solve.
    """

    #: method used to solve the Poisson equation within PIConGPU
    #: the PICMI standard lists ``FFT``/``Multigrid``, PIConGPU implements BiCGStab
    methods_list = ["BICGStab"]

    def __init__(
        self,
        grid=None,
        method: Literal["BICGStab"] = "BICGStab",
        required_precision: float = 1e-8,
        maximum_iterations: int = 2000,
        preconditioner: Literal["default", "none"] = "default",
        preconditioner_maximum_iterations: int = 20,
        **kw,
    ):
        assert method is None or method in self.methods_list, "method must be one of " + ", ".join(self.methods_list)
        self.grid = grid
        self.method = method or "BICGStab"
        self.required_precision = required_precision
        self.maximum_iterations = maximum_iterations
        self.preconditioner = preconditioner
        self.preconditioner_maximum_iterations = preconditioner_maximum_iterations
        self.handle_init(kw)

    def get_as_pypicongpu(self) -> PoissonSolver:
        return PoissonSolver(
            tolerance=self.required_precision,
            max_steps=self.maximum_iterations,
            preconditioner=self.preconditioner,
            preconditioner_max_steps=self.preconditioner_maximum_iterations,
        )
