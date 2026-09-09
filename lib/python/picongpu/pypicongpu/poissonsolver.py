"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: PIConGPU AI Agent (TT-78)
License: GPLv3+
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, computed_field

from .rendering import RenderedObject


class PoissonSolver(RenderedObject, BaseModel):
    """
    Poisson solver used to compute the initial electric field.

    Semantics
    ---------
    PIConGPU starts every simulation with a vanishing electric field. The
    Poisson solver is a *starting-condition* feature: it once solves the static
    (Poisson) equation

        divergence E = rho / epsilon_0

    for the electric field that corresponds to the charge density of the
    initially specified species, and initialises the electromagnetic field
    solver with that electric field before the time loop starts. During the
    simulation the electromagnetic Maxwell solver evolves the fields as usual;
    the Poisson solver does not replace it.

    Implementation
    --------------
    The (discretised) Poisson equation is solved iteratively with the
    BiCGStab (biconjugate gradient stabilized) Krylov method, optionally
    accelerated by a fixed set of preconditioner iterations. The corresponding
    command line options are emitted under the ``--poisson.*`` prefix, see
    ``N.cfg.mustache``.

    All parameters are optional and default to the CLI defaults of the C++
    implementation.
    """

    max_steps: Annotated[int, Field(..., gt=0)] = 2000
    """maximum number of iterations for the Poisson solver"""

    tolerance: Annotated[float, Field(..., gt=0.0)] = 1e-8
    """maximal tolerated error of the Poisson solver (residual norm)"""

    preconditioner: Literal["default", "none"] = "default"
    """preconditioner for the Poisson solver; set to "none" to disable it"""

    preconditioner_max_steps: Annotated[int, Field(..., gt=0)] = 20
    """maximum number of iterations for the preconditioner"""

    @computed_field
    def preconditioner_disabled(self) -> bool:
        return self.preconditioner == "none"

    @computed_field
    def tolerance_rendered(self) -> str:
        """shortest exact decimal representation of the tolerance (as a C++ literal)"""
        return repr(self.tolerance)
