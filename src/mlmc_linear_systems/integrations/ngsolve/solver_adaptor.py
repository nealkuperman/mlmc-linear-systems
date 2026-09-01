"""This module provides functions to adapt NGSolve finite element levels to the core solver interface
used throughout the mlmc-linear-systems package.

These adaptor functions convert NGSolve-specific data structures into a generic LinearSystem,
enabling the MLMC runner to assemble and solve linear systems for each multilevel correction.
This decouples finite element assembly from solution, allowing flexible solver and MLMC pipelines.
"""

import ngsolve as ng
import numpy as np
from scipy import sparse

from ...linear_solver import LinearSystem
from .level import Level


def build_free_dof_linear_system(
    level: Level,
) -> LinearSystem:
    """Build the core linear system for the level's free DOFs.

    Parameters
    ----------
    level
        NGSolve finite-element level containing the assembled bilinear
        form, assembled linear form, free and fixed DOF indices, configured
        Dirichlet values, and the grid function used for solution storage.

    Returns
    -------
    LinearSystem
        Reduced system whose matrix is the free-free block ``A_FF``, whose
        right-hand side is ``f_F - A_FD g_D``, and whose initial guess is a
        copy of the current free entries of ``level.gfu``. The matrix is a
        SciPy CSR matrix, and all vectors use the ordering in
        ``level.free_ids``.

    Notes
    -----
    ``F`` denotes every unconstrained DOF, including DOFs on Neumann
    boundaries. ``D`` denotes the fixed Dirichlet DOFs, and ``g_D`` contains
    their prescribed values. The free equations of the assembled system are

    ``A_FF u_F + A_FD g_D = f_F``.

    A full-length boundary lift is constructed with zeros at free DOFs and
    the level's configured boundary values at fixed Dirichlet DOFs. Computing
    ``f - A @ boundary_lift`` and selecting its free entries produces
    ``f_F - A_FD g_D`` without explicitly forming ``A_FD``.

    This function does not modify ``level.gfu``. Both the right-hand side and
    initial guess are copied so the returned system does not retain writable
    views into NGSolve vectors.
    """
    bilinear = level.A_csr
    right_hand_side = level.f.vec.FV().NumPy().copy()

    boundary_lift = np.zeros(level.ndof, dtype=right_hand_side.dtype)
    level.enforce_dirichlet(boundary_lift)
    corrected_right_hand_side = right_hand_side - bilinear @ boundary_lift

    free_ids = level.free_ids
    free_bilinear = sparse.csr_matrix(
        bilinear[free_ids][:, free_ids]
    )
    free_right_hand_side = np.asarray(
        corrected_right_hand_side[free_ids]
    ).copy()
    initial_guess = level.gfu.vec.FV().NumPy()[free_ids].copy()

    return LinearSystem(
        A=free_bilinear,
        b=free_right_hand_side,
        x0=initial_guess,
    )


def restore_free_dof_solution(
    level: Level,
    solution: np.ndarray,
) -> ng.GridFunction:
    """Restore a reduced solver result to the full NGSolve grid function.

    Parameters
    ----------
    level
        Level whose ``gfu`` will receive the complete finite-element
        solution.
    solution
        One-dimensional array containing one value per free DOF, ordered in
        the same way as ``level.free_ids``. This is normally the ``solution``
        array returned by the core linear-system solver.

    Returns
    -------
    ng.GridFunction
        The same object as ``level.gfu``, updated in place with the free
        solution values and the level's configured Dirichlet values.

    Raises
    ------
    ValueError
        If ``solution`` does not have shape
        ``(len(level.free_ids),)``.

    Notes
    -----
    The reduced solver result contains only the unknown free coefficients.
    This function writes those coefficients into their global DOF positions
    and then calls ``level.enforce_dirichlet()`` to populate every fixed DOF.
    DOFs on Neumann boundaries are free and are therefore supplied by
    ``solution`` rather than by the Dirichlet enforcement step.

    The function mutates ``level.gfu``; it does not create a separate grid
    function or retain the reduced solution array.
    """
    free_solution = np.asarray(solution)
    expected_shape = (level.free_ids.size,)
    if free_solution.shape != expected_shape:
        raise ValueError(
            "solution must contain one entry per free DOF; "
            f"expected shape {expected_shape}, but received "
            f"shape {free_solution.shape}."
        )

    coefficients = level.gfu.vec.FV().NumPy()
    coefficients[level.free_ids] = free_solution
    level.enforce_dirichlet(coefficients)
    return level.gfu
