"""Evaluation of individual MLMC corrections."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeVar

import numpy as np

from .linear_solver import (
    LinearSolveResult,
    LinearSystem,
    solve_linear_system,
)
from .mlmc_model import MultilevelModel


RandomnessT = TypeVar("RandomnessT")
ModelInputT = TypeVar("ModelInputT")

SystemSolver = Callable[[LinearSystem], LinearSolveResult]


@dataclass(frozen=True)
class SampleCorrection:
    """One sampled MLMC correction and its solve diagnostics."""

    fine_level: int

    fine_qoi: float
    coarse_qoi: float | None

    fine_solve_result: LinearSolveResult
    coarse_solve_result: LinearSolveResult | None

    elapsed_time: float

    @property
    def value(self) -> float:
        """Return Q_0 or Q_level - Q_(level - 1)."""
        if self.coarse_qoi is None:
            return self.fine_qoi

        return self.fine_qoi - self.coarse_qoi


def compute_sample_correction(
    model: MultilevelModel[RandomnessT, ModelInputT],
    fine_level: int,
    rng: np.random.Generator,
    *,
    solver: SystemSolver = solve_linear_system,
) -> SampleCorrection:
    """Compute one coupled MLMC correction sample.

    At level zero, this function computes ``Y_0 = Q_0``. At a positive
    fine level, it computes ``Y_level = Q_level - Q_(level - 1)``.
    Exactly one random realization is drawn and used to construct both
    adjacent-level inputs.

    Parameters
    ----------
    model
        User-supplied multilevel model.
    fine_level
        Fine level of the requested correction. Valid levels are zero
        through ``model.number_of_levels - 1``.
    rng
        Random-number generator used by the model to draw the shared
        randomness.
    solver
        Function that solves one ``LinearSystem``. The default dispatches
        to the direct method through ``solve_linear_system``.

    Returns
    -------
    SampleCorrection
        Fine and optional coarse quantities of interest, solve diagnostics,
        total elapsed time, and the resulting correction value.

    Raises
    ------
    ValueError
        If ``fine_level`` is unavailable or the model returns an invalid
        fine/coarse coupling.
    RuntimeError
        If either linear solve is unsuccessful.
    """
    if fine_level < 0 or fine_level >= model.number_of_levels:
        raise ValueError(
            f"fine_level must be between 0 and "
            f"{model.number_of_levels - 1}."
        )

    start_time = perf_counter()

    randomness = model.sample_randomness(fine_level, rng)
    inputs = model.couple_inputs(fine_level, randomness)
    coarse_input = inputs.coarse

    if fine_level == 0 and coarse_input is not None:
        raise ValueError(
            "The coarse input must be None at fine level zero."
        )
    if fine_level > 0 and coarse_input is None:
        raise ValueError(
            "A coarse input is required for a positive fine level."
        )

    fine_system = model.build_linear_system(
        fine_level,
        inputs.fine,
    )
    fine_solve_result = solver(fine_system)

    if not fine_solve_result.success:
        raise RuntimeError(
            f"Fine solve failed at level {fine_level}: "
            f"{fine_solve_result.message}"
        )

    fine_qoi = float(
        model.quantity_of_interest(
            fine_level,
            fine_solve_result.solution,
            inputs.fine,
        )
    )

    coarse_qoi: float | None = None
    coarse_solve_result: LinearSolveResult | None = None

    if coarse_input is not None:
        coarse_level = fine_level - 1
        coarse_system = model.build_linear_system(
            coarse_level,
            coarse_input,
        )
        coarse_solve_result = solver(coarse_system)

        if not coarse_solve_result.success:
            raise RuntimeError(
                f"Coarse solve failed at level {coarse_level}: "
                f"{coarse_solve_result.message}"
            )

        coarse_qoi = float(
            model.quantity_of_interest(
                coarse_level,
                coarse_solve_result.solution,
                coarse_input,
            )
        )

    elapsed_time = perf_counter() - start_time

    return SampleCorrection(
        fine_level=fine_level,
        fine_qoi=fine_qoi,
        coarse_qoi=coarse_qoi,
        fine_solve_result=fine_solve_result,
        coarse_solve_result=coarse_solve_result,
        elapsed_time=elapsed_time,
    )
