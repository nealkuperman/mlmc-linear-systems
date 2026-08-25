"""Evaluation of individual MLMC corrections."""

from dataclasses import dataclass

from .linear_solver import SolveResult


@dataclass(frozen=True)
class CorrectionSample:
    """One sampled MLMC correction and its solve diagnostics."""

    fine_level: int

    fine_qoi: float
    coarse_qoi: float | None

    fine_solve_result: SolveResult
    coarse_solve_result: SolveResult | None

    elapsed_time: float

    @property
    def value(self) -> float:
        """Return Q_0 or Q_level - Q_(level - 1)."""
        if self.coarse_qoi is None:
            return self.fine_qoi

        return self.fine_qoi - self.coarse_qoi