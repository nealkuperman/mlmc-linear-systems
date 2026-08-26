"""Orchestration of reproducible multilevel Monte Carlo runs."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from .linear_solver import solve_linear_system
from .mlmc_correction import (
    SampleCorrection,
    SystemSolver,
    compute_sample_correction,
)
from .mlmc_model import MultilevelModel
from .mlmc_statistics import LevelStatistics


RandomnessT = TypeVar("RandomnessT")
ModelInputT = TypeVar("ModelInputT")


def _make_sample_rng(
    base_seed: int,
    fine_level: int,
    sample_index: int,
) -> np.random.Generator:
    """Create the deterministic RNG for one correction sample."""
    ...


@dataclass(frozen=True)
class LevelResult:
    """Immutable scalar snapshot of one correction level."""

    level: int
    sample_count: int
    mean_correction: float
    sample_variance: float
    variance_of_mean: float
    mean_sample_cost: float
    total_sample_cost: float


@dataclass(frozen=True)
class MLMCResult:
    """Immutable correction-level snapshots and MLMC estimates."""

    finest_level: int
    level_results: tuple[LevelResult, ...]

    @property
    def estimate(self) -> float:
        """Return the sum of the correction-level sample means."""
        ...

    @property
    def estimator_variance(self) -> float:
        """Return the estimated sampling variance of the MLMC mean."""
        ...

    @property
    def standard_error(self) -> float:
        """Return the estimated standard error of the MLMC mean."""
        ...

    @property
    def total_cost(self) -> float:
        """Return the total measured cost across all correction levels."""
        ...


class MLMCRunner(Generic[RandomnessT, ModelInputT]):
    """Coordinate reproducible correction sampling and MLMC statistics."""

    _level_statistics: list[LevelStatistics]
    _finest_level: int | None

    def __init__(
        self,
        model: MultilevelModel[RandomnessT, ModelInputT],
        *,
        base_seed: int,
        solver: SystemSolver = solve_linear_system,
    ) -> None:
        """Store the model, base seed, and configured system solver."""
        ...

    def _resolve_finest_level(
        self,
        finest_level: int | None,
    ) -> int:
        """Resolve ``None`` to the model's finest level and validate it."""
        ...

    def _validate_initial_sample_counts(
        self,
        samples_per_level: Sequence[int],
        finest_level: int,
    ) -> tuple[int, ...]:
        """Validate initial counts for levels zero through finest level."""
        ...

    def _validate_additional_sample_counts(
        self,
        additional_samples_per_level: Sequence[int],
    ) -> tuple[int, ...]:
        """Validate nonnegative counts for the active levels."""
        ...

    def _initialize_run(self, finest_level: int) -> None:
        """Create empty statistics for a new active hierarchy."""
        ...

    def _compute_sample(
        self,
        fine_level: int,
        sample_index: int,
    ) -> SampleCorrection:
        """Compute one correction identified by level and sample index."""
        ...

    def _run_sample_batch(
        self,
        level_statistics: LevelStatistics,
        *,
        start_index: int,
        sample_count: int,
    ) -> None:
        """Add a consecutive batch of samples to one correction level."""
        ...

    def _create_result(self) -> MLMCResult:
        """Create an immutable snapshot of the active run."""
        ...

    def run_fixed(
        self,
        samples_per_level: Sequence[int],
        *,
        finest_level: int | None = None,
    ) -> MLMCResult:
        """Start a fixed run through the selected or model-finest level."""
        ...

    def add_samples(
        self,
        additional_samples_per_level: Sequence[int],
    ) -> MLMCResult:
        """Add fixed sample counts to the active run."""
        ...

    @property
    def result(self) -> MLMCResult:
        """Return an immutable snapshot of the active run."""
        ...

    def run_to_tolerance(
        self,
        sampling_tolerance: float,
        *,
        pilot_samples_per_level: int,
        maximum_samples: int | None = None,
    ) -> MLMCResult:
        """Add samples until a sampling-error target is met."""
        ...
