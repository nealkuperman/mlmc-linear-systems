"""Orchestration of reproducible multilevel Monte Carlo runs."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from numbers import Integral
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
    seed_sequence = np.random.SeedSequence(
        [base_seed, fine_level, sample_index]
    )
    return np.random.default_rng(seed_sequence)


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
        return sum(
            (
                result.mean_correction
                for result in self.level_results
            ),
            start=0.0,
        )

    @property
    def estimator_variance(self) -> float:
        """Return the estimated sampling variance of the MLMC mean."""
        return sum(
            (
                result.variance_of_mean
                for result in self.level_results
            ),
            start=0.0,
        )

    @property
    def standard_error(self) -> float:
        """Return the estimated standard error of the MLMC mean."""
        return sqrt(self.estimator_variance)

    @property
    def total_cost(self) -> float:
        """Return the total measured cost across all correction levels."""
        return sum(
            (
                result.total_sample_cost
                for result in self.level_results
            ),
            start=0.0,
        )


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

        number_of_levels = model.number_of_levels
        if (
            isinstance(number_of_levels, bool)
            or not isinstance(number_of_levels, Integral)
            or number_of_levels <= 0
        ):
            raise ValueError(
                "model.number_of_levels must be a positive integer."
            )

        if isinstance(base_seed, bool) or not isinstance(
            base_seed, Integral
        ):
            raise TypeError("base_seed must be an integer.")

        if base_seed < 0:
            raise ValueError("base_seed must be nonnegative.")

        if not callable(solver):
            raise TypeError("solver must be callable.")

        self._model = model
        self._base_seed = int(base_seed)
        self._solver = solver
        self._level_statistics = []
        self._finest_level = None

    def _resolve_finest_level(
        self,
        finest_level: int | None,
    ) -> int:
        """Resolve ``None`` to the model's finest level and validate it."""
        if finest_level is None:
            return int(self._model.number_of_levels) - 1

        if isinstance(finest_level, bool) or not isinstance(
            finest_level, Integral
        ):
            raise TypeError("finest_level must be an integer or None.")

        finest_level = int(finest_level)
        if (
            finest_level < 0
            or finest_level >= self._model.number_of_levels
        ):
            raise ValueError(
                "finest_level must be between 0 and "
                f"{self._model.number_of_levels - 1}."
            )

        return finest_level

    def _validate_initial_sample_counts(
        self,
        samples_per_level: Sequence[int],
        finest_level: int,
    ) -> tuple[int, ...]:
        """Validate initial counts for levels zero through finest level."""
        counts = tuple(samples_per_level)
        expected_count = finest_level + 1

        if len(counts) != expected_count:
            raise ValueError(
                "samples_per_level must contain exactly "
                f"{expected_count} counts for levels 0 (coarsest) through "
                f"{finest_level} (finest)."
            )

        normalized_counts: list[int] = []
        for level, count in enumerate(counts):
            if isinstance(count, bool) or not isinstance(count, Integral):
                raise TypeError(
                    f"Sample count at level {level} must be an integer."
                )
            if count < 2:
                raise ValueError(
                    f"Initial sample count at level {level} must be "
                    "at least two."
                )
            normalized_counts.append(int(count))

        return tuple(normalized_counts)

    def _validate_additional_sample_counts(
        self,
        additional_samples_per_level: Sequence[int],
    ) -> tuple[int, ...]:
        """Validate nonnegative counts for the active levels."""
        if self._finest_level is None:
            raise RuntimeError(
                "add_samples() requires an active run."
            )

        counts = tuple(additional_samples_per_level)
        expected_count = self._finest_level + 1

        if len(counts) != expected_count:
            raise ValueError(
                "additional_samples_per_level must contain exactly "
                f"{expected_count} counts for the active levels."
            )

        normalized_counts: list[int] = []
        for level, count in enumerate(counts):
            if isinstance(count, bool) or not isinstance(count, Integral):
                raise TypeError(
                    f"Additional sample count at level {level} must "
                    "be an integer."
                )
            if count < 0:
                raise ValueError(
                    f"Additional sample count at level {level} must "
                    "be nonnegative."
                )
            normalized_counts.append(int(count))

        if not any(normalized_counts):
            raise ValueError(
                "At least one additional sample count must be positive."
            )

        return tuple(normalized_counts)

    def _initialize_run(self, finest_level: int) -> None:
        """Create empty statistics for a new active hierarchy."""
        self._finest_level = finest_level
        self._level_statistics = [
            LevelStatistics(level=level)
            for level in range(finest_level + 1)
        ]

    def _compute_sample(
        self,
        fine_level: int,
        sample_index: int,
    ) -> SampleCorrection:
        """Compute one correction identified by level and sample index."""
        rng = _make_sample_rng(
            self._base_seed,
            fine_level,
            sample_index,
        )
        return compute_sample_correction(
            self._model,
            fine_level,
            rng,
            solver=self._solver,
        )

    def _run_sample_batch(
        self,
        level_statistics: LevelStatistics,
        *,
        start_index: int,
        sample_count: int,
    ) -> None:
        """Add a consecutive batch of samples to one correction level."""
        stop_index = start_index + sample_count
        for sample_index in range(start_index, stop_index):
            sample = self._compute_sample(
                level_statistics.level,
                sample_index,
            )
            level_statistics.update(sample)

    def _create_result(self) -> MLMCResult:
        """Create an immutable snapshot of the active run."""
        if self._finest_level is None:
            raise RuntimeError("No MLMC run has been started.")

        level_results = tuple(
            LevelResult(
                level=statistics.level,
                sample_count=statistics.sample_count,
                mean_correction=statistics.mean_correction,
                sample_variance=statistics.sample_variance,
                variance_of_mean=statistics.variance_of_mean,
                mean_sample_cost=statistics.mean_sample_cost,
                total_sample_cost=statistics.total_sample_cost,
            )
            for statistics in self._level_statistics
        )

        return MLMCResult(
            finest_level=self._finest_level,
            level_results=level_results,
        )

    def run_fixed(
        self,
        samples_per_level: Sequence[int],
        *,
        finest_level: int | None = None,
    ) -> MLMCResult:
        """Start a fixed run through the selected or model-finest level."""
        if self._finest_level is not None:
            raise RuntimeError(
                "A run is already active; use add_samples() or create "
                "a new runner."
            )

        resolved_finest_level = self._resolve_finest_level(finest_level)
        counts = self._validate_initial_sample_counts(
            samples_per_level,
            resolved_finest_level,
        )
        self._initialize_run(resolved_finest_level)

        for statistics, sample_count in zip(
            self._level_statistics,
            counts,
            strict=True,
        ):
            self._run_sample_batch(
                statistics,
                start_index=0,
                sample_count=sample_count,
            )

        return self._create_result()

    def add_samples(
        self,
        additional_samples_per_level: Sequence[int],
    ) -> MLMCResult:
        """Add fixed sample counts to the active run."""
        counts = self._validate_additional_sample_counts(
            additional_samples_per_level
        )

        for statistics, sample_count in zip(
            self._level_statistics,
            counts,
            strict=True,
        ):
            self._run_sample_batch(
                statistics,
                start_index=statistics.sample_count,
                sample_count=sample_count,
            )

        return self._create_result()

    @property
    def result(self) -> MLMCResult:
        """Return an immutable snapshot of the active run."""
        return self._create_result()

    def run_to_tolerance(
        self,
        sampling_tolerance: float,
        *,
        pilot_samples_per_level: int,
        maximum_samples: int | None = None,
    ) -> MLMCResult:
        """Add samples until a sampling-error target is met."""
        raise NotImplementedError(
            "Tolerance-based sample allocation is not implemented yet."
        )
