"""Online statistics used by MLMC estimators."""

from dataclasses import dataclass, field
from math import isfinite

from .mlmc_correction import SampleCorrection


@dataclass
class RunningStatistics:
    """Online statistics for scalar observations."""

    count: int = field(default=0, init=False)
    mean: float = field(default=0.0, init=False)

    _sum_squared_deviations: float = field(
        default=0.0,
        init=False,
        repr=False,
    )

    def update(self, value: float) -> None:
        """Add one scalar observation using Welford's algorithm."""
        value = float(value)

        if not isfinite(value):
            raise ValueError(
                "Running statistics require finite values."
            )

        self.count += 1

        difference = value - self.mean
        self.mean += difference / self.count

        difference_from_new_mean = value - self.mean

        self._sum_squared_deviations += (
            difference * difference_from_new_mean
        )

    @property
    def sample_variance(self) -> float:
        """Return the unbiased sample variance."""
        if self.count < 2:
            return float("nan")

        return (
            self._sum_squared_deviations
            / (self.count - 1)
        )

    @property
    def variance_of_mean(self) -> float:
        """Return the estimated variance of the sample mean."""
        if self.count < 2:
            return float("nan")

        return self.sample_variance / self.count

    @property
    def total(self) -> float:
        """Return the sum of all observations."""
        return self.mean * self.count


@dataclass
class LevelStatistics:
    """Correction and cost statistics for one MLMC level."""

    level: int

    correction: RunningStatistics = field(
        default_factory=RunningStatistics
    )
    cost: RunningStatistics = field(
        default_factory=RunningStatistics
    )

    def __post_init__(self) -> None:
        """Validate the represented MLMC level."""
        if self.level < 0:
            raise ValueError("level must be nonnegative.")

    def update(self, sample: SampleCorrection) -> None:
        """Consume one correction without retaining its solve results."""
        if sample.fine_level != self.level:
            raise ValueError(
                f"Expected a level-{self.level} sample, "
                f"but received level {sample.fine_level}."
            )

        if (
            not isfinite(sample.elapsed_time)
            or sample.elapsed_time < 0.0
        ):
            raise ValueError(
                "Sample elapsed time must be finite and nonnegative."
            )

        self.correction.update(sample.value)
        self.cost.update(sample.elapsed_time)

    @property
    def sample_count(self) -> int:
        """Return the number of correction samples."""
        return self.correction.count

    @property
    def mean_correction(self) -> float:
        """Return the estimated mean correction at this level."""
        return self.correction.mean

    @property
    def sample_variance(self) -> float:
        """Return the sample variance of the corrections."""
        return self.correction.sample_variance

    @property
    def variance_of_mean(self) -> float:
        """Return the estimated variance of the level mean."""
        return self.correction.variance_of_mean

    @property
    def mean_sample_cost(self) -> float:
        """Return the mean elapsed time per correction sample."""
        return self.cost.mean

    @property
    def total_sample_cost(self) -> float:
        """Return the total elapsed time for this level."""
        return self.cost.total
