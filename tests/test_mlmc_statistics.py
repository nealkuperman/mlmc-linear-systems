"""Tests for online MLMC statistics."""

import gc
import weakref

import numpy as np
import pytest

from mlmc_linear_systems.linear_solver import LinearSolveResult
from mlmc_linear_systems.mlmc_correction import SampleCorrection
from mlmc_linear_systems.mlmc_statistics import (
    LevelStatistics,
    RunningStatistics,
)


def make_solve_result(solution: np.ndarray) -> LinearSolveResult:
    """Create a successful solve result for a statistics-only test."""
    return LinearSolveResult(
        solution=solution,
        success=True,
        solver_name="test",
        iterations=None,
        initial_residual_norm=0.0,
        final_residual_norm=0.0,
        residual_history=[],
        solve_time=0.0,
        message="Test solve succeeded.",
    )


def make_sample_correction(
    *,
    fine_level: int,
    fine_qoi: float,
    coarse_qoi: float | None,
    elapsed_time: float,
    fine_solution: np.ndarray | None = None,
) -> SampleCorrection:
    """Create one correction sample without running the model layer."""
    if fine_solution is None:
        fine_solution = np.array([fine_qoi])

    coarse_result = None
    if coarse_qoi is not None:
        coarse_result = make_solve_result(np.array([coarse_qoi]))

    return SampleCorrection(
        fine_level=fine_level,
        fine_qoi=fine_qoi,
        coarse_qoi=coarse_qoi,
        fine_solve_result=make_solve_result(fine_solution),
        coarse_solve_result=coarse_result,
        elapsed_time=elapsed_time,
    )


def test_running_statistics_start_empty():
    """Represent an accumulator before any observations arrive."""
    statistics = RunningStatistics()

    assert statistics.count == 0
    assert statistics.mean == 0.0
    assert statistics.total == 0.0
    assert np.isnan(statistics.sample_variance)
    assert np.isnan(statistics.variance_of_mean)


def test_running_statistics_match_batch_statistics():
    """Update mean and variance online using Welford's algorithm."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    statistics = RunningStatistics()

    for value in values:
        statistics.update(value)

    assert statistics.count == len(values)
    assert np.isclose(statistics.mean, np.mean(values))
    assert np.isclose(statistics.total, np.sum(values))
    assert "total" not in vars(statistics)
    assert np.isclose(
        statistics._sum_squared_deviations,
        5.0,
    )
    assert np.isclose(
        statistics.sample_variance,
        np.var(values, ddof=1),
    )
    assert np.isclose(
        statistics.variance_of_mean,
        np.var(values, ddof=1) / len(values),
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_running_statistics_reject_nonfinite_values(value: float):
    """Keep the accumulator unchanged when an observation is invalid."""
    statistics = RunningStatistics()

    with pytest.raises(ValueError, match="finite"):
        statistics.update(value)

    assert statistics.count == 0
    assert statistics.mean == 0.0


def test_level_statistics_track_corrections_and_costs():
    """Accumulate correction values and elapsed times independently."""
    statistics = LevelStatistics(level=1)
    samples = (
        make_sample_correction(
            fine_level=1,
            fine_qoi=3.0,
            coarse_qoi=2.0,
            elapsed_time=0.25,
        ),
        make_sample_correction(
            fine_level=1,
            fine_qoi=6.0,
            coarse_qoi=2.0,
            elapsed_time=0.75,
        ),
    )

    for sample in samples:
        statistics.update(sample)

    assert statistics.sample_count == 2
    assert statistics.mean_correction == 2.5
    assert statistics.sample_variance == 4.5
    assert statistics.variance_of_mean == 2.25
    assert statistics.mean_sample_cost == 0.5
    assert statistics.total_sample_cost == 1.0


def test_level_statistics_reject_negative_levels():
    """Require each accumulator to identify a nonnegative level."""
    with pytest.raises(ValueError, match="nonnegative"):
        LevelStatistics(level=-1)


def test_level_statistics_reject_a_sample_from_another_level():
    """Prevent corrections from being accumulated at the wrong level."""
    statistics = LevelStatistics(level=1)
    sample = make_sample_correction(
        fine_level=0,
        fine_qoi=1.0,
        coarse_qoi=None,
        elapsed_time=0.25,
    )

    with pytest.raises(ValueError, match="received level 0"):
        statistics.update(sample)

    assert statistics.sample_count == 0
    assert statistics.cost.count == 0


@pytest.mark.parametrize("elapsed_time", [-0.1, np.nan, np.inf])
def test_level_statistics_reject_invalid_elapsed_time(
    elapsed_time: float,
):
    """Require a finite, nonnegative cost before updating either statistic."""
    statistics = LevelStatistics(level=0)
    sample = make_sample_correction(
        fine_level=0,
        fine_qoi=1.0,
        coarse_qoi=None,
        elapsed_time=elapsed_time,
    )

    with pytest.raises(ValueError, match="finite and nonnegative"):
        statistics.update(sample)

    assert statistics.sample_count == 0
    assert statistics.cost.count == 0


def test_level_statistics_do_not_retain_solution_arrays():
    """Release large solve outputs after their scalar data are consumed."""
    statistics = LevelStatistics(level=0)
    solution = np.ones(100)
    solution_reference = weakref.ref(solution)
    sample = make_sample_correction(
        fine_level=0,
        fine_qoi=1.0,
        coarse_qoi=None,
        elapsed_time=0.25,
        fine_solution=solution,
    )

    statistics.update(sample)
    del sample
    del solution
    gc.collect()

    assert solution_reference() is None
    assert statistics.mean_correction == 1.0
    assert statistics.mean_sample_cost == 0.25
