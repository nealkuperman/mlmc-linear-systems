"""Tests for MLMC run orchestration."""

import numpy as np
import pytest

from mlmc_linear_systems.linear_solver import solve_linear_system
from mlmc_linear_systems.mlmc_runner import (
    MLMCRunner,
    _make_sample_rng,
)

from .synthetic_model import SyntheticLinearModel


def test_sample_rng_is_reproducible_for_one_task():
    """Reconstruct the same stream from the same sample identity."""
    first_rng = _make_sample_rng(123, 2, 7)
    second_rng = _make_sample_rng(123, 2, 7)

    assert np.array_equal(
        first_rng.normal(size=5),
        second_rng.normal(size=5),
    )


def test_sample_rng_changes_with_level_or_sample_index():
    """Assign distinct streams to distinct correction tasks."""
    level_zero_values = _make_sample_rng(123, 0, 0).normal(size=5)
    level_one_values = _make_sample_rng(123, 1, 0).normal(size=5)
    next_sample_values = _make_sample_rng(123, 0, 1).normal(size=5)

    assert not np.array_equal(level_zero_values, level_one_values)
    assert not np.array_equal(level_zero_values, next_sample_values)


def test_sample_rng_does_not_depend_on_task_creation_order():
    """Allow serial or parallel scheduling without changing a task stream."""
    tasks = [(0, 0), (0, 1), (1, 0), (1, 1)]

    forward_values = {
        task: _make_sample_rng(321, *task).normal()
        for task in tasks
    }
    reverse_values = {
        task: _make_sample_rng(321, *task).normal()
        for task in reversed(tasks)
    }

    assert forward_values == reverse_values


def test_fixed_run_defaults_to_the_model_finest_level():
    """Use every available correction when no finest level is supplied."""
    runner = MLMCRunner(
        SyntheticLinearModel(number_of_levels=3),
        base_seed=123,
    )

    result = runner.run_fixed([5, 3, 2])

    assert result.finest_level == 2
    assert tuple(
        level_result.level for level_result in result.level_results
    ) == (0, 1, 2)
    assert tuple(
        level_result.sample_count
        for level_result in result.level_results
    ) == (5, 3, 2)


def test_fixed_run_can_target_a_lower_finest_level():
    """Estimate a deliberate prefix of a larger model hierarchy."""
    runner = MLMCRunner(
        SyntheticLinearModel(number_of_levels=5),
        base_seed=123,
    )

    result = runner.run_fixed([4, 2], finest_level=1)

    assert result.finest_level == 1
    assert len(result.level_results) == 2
    assert tuple(
        level_result.sample_count
        for level_result in result.level_results
    ) == (4, 2)


def test_lower_finest_level_preserves_shared_corrections():
    """Keep shared correction samples unchanged across finest levels."""
    model = SyntheticLinearModel(number_of_levels=4)
    sample_counts = [5, 4, 3, 2]
    base_seed = 123

    full_result = MLMCRunner(
        model,
        base_seed=base_seed,
    ).run_fixed(sample_counts)

    lower_result = MLMCRunner(
        model,
        base_seed=base_seed,
    ).run_fixed(
        sample_counts[:3],
        finest_level=2,
    )

    for lower_level, full_level in zip(
        lower_result.level_results,
        full_result.level_results[:3],
        strict=True,
    ):
        assert lower_level.sample_count == full_level.sample_count
        assert lower_level.mean_correction == full_level.mean_correction
        assert lower_level.sample_variance == full_level.sample_variance

    finest_correction_mean = (
        full_result.level_results[3].mean_correction
    )

    assert np.isclose(
        full_result.estimate,
        lower_result.estimate + finest_correction_mean,
    )


def test_result_properties_aggregate_level_snapshots():
    """Calculate estimator values from immutable per-level snapshots."""
    runner = MLMCRunner(
        SyntheticLinearModel(number_of_levels=3),
        base_seed=123,
    )

    result = runner.run_fixed([5, 4, 3])

    assert np.isclose(
        result.estimate,
        sum(item.mean_correction for item in result.level_results),
    )
    assert np.isclose(
        result.estimator_variance,
        sum(item.variance_of_mean for item in result.level_results),
    )
    assert np.isclose(
        result.standard_error,
        np.sqrt(result.estimator_variance),
    )
    assert np.isclose(
        result.total_cost,
        sum(item.total_sample_cost for item in result.level_results),
    )


def test_fixed_run_uses_the_configured_solver():
    """Pass every generated system through the runner's solver."""
    solve_count = 0

    def counting_solver(system):
        nonlocal solve_count
        solve_count += 1
        return solve_linear_system(system)

    runner = MLMCRunner(
        SyntheticLinearModel(number_of_levels=2),
        base_seed=123,
        solver=counting_solver,
    )

    runner.run_fixed([2, 3])

    assert solve_count == 8


def test_add_samples_allows_zeros_and_continues_sample_streams():
    """Match a one-shot run while leaving zero-count levels unchanged."""
    model = SyntheticLinearModel(number_of_levels=3)
    incremental_runner = MLMCRunner(model, base_seed=456)
    initial_result = incremental_runner.run_fixed([2, 2, 2])

    updated_result = incremental_runner.add_samples([2, 0, 1])

    one_shot_result = MLMCRunner(model, base_seed=456).run_fixed(
        [4, 2, 3]
    )

    assert tuple(
        item.sample_count for item in updated_result.level_results
    ) == (4, 2, 3)
    assert tuple(
        item.sample_count for item in initial_result.level_results
    ) == (2, 2, 2)

    for incremental, one_shot in zip(
        updated_result.level_results,
        one_shot_result.level_results,
        strict=True,
    ):
        assert np.isclose(
            incremental.mean_correction,
            one_shot.mean_correction,
        )
        assert np.isclose(
            incremental.sample_variance,
            one_shot.sample_variance,
        )


def test_result_requires_an_active_run():
    """Do not produce an apparently valid empty estimator."""
    runner = MLMCRunner(
        SyntheticLinearModel(number_of_levels=2),
        base_seed=123,
    )

    with pytest.raises(RuntimeError, match="No MLMC run"):
        _ = runner.result


@pytest.mark.parametrize(
    ("sample_counts", "finest_level", "message"),
    [
        ([2, 2], None, "exactly 3"),
        ([2, 1], 1, "at least two"),
        ([2, 2, 2, 2], 3, "between 0 and 2"),
    ],
)
def test_fixed_run_rejects_invalid_requests(
    sample_counts,
    finest_level,
    message,
):
    """Validate the selected hierarchy before computing corrections."""
    runner = MLMCRunner(
        SyntheticLinearModel(number_of_levels=3),
        base_seed=123,
    )

    with pytest.raises(ValueError, match=message):
        runner.run_fixed(
            sample_counts,
            finest_level=finest_level,
        )


def test_add_samples_rejects_invalid_requests():
    """Require nonnegative work for every already-active level."""
    runner = MLMCRunner(
        SyntheticLinearModel(number_of_levels=2),
        base_seed=123,
    )

    with pytest.raises(RuntimeError, match="active run"):
        runner.add_samples([1, 0])

    runner.run_fixed([2, 2])

    with pytest.raises(ValueError, match="exactly 2"):
        runner.add_samples([1])
    with pytest.raises(ValueError, match="nonnegative"):
        runner.add_samples([0, -1])
    with pytest.raises(ValueError, match="must be positive"):
        runner.add_samples([0, 0])


def test_fixed_run_rejects_replacing_an_active_run():
    """Protect accumulated statistics from an accidental restart."""
    runner = MLMCRunner(
        SyntheticLinearModel(number_of_levels=1),
        base_seed=123,
    )
    runner.run_fixed([2])

    with pytest.raises(RuntimeError, match="already active"):
        runner.run_fixed([2])


def test_run_to_tolerance_is_explicitly_deferred():
    """Avoid presenting the future adaptive policy as implemented."""
    runner = MLMCRunner(
        SyntheticLinearModel(number_of_levels=1),
        base_seed=123,
    )

    with pytest.raises(NotImplementedError, match="not implemented"):
        runner.run_to_tolerance(
            0.01,
            pilot_samples_per_level=2,
        )
