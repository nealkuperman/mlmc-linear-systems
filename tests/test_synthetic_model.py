"""Tests for the synthetic multilevel linear-system model."""

import numpy as np
import pytest

from mlmc_linear_systems.linear_solver import direct_solve
from mlmc_linear_systems.mlmc_model import MultilevelModel

from .synthetic_model import SyntheticLinearModel, SyntheticModelInput


def evaluate_quantity(
    model: SyntheticLinearModel,
    level: int,
    model_input: SyntheticModelInput,
) -> float:
    """Build, solve, and evaluate the synthetic model at one level."""
    system = model.build_linear_system(level, model_input)
    result = direct_solve(system.A, system.b)

    assert result.success
    return model.quantity_of_interest(
        level,
        result.solution,
        model_input,
    )


def test_sample_randomness_returns_float():
    """Return the concrete randomness type declared for this model."""
    model: MultilevelModel[float, SyntheticModelInput] = (
        SyntheticLinearModel(number_of_levels=1)
    )

    randomness = model.sample_randomness(0, np.random.default_rng(123))

    assert isinstance(randomness, float)


@pytest.mark.parametrize("number_of_levels", [0, -1])
def test_model_requires_at_least_one_level(number_of_levels: int):
    """Reject models that would contain no valid level-zero system."""
    with pytest.raises(ValueError, match="must be positive"):
        SyntheticLinearModel(number_of_levels=number_of_levels)


def test_model_preconstructs_the_requested_hierarchy():
    """Store deterministic data for every configured level."""
    model = SyntheticLinearModel(number_of_levels=3)

    assert model.number_of_levels == 3
    assert len(model.levels) == 3
    assert [level.index for level in model.levels] == [0, 1, 2]
    assert [level.size for level in model.levels] == [4, 8, 16]
    assert np.allclose(model.levels[1].matrix.diagonal(), 3.0)


@pytest.mark.parametrize("level", [-1, 2])
def test_model_rejects_unavailable_levels(level: int):
    """Allow access only to levels constructed during model setup."""
    model = SyntheticLinearModel(number_of_levels=2)

    with pytest.raises(ValueError, match="between 0 and 1"):
        model.level_size(level)


def test_level_step_is_tied_to_system_size():
    """Halve h_level whenever the number of unknowns doubles."""
    model = SyntheticLinearModel(number_of_levels=2)

    assert model.level_size(0) == 4
    assert model.level_step(0) == 1.0 / 4.0
    assert model.level_size(1) == 8
    assert model.level_step(1) == 1.0 / 8.0


def test_level_zero_has_no_coarse_input():
    """Represent the base correction as one level-zero model input."""
    model = SyntheticLinearModel(number_of_levels=1)

    inputs = model.couple_inputs(0, randomness=0.75)

    assert inputs.fine.random_value == 0.75
    assert inputs.coarse is None


def test_adjacent_inputs_share_the_same_random_value():
    """Couple adjacent levels with one shared scalar realization."""
    model = SyntheticLinearModel(number_of_levels=3)

    inputs = model.couple_inputs(2, randomness=-0.75)

    assert inputs.fine.random_value == -0.75
    assert inputs.coarse is not None
    assert inputs.coarse.random_value == -0.75


def test_synthetic_system_has_the_documented_analytical_solution():
    """Recover a uniform vector containing q_level(X)."""
    model = SyntheticLinearModel(number_of_levels=2)
    model_input = SyntheticModelInput(random_value=1.0)

    system = model.build_linear_system(1, model_input)
    result = direct_solve(system.A, system.b)
    expected_value = model.level_value(1, model_input)

    assert result.success
    assert result.solution.shape == (8,)
    assert np.allclose(result.solution, expected_value)

    quantity = model.quantity_of_interest(
        1,
        result.solution,
        model_input,
    )
    assert np.isclose(quantity, expected_value)


def test_build_system_reuses_the_preconstructed_matrix():
    """Change the random right-hand side without rebuilding the level matrix."""
    model = SyntheticLinearModel(number_of_levels=2)

    first_system = model.build_linear_system(
        1,
        SyntheticModelInput(random_value=0.25),
    )
    second_system = model.build_linear_system(
        1,
        SyntheticModelInput(random_value=1.25),
    )

    assert first_system.A is model.levels[1].matrix
    assert second_system.A is model.levels[1].matrix
    assert not np.allclose(first_system.b, second_system.b)


def test_coupled_correction_equals_the_difference_of_level_errors():
    """Cancel the shared random value in an adjacent-level correction."""
    model = SyntheticLinearModel(number_of_levels=3)
    fine_level = 2
    random_value = -0.75
    inputs = model.couple_inputs(fine_level, random_value)

    assert inputs.coarse is not None
    fine_quantity = evaluate_quantity(model, fine_level, inputs.fine)
    coarse_quantity = evaluate_quantity(
        model,
        fine_level - 1,
        inputs.coarse,
    )
    correction = fine_quantity - coarse_quantity

    expected_correction = (
        model.level_step(fine_level)
        - model.level_step(fine_level - 1)
    ) * np.sqrt(1e-4 + abs(random_value))

    assert np.isclose(correction, expected_correction)
