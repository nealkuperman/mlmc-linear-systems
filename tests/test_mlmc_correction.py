"""Tests for computing individual MLMC correction samples."""

import numpy as np
import pytest

from mlmc_linear_systems.linear_solver import (
    LinearSolveResult,
    LinearSystem,
    solve_linear_system,
)
from mlmc_linear_systems.mlmc_correction import (
    SampleCorrection,
    compute_sample_correction,
)
from mlmc_linear_systems.mlmc_model import CoupledInputs

from .synthetic_model import SyntheticLinearModel, SyntheticModelInput


def test_level_zero_computes_one_qoi_without_a_coarse_solve():
    """Represent the base correction as Y_0 = Q_0."""
    model = SyntheticLinearModel(number_of_levels=1)
    seed = 123
    expected_random_value = float(np.random.default_rng(seed).normal())

    sample = compute_sample_correction(
        model,
        fine_level=0,
        rng=np.random.default_rng(seed),
    )

    expected_qoi = model.level_value(
        0,
        SyntheticModelInput(expected_random_value),
    )

    assert isinstance(sample, SampleCorrection)
    assert sample.fine_level == 0
    assert np.isclose(sample.fine_qoi, expected_qoi)
    assert sample.coarse_qoi is None
    assert sample.coarse_solve_result is None
    assert np.isclose(sample.value, expected_qoi)
    assert sample.elapsed_time >= 0.0


def test_positive_level_computes_a_coupled_correction():
    """Solve adjacent levels using one shared random realization."""
    model = SyntheticLinearModel(number_of_levels=3)
    fine_level = 2
    seed = 456
    random_value = float(np.random.default_rng(seed).normal())
    solve_count = 0

    def counting_solver(system: LinearSystem) -> LinearSolveResult:
        nonlocal solve_count
        solve_count += 1
        return solve_linear_system(system)

    sample = compute_sample_correction(
        model,
        fine_level=fine_level,
        rng=np.random.default_rng(seed),
        solver=counting_solver,
    )

    expected_correction = (
        model.level_step(fine_level)
        - model.level_step(fine_level - 1)
    ) * np.sqrt(1e-4 + abs(random_value))

    assert solve_count == 2
    assert sample.coarse_qoi is not None
    assert sample.coarse_solve_result is not None
    assert sample.fine_solve_result.success
    assert sample.coarse_solve_result.success
    assert np.isclose(sample.value, expected_correction)


@pytest.mark.parametrize("fine_level", [-1, 2])
def test_unavailable_fine_level_is_rejected(fine_level: int):
    """Validate a requested level before drawing model randomness."""
    model = SyntheticLinearModel(number_of_levels=2)

    with pytest.raises(ValueError, match="fine_level must be between"):
        compute_sample_correction(
            model,
            fine_level=fine_level,
            rng=np.random.default_rng(123),
        )


class InvalidCouplingModel(SyntheticLinearModel):
    """Synthetic model that deliberately violates the coupling contract."""

    def __init__(self, number_of_levels: int, *, omit_coarse: bool):
        super().__init__(number_of_levels)
        self.omit_coarse = omit_coarse

    def couple_inputs(
        self,
        fine_level: int,
        randomness: float,
    ) -> CoupledInputs[SyntheticModelInput]:
        model_input = SyntheticModelInput(randomness)
        if self.omit_coarse:
            return CoupledInputs(fine=model_input, coarse=None)
        return CoupledInputs(fine=model_input, coarse=model_input)


def test_level_zero_rejects_a_coarse_input():
    """Require only the fine input for the base-level correction."""
    model = InvalidCouplingModel(1, omit_coarse=False)

    with pytest.raises(ValueError, match="must be None"):
        compute_sample_correction(
            model,
            fine_level=0,
            rng=np.random.default_rng(123),
        )


def test_positive_level_requires_a_coarse_input():
    """Require adjacent inputs for every positive-level correction."""
    model = InvalidCouplingModel(2, omit_coarse=True)

    with pytest.raises(ValueError, match="coarse input is required"):
        compute_sample_correction(
            model,
            fine_level=1,
            rng=np.random.default_rng(123),
        )


def test_unsuccessful_solve_stops_the_correction():
    """Do not calculate a QOI from an unsuccessful solver result."""
    model = SyntheticLinearModel(number_of_levels=1)

    def failed_solver(system: LinearSystem) -> LinearSolveResult:
        return LinearSolveResult(
            solution=np.zeros_like(system.b),
            success=False,
            solver_name="test",
            iterations=None,
            initial_residual_norm=1.0,
            final_residual_norm=1.0,
            residual_history=[],
            solve_time=0.0,
            message="Deliberate test failure.",
        )

    with pytest.raises(RuntimeError, match="Fine solve failed"):
        compute_sample_correction(
            model,
            fine_level=0,
            rng=np.random.default_rng(123),
            solver=failed_solver,
        )
