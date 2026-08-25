"""Synthetic multilevel model used by the MLMC runner tests.

For one standard-normal random value ``X``, the level-dependent scalar is

    q_level(X) = X + h_level * sqrt(1e-4 + abs(X)),

where ``h_level = 1 / n_level`` and ``n_level`` is the number of unknowns.
The level solution is the uniform vector ``q_level(X) * ones(n_level)``.

The model preconstructs a finite hierarchy of sparse diagonal matrices.
Adjacent levels use the same ``X``, so the common random contribution cancels
when the MLMC correction is formed.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from mlmc_linear_systems.linear_solver import LinearSystem
from mlmc_linear_systems.mlmc_model import CoupledInputs


@dataclass(frozen=True)
class SyntheticModelInput:
    """Random input used to construct one synthetic level system."""

    random_value: float


@dataclass(frozen=True)
class SyntheticLevel:
    """Deterministic data stored for one synthetic level."""

    index: int
    size: int
    step: float
    matrix: sparse.csr_matrix


class SyntheticLinearModel:
    """Manufactured hierarchy with preconstructed deterministic levels."""

    def __init__(self, number_of_levels: int):
        """Construct and store all available model levels."""
        if number_of_levels <= 0:
            raise ValueError("number_of_levels must be positive.")

        self.number_of_levels = number_of_levels
        self.levels: tuple[SyntheticLevel, ...] = tuple(
            self._create_level(level)
            for level in range(number_of_levels)
        )

    @staticmethod
    def _create_level(level: int) -> SyntheticLevel:
        """Construct the deterministic data for one level."""
        size = 4 * 2**level
        step = 1.0 / size
        matrix = sparse.eye(
            size,
            format="csr",
            dtype=float,
        ) * float(level + 2)

        return SyntheticLevel(
            index=level,
            size=size,
            step=step,
            matrix=matrix,
        )

    def _get_level(self, level: int) -> SyntheticLevel:
        """Return one stored level after validating its index."""
        if level < 0 or level >= self.number_of_levels:
            raise ValueError(
                f"level must be between 0 and "
                f"{self.number_of_levels - 1}."
            )

        return self.levels[level]

    def level_size(self, level: int) -> int:
        """Return the number of unknowns at one stored level."""
        return self._get_level(level).size

    def level_step(self, level: int) -> float:
        """Return the approximation step at one stored level."""
        return self._get_level(level).step

    def sample_randomness(
        self,
        fine_level: int,
        rng: np.random.Generator,
    ) -> float:
        """Draw the standard-normal scalar shared by the level pair."""
        self.level_size(fine_level)
        return float(rng.normal())

    def couple_inputs(
        self,
        fine_level: int,
        randomness: float,
    ) -> CoupledInputs[SyntheticModelInput]:
        """Use the same scalar random realization at adjacent levels.

        Sharing the exact same value causes the limiting random contribution
        ``X`` to cancel in ``Q_level(X) - Q_(level - 1)(X)``. Only the smaller
        difference between the two level errors remains.
        """
        self.level_size(fine_level)
        fine_input = SyntheticModelInput(random_value=float(randomness))

        if fine_level == 0:
            return CoupledInputs(fine=fine_input, coarse=None)

        return CoupledInputs(
            fine=fine_input,
            coarse=SyntheticModelInput(random_value=float(randomness)),
        )

    def level_value(
        self,
        level: int,
        model_input: SyntheticModelInput,
    ) -> float:
        """Return the exact scalar approximation q_level(X)."""
        random_value = model_input.random_value
        level_error = self.level_step(level) * np.sqrt(
            1e-4 + abs(random_value)
        )
        return float(random_value + level_error)

    def build_linear_system(
        self,
        level: int,
        model_input: SyntheticModelInput,
    ) -> LinearSystem:
        """Inject the random input into one preconstructed level."""
        level_data = self._get_level(level)
        level_value = self.level_value(level, model_input)
        expected_solution = np.full(level_data.size, level_value)
        right_hand_side = np.asarray(
            level_data.matrix @ expected_solution
        )

        return LinearSystem(
            A=level_data.matrix,
            b=right_hand_side,
        )

    def quantity_of_interest(
        self,
        level: int,
        solution: np.ndarray,
        model_input: SyntheticModelInput,
    ) -> float:
        """Return the mean solution as the scalar quantity Q_level."""
        level_data = self._get_level(level)
        expected_shape = (level_data.size,)
        solution_vector = np.asarray(solution)

        if solution_vector.shape != expected_shape:
            raise ValueError(
                f"Solution for level {level} must have shape "
                f"{expected_shape}."
            )

        return float(np.mean(solution_vector))
