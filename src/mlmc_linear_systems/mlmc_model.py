"""Interfaces implemented by user-supplied MLMC models.

For a correction whose fine level is greater than zero, the runner performs:

    sample_randomness(fine_level, rng)
        |
        v
    couple_inputs(fine_level, randomness)
        |
        +-- fine input for fine_level
        |
        +-- coarse input for fine_level - 1
                    |
                    v
    build_linear_system(fine_level, fine input)
    build_linear_system(fine_level - 1, coarse input)
                    |
                    v
    solve both linear systems
                    |
                    v
    evaluate both quantities of interest
                    |
                    v
    correction = Q_fine_level - Q_(fine_level - 1)

At fine level zero, there is no coarse input or coarse solve:

    correction = Q_0
"""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np

from .linear_solver import LinearSystem


RandomnessT = TypeVar("RandomnessT")
ModelInputT = TypeVar("ModelInputT")


@dataclass(frozen=True)
class CoupledInputs(Generic[ModelInputT]):
    """Inputs at adjacent levels derived from one random realization.

    Attributes
    ----------
    fine
        Model input for the requested fine level.

    coarse
        Coupled model input for the preceding level. This must be ``None``
        when the requested fine level is zero.
    """

    fine: ModelInputT
    coarse: ModelInputT | None


class MultilevelModel(Protocol[RandomnessT, ModelInputT]):
    """Interface implemented by a user-supplied multilevel model.

    The model defines how randomness is generated and coupled, how each
    level-specific linear system is constructed, and how a solved system
    is reduced to a scalar quantity of interest.

    The model does not run the MLMC estimator. The runner calls these
    methods, solves the resulting systems, forms corrections, and
    accumulates statistics.
    """

    @property
    def number_of_levels(self) -> int:
        """Number of levels available to the MLMC runner."""
        ...

    def sample_randomness(
        self,
        fine_level: int,
        rng: np.random.Generator,
    ) -> RandomnessT:
        """Draw the shared randomness for one MLMC correction.

        Parameters
        ----------
        fine_level
            Fine level of the requested correction. For ``fine_level > 0``,
            the correction is

                Q_fine_level - Q_(fine_level - 1).

            For ``fine_level == 0``, the correction is simply ``Q_0``.

        rng
            Random-number generator supplied by the MLMC runner. The model
            should use this generator instead of creating an independently
            seeded generator.

        Returns
        -------
        RandomnessT
            Random information used to construct the coupled model inputs.

            For ``fine_level > 0``, it must contain enough information to
            construct inputs at both ``fine_level`` and ``fine_level - 1``.
            It may be a vector, field, matrix, tuple, or custom object.
        """
        ...

    def couple_inputs(
        self,
        fine_level: int,
        randomness: RandomnessT,
    ) -> CoupledInputs[ModelInputT]:
        """Construct adjacent-level inputs from one random realization.

        Parameters
        ----------
        fine_level
            Fine level of the requested correction.

        randomness
            Random object returned by ``sample_randomness()`` for the same
            fine level.

        Returns
        -------
        CoupledInputs[ModelInputT]
            Fine input for ``fine_level`` and coarse input for
            ``fine_level - 1``.

            The fine and coarse inputs may have different dimensions, but
            they must be derived from the same underlying randomness.

            The coarse input must be ``None`` when ``fine_level == 0``.

        Notes
        -----
        This method defines the coupling used by MLMC. It must not generate
        an independent random realization for the coarse input.
        """
        ...

    def build_linear_system(
        self,
        level: int,
        model_input: ModelInputT,
    ) -> LinearSystem:
        """Build one sample-dependent linear system at one level.

        Parameters
        ----------
        level
            Exact level of the system being constructed.

        model_input
            Level-specific input returned by ``couple_inputs()``. It contains
            the sampled coefficients, forcing data, or other information
            needed to construct the system.

        Returns
        -------
        LinearSystem
            Matrix, right-hand side, and optional initial guess representing

                A_level x_level = b_level.

        Notes
        -----
        This method constructs only one system. It does not draw randomness,
        construct the adjacent level, solve the system, or calculate the
        MLMC correction.
        """
        ...

    def quantity_of_interest(
        self,
        level: int,
        solution: np.ndarray,
        model_input: ModelInputT,
    ) -> float:
        """Calculate the scalar quantity of interest for one solution.

        Parameters
        ----------
        level
            Level at which the solution was calculated.

        solution
            Solution returned by the linear-system solver.

        model_input
            Same level-specific input used to construct the solved system.

        Returns
        -------
        float
            Scalar output ``Q_level``. The MLMC runner uses this value to
            form either ``Q_0`` or ``Q_level - Q_(level - 1)``.
        """
        ...
