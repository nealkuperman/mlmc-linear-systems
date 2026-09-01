"""Representation of one NGSolve finite element level."""

import warnings
from dataclasses import dataclass, field

import ngsolve as ng
import numpy as np
from scipy import sparse

from .conversion import bilinear_form_to_csr


BoundaryValue = float | np.ndarray | ng.CoefficientFunction
DirichletValue = BoundaryValue | dict[str, BoundaryValue]


def _vector_numpy_view(
    vector: ng.BaseVector | np.ndarray,
) -> np.ndarray:
    """Return a writable NumPy view of a supported vector."""
    if isinstance(vector, ng.BaseVector):
        return vector.FV().NumPy()
    if isinstance(vector, np.ndarray):
        return vector
    raise TypeError(
        "vector must be an NGSolve BaseVector or NumPy array."
    )


def get_free_fixed_ids(
    fes: ng.FESpace,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the free and fixed degree-of-freedom indices."""
    free_dofs = fes.FreeDofs()
    free_mask = np.array(
        [
            bool(free_dofs[index])
            for index in range(fes.ndof)
        ],
        dtype=bool,
    )
    return np.flatnonzero(free_mask), np.flatnonzero(~free_mask)


def boundary_dof_ids(
    fes: ng.FESpace,
    name: str,
) -> np.ndarray:
    """Return indices of DOFs lying on the named boundary region.

    Parameters
    ----------
    fes : ngsolve.FESpace
        The finite element space.
    name : str
        Boundary name or regex, same syntax as the ``dirichlet=`` argument and
        ``mesh.Boundaries`` (e.g. ``"left"`` or ``"left|top"``).

    Returns
    -------
    np.ndarray
        Integer indices of the DOFs associated with that boundary. This is
        purely geometric membership; it does not depend on whether those DOFs
        are constrained (Dirichlet) or free.

    Notes
    -----
    A DOF on a shared corner belongs to every boundary it touches, so it will
    appear in the result for each of those boundary names.
    """

    boundary_dofs = fes.GetDofs(fes.mesh.Boundaries(name))
    boundary_mask = np.array(
        [
            bool(boundary_dofs[index])
            for index in range(fes.ndof)
        ],
        dtype=bool,
    )
    return np.flatnonzero(boundary_mask)


def apply_dirichlet(
    vector: ng.BaseVector | np.ndarray,
    fixed_ids: np.ndarray,
    values: float | np.ndarray = 0.0,
) -> ng.BaseVector | np.ndarray:
    """Write Dirichlet ``values`` into the ``fixed_ids`` entries of ``vector``.

    Generic, stateless helper: it does not know about any FE level, it only
    pins the given indices of a vector to the given values. 

    Parameters
    ----------
    vector : ngsolve.BaseVector or np.ndarray
        Vector to modify in place. NGSolve vectors are accessed via
        ``vector.FV().NumPy()``; anything else is treated as a NumPy array.
    fixed_ids : np.ndarray
        Indices of the (Dirichlet/fixed) DOFs to overwrite.
    values : float or np.ndarray, optional
        Either a scalar (broadcast to every fixed DOF) or a 1-D array of shape
        ``(len(fixed_ids),)`` giving one value per fixed DOF, in the same order
        as ``fixed_ids``. Defaults to ``0.0`` (homogeneous). A full-length
        ``(ndof,)`` array is *not* accepted; slice it yourself with
        ``values[fixed_ids]``.

    Returns
    -------
    The same ``vector`` object (for chaining), modified in place.

    Raises
    ------
    ValueError
        If ``values`` is an array whose shape is neither scalar nor
        ``(len(fixed_ids),)``.
    """

    vector_values = _vector_numpy_view(vector)
    boundary_values = np.asarray(values, dtype=float)
    number_of_fixed_dofs = len(fixed_ids)
    if (
        boundary_values.ndim != 0
        and boundary_values.shape != (number_of_fixed_dofs,)
    ):
        raise ValueError(
            "values must be a scalar or a one-dimensional array of "
            f"shape ({number_of_fixed_dofs},), but received "
            f"shape {boundary_values.shape}. If values contains one "
            "entry per global DOF, pass values[fixed_ids]."
        )
    vector_values[fixed_ids] = boundary_values
    return vector


@dataclass
class Level:
    """Store the assembled forms, solution, and transfers for one level.

    Parameters
    ----------
    mesh
        Mesh used by this level.
    fes
        Finite element space defined on ``mesh``.
    a
        Assembled bilinear form whose matrix is the level operator.
    f
        Assembled linear form whose vector is the level right-hand side.
    gfu
        Grid function used to store a solution on this level.
    free_ids, fixed_ids
        Free and constrained degrees of freedom in global ordering.
    P
        Transfer from the immediately coarser level to this level. It is
        ``None`` on a coarsest level.
    PT
        Transfer from this level to the immediately coarser level. It is
        ``None`` on a coarsest level.
    dirichlet_value
        Scalar, per-DOF array, coefficient function, or mapping from boundary
        names to any of those value types.
    dirichlet
        Boundary pattern used when constructing ``fes``.
    """

    mesh: ng.Mesh
    fes: ng.FESpace
    a: ng.BilinearForm
    f: ng.LinearForm
    gfu: ng.GridFunction
    free_ids: np.ndarray
    fixed_ids: np.ndarray
    P: ng.BaseMatrix | None = None
    PT: ng.BaseMatrix | None = None
    dirichlet_value: DirichletValue = 0.0
    dirichlet: str = ""
    _A_csr: sparse.csr_matrix | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _boundary_ids: dict[str, np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @staticmethod
    def transfer_available(
        fes: ng.FESpace,
    ) -> bool:
        """Return whether a coarser mesh level is available for transfer."""
        return fes.mesh.levels > 1

    @classmethod
    def from_forms(
        cls,
        mesh: ng.Mesh,
        fes: ng.FESpace,
        a: ng.BilinearForm,
        f: ng.LinearForm,
        *,
        P: ng.BaseMatrix | None = None,
        PT: ng.BaseMatrix | None = None,
        gfu: ng.GridFunction | None = None,
        build_prolongation: bool = True,
        dirichlet_value: DirichletValue = 0.0,
        dirichlet: str = "",
    ) -> "Level":
        """Construct one level from assembled NGSolve forms.

        When ``P`` is supplied or built automatically, its transpose is
        used as ``PT`` unless a restriction is supplied.
        """
        if gfu is None:
            gfu = ng.GridFunction(fes)

        free_ids, fixed_ids = get_free_fixed_ids(fes)

        if (
            P is None
            and build_prolongation
            and cls.transfer_available(fes)
        ):
            P = fes.Prolongation().CreateMatrix(
                fes.mesh.levels - 1
            )

        if PT is None and P is not None:
            PT = P.CreateTranspose()

        return cls(
            mesh=mesh,
            fes=fes,
            a=a,
            f=f,
            gfu=gfu,
            free_ids=free_ids,
            fixed_ids=fixed_ids,
            P=P,
            PT=PT,
            dirichlet_value=dirichlet_value,
            dirichlet=dirichlet,
        )

    @property
    def ndof(self) -> int:
        """Return the number of degrees of freedom on this level."""
        return int(self.fes.ndof)

    @property
    def A_csr(self) -> sparse.csr_matrix:
        """Return and cache the level operator in SciPy CSR format."""
        if self._A_csr is None:
            self._A_csr = bilinear_form_to_csr(self.a)
        return self._A_csr

    def refresh(self) -> None:
        """Clear data cached from the assembled bilinear form."""
        self._A_csr = None

    def dirichlet_ids(self, name: str) -> np.ndarray:
        """Return the fixed DOF indices on a named boundary."""
        if name not in self._boundary_ids:
            geometric_ids = boundary_dof_ids(self.fes, name)
            self._boundary_ids[name] = np.intersect1d(
                geometric_ids,
                self.fixed_ids,
            )
        return self._boundary_ids[name]

    def enforce_dirichlet(
        self,
        vector: ng.BaseVector | np.ndarray | None = None,
        values: DirichletValue | None = None,
    ) -> None:
        """Set boundary values without modifying free DOFs."""
        if values is None:
            values = self.dirichlet_value
        target = self.gfu.vec if vector is None else vector

        if isinstance(values, dict):
            self._enforce_dirichlet_dict(target, values)
        elif isinstance(values, ng.CoefficientFunction):
            self._project_cf_onto(
                target,
                values,
                self.dirichlet,
                self.fixed_ids,
            )
        else:
            apply_dirichlet(target, self.fixed_ids, values)

    def _enforce_dirichlet_dict(
        self,
        vector: ng.BaseVector | np.ndarray,
        specification: dict[str, BoundaryValue],
    ) -> None:
        """Apply named values and zero unspecified fixed DOFs."""
        vector_values = _vector_numpy_view(vector)

        assigned_mask = np.zeros(self.ndof, dtype=bool)
        assigned_values = np.zeros(self.ndof, dtype=float)

        for name, value in specification.items():
            ids = self.dirichlet_ids(name)
            if ids.size == 0:
                warnings.warn(
                    f"Dirichlet boundary {name!r} contains no fixed DOFs.",
                    stacklevel=3,
                )
                continue

            boundary_values = self._boundary_values(
                value,
                name,
                ids,
            )
            overlap = assigned_mask[ids]
            if overlap.any() and not np.allclose(
                assigned_values[ids][overlap],
                boundary_values[overlap],
            ):
                raise ValueError(
                    "Named Dirichlet boundaries prescribe conflicting "
                    "values at a shared degree of freedom."
                )

            assigned_mask[ids] = True
            assigned_values[ids] = boundary_values

        vector_values[self.fixed_ids] = assigned_values[self.fixed_ids]

        number_covered = np.count_nonzero(
            assigned_mask[self.fixed_ids]
        )
        if number_covered < self.fixed_ids.size:
            warnings.warn(
                "The Dirichlet specification does not cover every "
                "fixed DOF; unspecified fixed values were set to zero.",
                stacklevel=3,
            )

    def _boundary_values(
        self,
        value: BoundaryValue,
        boundary_name: str,
        ids: np.ndarray,
    ) -> np.ndarray:
        """Return one scalar value for every selected boundary DOF."""
        if isinstance(value, ng.CoefficientFunction):
            temporary = ng.GridFunction(self.fes)
            temporary.Set(
                value,
                definedon=self.mesh.Boundaries(boundary_name),
            )
            return temporary.vec.FV().NumPy()[ids].copy()

        boundary_values = np.asarray(value, dtype=float)
        if boundary_values.ndim == 0:
            return np.full(ids.size, float(boundary_values))
        if boundary_values.shape != (ids.size,):
            raise ValueError(
                "A named boundary array must have one value per selected "
                f"DOF; boundary {boundary_name!r} has {ids.size} DOFs, "
                f"but received shape {boundary_values.shape}."
            )
        return boundary_values

    def _project_cf_onto(
        self,
        vector: ng.BaseVector | np.ndarray,
        coefficient: ng.CoefficientFunction,
        region_name: str,
        ids: np.ndarray,
    ) -> None:
        """Project a coefficient function onto selected boundary DOFs."""
        temporary = ng.GridFunction(self.fes)
        pattern = region_name if region_name else ".*"
        temporary.Set(
            coefficient,
            definedon=self.mesh.Boundaries(pattern),
        )
        vector_values = _vector_numpy_view(vector)
        vector_values[ids] = temporary.vec.FV().NumPy()[ids]

    def set_initial_guess(
        self,
        value: ng.CoefficientFunction | float,
        *,
        enforce_boundary_conditions: bool = True,
    ) -> None:
        """Set the solution field and optionally enforce boundary values."""
        self.gfu.Set(value)
        if enforce_boundary_conditions:
            self.enforce_dirichlet()
