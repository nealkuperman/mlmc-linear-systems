"""
graph_mlmc_model.py

Adapts the graph-based white-noise-sampling two-level pipeline
(functions_v2.py, two_level_mc.py) to the team's MultilevelModel
protocol (mlmc_model.py), so both the graph-based approach and the
PDE-based approach can be run through the same MLMCRunner and compared
directly using the same statistics, reproducibility, and reporting
machinery.

-------------------------------------------------------------------
WHAT THIS FILE DOES NOT CHANGE
-------------------------------------------------------------------
No graph-domain logic is altered. Aggregation (build_capped_aggregation),
the boundary-fraction precondition, and the sparse-Cholesky solve are
all preserved exactly as validated in two_level_mc.py / functions_v2.py.
This file is purely an adapter layer -- it re-packages existing
functions to satisfy the MultilevelModel Protocol's four methods.

-------------------------------------------------------------------
TWO THINGS THIS ADAPTER DELIBERATELY WORKS AROUND, WORTH DISCUSSING
WITH THE TEAM BEFORE TREATING THIS AS A FINAL MERGE:
-------------------------------------------------------------------
1. Sparse Cholesky with cached factorization is NOT available through
   their linear_solver.py (only dense `solve` / sparse `spsolve` via
   LU). The white-noise solve (Phase 3) is therefore performed here,
   inside couple_inputs(), using the existing scikit-sparse factor
   directly -- OUTSIDE their solve_linear_system() call path. Only
   the Darcy flow solve (fine and coarse) goes through their solver.
   This preserves the ~500x validated speedup on large graphs, but
   means this adapter does not exercise their solver abstraction for
   every linear solve in the pipeline, only the Darcy step.

2. number_of_levels is fixed at 2 (level 0 = coarse, level 1 = fine).
   The aggregation logic could in principle be applied recursively to
   build a true multi-level hierarchy (coarsen the coarse graph again,
   and again), which their framework already supports structurally --
   this adapter does not attempt that extension.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

from linear_solver import LinearSystem
from mlmc_model import CoupledInputs

# --- existing, unmodified project code ---
from two_level_mc import TwoLevelSetup


# ============================================================
# Model input: everything build_linear_system / quantity_of_interest
# need for ONE level (fine or coarse), for ONE sample
# ============================================================

@dataclass(frozen=True)
class GraphModelInput:
    """
    One level's sample-dependent data: this sample's permeability
    values, plus the fixed (never-changes-per-sample) graph structure
    needed to build that level's linear system and extract Q.
    """

    k_vals: np.ndarray          # this sample's edge permeabilities (fine or coarse)
    edges: list                 # edge list for this level
    n_vertices: int
    interior: np.ndarray
    boundary: list
    p_known: np.ndarray
    i_arr: np.ndarray
    j_arr: np.ndarray
    row_idx: np.ndarray
    col_idx: np.ndarray
    gamma_out_set: set


# ============================================================
# The adapter itself
# ============================================================

class GraphTwoLevelModel:
    """
    Adapts an existing TwoLevelSetup (built via TwoLevelSetup.build(),
    exactly as validated across Power Grid, Oregon Router, bio-grid-yeast,
    C. elegans, and Facebook ego network) to the MultilevelModel Protocol.

    Usage
    -----
    setup = TwoLevelSetup.build(edges, n_vertices, L_sigma, lambda_min,
                                 gamma_in, gamma_out,
                                 build_incidence_matrix, sparse_cholesky,
                                 max_size=10)
    model = GraphTwoLevelModel(setup)

    runner = MLMCRunner(model, base_seed=0)
    result = runner.run_fixed(samples_per_level=[2000, 300])
    print(result.estimate, result.standard_error)
    """

    def __init__(self, setup: TwoLevelSetup):
        self.setup = setup

    # ------------------------------------------------------------
    # 1. number_of_levels
    # ------------------------------------------------------------
    @property
    def number_of_levels(self) -> int:
        # Level 0 = coarse graph, Level 1 = fine graph.
        # See module docstring: this could be extended to >2 by
        # recursively coarsening, which is not attempted here.
        return 2

    # ------------------------------------------------------------
    # 2. sample_randomness
    # ------------------------------------------------------------
    def sample_randomness(
        self,
        fine_level: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw w ~ N(0,1) on the FINE graph's edges.

        Both levels' inputs are derived from this single draw in
        couple_inputs() -- this is what keeps Q_fine and Q_coarse
        correlated, which is the entire basis for the method's
        variance reduction.
        """
        return rng.standard_normal(len(self.setup.edges))

    # ------------------------------------------------------------
    # 3. couple_inputs
    # ------------------------------------------------------------
    def couple_inputs(self, fine_level, randomness):
        setup = self.setup
        w = randomness
    
        f = setup.B @ w
        u = setup.factor.solve_A(np.sqrt(setup.lambda_min) * f)
        k_vals_fine = np.exp((u[setup.i_arr] + u[setup.j_arr]) / 2)
    
        # Always derive coarse permeability -- it's cheap once u is known
        coarse_k_vals = np.array([
            sum(k_vals_fine[idx] for idx in setup.coarse_contribs[e])
            for e in setup.coarse_edges
        ])
        coarse_input = GraphModelInput(k_vals=coarse_k_vals, edges=setup.coarse_edges,
                                         n_vertices=setup.n_coarse, interior=setup.interior_coarse,
                                         boundary=list(setup.boundary_coarse), p_known=setup.p_known_coarse,
                                         i_arr=setup.i_arr_c, j_arr=setup.j_arr_c,
                                         row_idx=setup.row_idx_c, col_idx=setup.col_idx_c,
                                         gamma_out_set=setup.gamma_out_coarse_set)
    
        if fine_level == 0:
            # Coarsest level: "fine" input for the correction IS the coarse graph
            return CoupledInputs(fine=coarse_input, coarse=None)
    
        # fine_level == 1: correction is fine minus coarse
        fine_input = GraphModelInput(k_vals=k_vals_fine, edges=setup.edges, n_vertices=setup.n_vertices,
                                       interior=setup.interior, boundary=list(setup.boundary),
                                       p_known=setup.p_known, i_arr=setup.i_arr, j_arr=setup.j_arr,
                                       row_idx=setup.row_idx, col_idx=setup.col_idx,
                                       gamma_out_set=setup.gamma_out_set)
        return CoupledInputs(fine=fine_input, coarse=coarse_input)

    # ------------------------------------------------------------
    # 4. build_linear_system
    # ------------------------------------------------------------
    def build_linear_system(
        self,
        level: int,
        model_input: GraphModelInput,
    ) -> LinearSystem:
        """
        Build the reduced Darcy-flow system L_ii @ p_i = -L_ib @ p_b
        (Phase 5a + interior/boundary split of Phase 5b) for one level,
        given that level's permeability values.

        This step -- unlike the white-noise solve -- DOES go through
        the team's solve_linear_system(), since it is solved fresh
        every sample with no reusable factorization (L_k's entries
        change every sample; there is nothing to cache here).
        """
        k_vals = model_input.k_vals
        diag_data = np.concatenate([k_vals, k_vals])
        offdiag_data = np.concatenate([-k_vals, -k_vals])
        data = np.concatenate([diag_data, offdiag_data])

        L_k = coo_matrix(
            (data, (model_input.row_idx, model_input.col_idx)),
            shape=(model_input.n_vertices, model_input.n_vertices),
        ).tocsr()

        L_interior = L_k[model_input.interior, :][:, model_input.interior]
        rhs = -L_k[model_input.interior, :][:, model_input.boundary] \
            @ model_input.p_known[model_input.boundary]

        return LinearSystem(A=L_interior, b=rhs)

    # ------------------------------------------------------------
    # 5. quantity_of_interest
    # ------------------------------------------------------------
    def quantity_of_interest(
        self,
        level: int,
        solution: np.ndarray,
        model_input: GraphModelInput,
    ) -> float:
        """
        Reassemble the full pressure field from the solved interior
        values and the fixed boundary values (Phase 5b), then compute
        Q = sum of k_e * |p_i - p_j| over edges touching gamma_out
        (Phase 6).
        """
        p = np.zeros(model_input.n_vertices)
        p[model_input.interior] = solution
        boundary_indices = np.array(model_input.boundary)
        p[boundary_indices] = model_input.p_known[boundary_indices]

        p_diff = np.abs(p[model_input.i_arr] - p[model_input.j_arr])
        outlet_mask = np.array([
            i in model_input.gamma_out_set or j in model_input.gamma_out_set
            for i, j in model_input.edges
        ])

        return float(np.sum(model_input.k_vals[outlet_mask] * p_diff[outlet_mask]))
