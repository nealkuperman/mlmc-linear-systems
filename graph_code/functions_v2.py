# =============================================================================
# functions_v2.py
# White Noise Sampling on Graphs — Updated Implementation
#
# Changes from functions.py:
#   - build_weighted_laplacian: now uses sparse CSR format
#   - solve_darcy: uses sparse slicing + spsolve (faster than dense)
#   - solve_darcy_amg: AMG preconditioned PCG (better for large graphs)
#   - monte_carlo_loop_tqdm: updated to use sparse pipeline
#   - extract_qoi: now uses abs(pressure difference)
#   - All other functions unchanged from functions.py
#
# Usage:
#   from functions_v2 import *
# =============================================================================

from tqdm import tqdm
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.io import mmread
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve, cg
import pyamg
from scipy.sparse import lil_matrix, diags
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky as sparse_cholesky

# =============================================================================
# DATA LOADING
# =============================================================================

def load_graph(filepath):
    """
    Load a graph from either a Matrix Market (.mtx) file or a plain edge list.
    Automatically detects format, delimiter (whitespace or comma), and whether
    edges are weighted. Automatically remaps non-contiguous or non-zero-starting
    node IDs to a contiguous 0-indexed range, since n_vertices is always based
    on the actual count of distinct nodes referenced — not the maximum ID seen.
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()

    if first_line.startswith('%%MatrixMarket'):
        sparse_matrix = mmread(filepath).tocsr()
        coo = sparse_matrix.tocoo()
        edges = []
        edge_weights = {}
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i < j:
                edges.append((int(i), int(j)))
                if v != 1.0:
                    edge_weights[(int(i), int(j))] = float(v)
        edge_weights = edge_weights if edge_weights else None

    else:
        edges = []
        edge_weights = {}
        weighted = None
        delimiter = None
        zero_indexed = None

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('%') or line.startswith('#'):
                    continue
                if delimiter is None:
                    delimiter = ',' if (',' in line and len(line.split()) == 1) else None
                parts = line.split(delimiter) if delimiter else line.split()
                parts = [p.strip() for p in parts if p.strip() != '']
                if weighted is None:
                    weighted = len(parts) == 3
                i_raw, j_raw = int(parts[0]), int(parts[1])
                if zero_indexed is None:
                    zero_indexed = (i_raw == 0 or j_raw == 0)
                if zero_indexed:
                    i, j = i_raw, j_raw
                else:
                    i, j = i_raw - 1, j_raw - 1
                if i == j:
                    continue
                i, j = min(i, j), max(i, j)
                edges.append((i, j))
                if weighted:
                    edge_weights[(i, j)] = float(parts[2])

        edges = list(set(edges))
        edge_weights = edge_weights if weighted else None

    # --- THIS IS THE REMAPPING BLOCK — check if your version has this ---
    all_nodes = sorted(set(v for e in edges for v in e))
    n_vertices = len(all_nodes)
    node_map = {old: new for new, old in enumerate(all_nodes)}

    edges = [(node_map[i], node_map[j]) for (i, j) in edges]
    if edge_weights is not None:
        edge_weights = {(node_map[i], node_map[j]): w for (i, j), w in edge_weights.items()}

    print(f"✓ Loaded: {filepath}")
    print(f"  Vertices : {n_vertices}")
    print(f"  Edges    : {len(edges)}")
    print(f"  Weighted : {'yes' if edge_weights else 'no'}")
    print()

    return edges, n_vertices, edge_weights


# check for connectivity and find number of components

def check_connectivity(edges, n_vertices):
    """
    Report the number of connected components in a graph. Use this
    right after load_graph() to decide whether filter_to_largest_component()
    needs to be called before proceeding to Phase 1 -- a disconnected
    graph produces a Laplacian with multiple zero eigenvalues, which
    breaks compute_lambda_min's k=2 eigenvalue request and can cause
    downstream Darcy solves to silently return NaN.

    Parameters
    ----------
    edges      : list of (i, j) tuples
    n_vertices : int

    Returns
    -------
    num_components : int
    """
    import networkx as nx

    G_nx = nx.Graph()
    G_nx.add_edges_from(edges)
    num_components = nx.number_connected_components(G_nx)

    print(f"Components: {num_components}")
    if num_components > 1:
        component_sizes = sorted((len(c) for c in nx.connected_components(G_nx)), reverse=True)
        print(f"  Component sizes: {component_sizes}")
        print(f"  -> Call filter_to_largest_component() before proceeding to Phase 1")
    else:
        print(f"  -> Graph is fully connected, safe to proceed")
    print()

    return num_components

# filter out the disconnected components
def filter_to_largest_component(edges, n_vertices):
    """
    Filter a graph down to its largest connected component, remapping
    node IDs to a contiguous 0-indexed range. Always run this after
    load_graph() and before Phase 1, since a disconnected graph produces
    a Laplacian with multiple zero eigenvalues -- this breaks
    compute_lambda_min's k=2 eigenvalue request (which expects exactly
    one zero eigenvalue plus lambda_min), and downstream Darcy solves
    can silently return NaN if gamma_in/gamma_out/interior end up split
    across disconnected pieces.

    If the graph is already fully connected, this is a no-op.

    Parameters
    ----------
    edges        : list of (i, j) tuples
    n_vertices   : int
    edge_weights : dict of {(i,j): weight} or None

    Returns
    -------
    edges        : list of (i, j) tuples, filtered and remapped
    n_vertices   : int -- vertex count of the largest component only
    edge_weights : dict or None, remapped to match
    """


    G_nx = nx.Graph()
    G_nx.add_edges_from(edges)
    num_components = nx.number_connected_components(G_nx)

    print(f"Components: {num_components}")

    if num_components == 1:
        print("✓ Graph is already fully connected — no filtering needed")
        print()
        return edges, n_vertices, edge_weights

    component_sizes = sorted((len(c) for c in nx.connected_components(G_nx)), reverse=True)
    print(f"  Component sizes: {component_sizes}")

    largest_cc = max(nx.connected_components(G_nx), key=len)
    G_sub = G_nx.subgraph(largest_cc).copy()

    all_nodes = sorted(largest_cc)
    node_map = {old: new for new, old in enumerate(all_nodes)}
    filtered_edges = [(node_map[i], node_map[j]) for i, j in G_sub.edges()]
    filtered_n_vertices = len(all_nodes)

    filtered_edge_weights = None

    print(f"✓ Filtered to largest component")
    print(f"  Vertices : {n_vertices} -> {filtered_n_vertices} "
          f"(dropped {n_vertices - filtered_n_vertices})")
    print(f"  Edges    : {len(edges)} -> {len(filtered_edges)}")
    print()

    return filtered_edges, filtered_n_vertices


# =============================================================================
# PHASE 1: Graph Setup
# =============================================================================

def build_graph_matrices(edges, n_vertices, edge_weights=None):
    """
    Build adjacency matrix A, degree matrix D, and graph Laplacian L = D - A
    using sparse CSR format for memory efficiency.

    For large graphs, dense n x n matrices are infeasible — an 11174 x 11174
    dense matrix requires ~953 MB RAM whereas sparse format requires only a few MB
    since most entries are zero.

    Parameters
    ----------
    edges        : list of (i, j) tuples
    n_vertices   : int
    edge_weights : dict of {(i,j): weight} or None for unweighted

    Returns
    -------
    A : scipy.sparse.csr_matrix (n x n) — adjacency matrix
    D : scipy.sparse.csr_matrix (n x n) — diagonal degree matrix
    L : scipy.sparse.csr_matrix (n x n) — graph Laplacian
    """
    

    A = lil_matrix((n_vertices, n_vertices))

    for (i, j) in edges:
        w = edge_weights.get((i,j), edge_weights.get((j,i), 1.0)) if edge_weights else 1.0
        A[i, j] = w
        A[j, i] = w

    A = A.tocsr()

    # degree = row sums of A
    degrees = np.array(A.sum(axis=1)).flatten()
    D = diags(degrees, format='csr')
    L = D - A

    print("✓ Phase 1 complete: A, D, L built as sparse matrices")
    print(f"  Matrix size : {n_vertices} x {n_vertices}")
    print(f"  Degree range: [{int(degrees.min())}, {int(degrees.max())}]")
    print(f"  Non-zeros in L: {L.nnz}")
    print()
    return A, D, L


# =============================================================================
# PHASE 2: Spectral Shift
# =============================================================================

def compute_lambda_min(L, D):
    """
    Compute the smallest positive eigenvalue of D^{-1}L by solving
    the generalized eigenvalue problem Lv = lambda Dv.

    For a connected graph, L has exactly one zero eigenvalue corresponding
    to the constant vector 1. We find only the 2 smallest eigenvalues using
    a sparse eigensolver (eigsh) — the first is zero and the second is
    lambda_min. This is significantly faster than computing all n eigenvalues,
    especially for large graphs.

    The eigenvalues are found using scipy.sparse.linalg.eigsh with which='SM'
    (smallest magnitude). Results are sorted ascending and filtered to remove
    the zero eigenvalue (anything < 1e-10 is treated as zero to handle
    floating point noise, e.g. -1e-15 instead of exactly 0).

    Parameters
    ----------
    L : np.ndarray (n x n) — graph Laplacian
    D : np.ndarray (n x n) — degree matrix

    Returns
    -------
    lambda_min : float — smallest positive eigenvalue of D^{-1}L,
                 i.e. the graph's fundamental frequency — the slowest
                 non-trivial mode of variation consistent with the
                 graph's connectivity structure

    Notes
    -----
    This relies on the graph being connected. A disconnected graph would
    have multiple zero eigenvalues and k=2 would not be sufficient.
    The sanity check assert len(positive) > 0 will catch this case.
    """
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csr_matrix

    L_sparse = csr_matrix(L)
    D_sparse = csr_matrix(D)

    # find only the 2 smallest eigenvalues
    # k=2 is sufficient for connected graphs — first is zero, second is lambda_min
    eigenvalues, _ = eigsh(L_sparse, k=2, M=D_sparse,
                           which='SM', tol=1e-10)

    # sort ascending and take abs to handle floating point noise
    # e.g. zero eigenvalue may come back as -1e-15
    eigenvalues = np.sort(np.abs(eigenvalues))

    # filter out zero eigenvalue
    positive = eigenvalues[eigenvalues > 1e-10]

    assert len(positive) > 0, \
        "No positive eigenvalues found — is the graph connected?"

    lambda_min = positive[0]

    print(f"✓ lambda_min = {lambda_min:.6f}")
    print(f"  (eigenvalues found: {np.round(eigenvalues, 8)})")
    return lambda_min


def build_shifted_laplacian(L, D, lambda_min, sigma_squared=0.25):
    """
    Build shifted Laplacian L_sigma = L + sigma^2 * lambda_min * D.
    Works with both dense and sparse L and D. Returns sparse (CSC) when
    given sparse input, for use with sparse Cholesky (scikit-sparse/
    CHOLMOD) in Phase 3 -- do not convert to dense, since that defeats
    the ~500x speedup validated on Oregon Router.
    """
    from scipy.sparse import issparse
    if issparse(L):
        L_sigma = (L + sigma_squared * lambda_min * D).tocsc()
    else:
        L_sigma = L + sigma_squared * lambda_min * D
    print(f"✓ Phase 2 complete: L_sigma built (sigma^2 = {sigma_squared})")
    print(f"  L_sigma type: {'sparse' if issparse(L_sigma) else 'dense'}")
    print()
    return L_sigma


def find_diameter_endpoints(G_nx, n_sample=5, k_hop=4):
    """
    Approximate the graph's diameter endpoints using the double-BFS
    heuristic, then expand each endpoint into a k-hop neighborhood to
    use as gamma_in / gamma_out.

    Double-BFS mechanism: from an arbitrary starting vertex, BFS finds
    the farthest reachable vertex (call it v). A second BFS, starting
    from v, finds the farthest vertex from v (call it w). The distance
    between v and w closely approximates the graph's true diameter,
    without needing to check all-pairs shortest paths. Repeating this
    from several different starting vertices and keeping the best
    result found hedges against an unlucky first choice of start.

    NOTE: k_hop should be chosen relative to the graph's diameter, not
    as a fixed constant across datasets -- a k_hop that works well on
    a large-diameter graph can claim almost the entire graph on a
    small-diameter, hub-dominated one. Always check boundary fraction
    (see check_boundary_fraction) after calling this.

    Parameters
    ----------
    G_nx     : networkx.Graph
    n_sample : int -- number of different starting vertices to try
    k_hop    : int -- neighborhood radius around each endpoint

    Returns
    -------
    gamma_in   : list of int -- k-hop neighborhood around one endpoint
    gamma_out  : list of int -- k-hop neighborhood around the other endpoint
    diameter   : int -- the approximated diameter (distance between endpoints)
    """

    best_distance = 0
    best_endpoint_1, best_endpoint_2 = None, None

    start_vertices = list(G_nx.nodes())[:n_sample]

    for start in start_vertices:
        # First BFS: find the vertex farthest from `start`
        distances_from_start = nx.single_source_shortest_path_length(G_nx, start)
        farthest_from_start = max(distances_from_start, key=distances_from_start.get)

        # Second BFS: find the vertex farthest from that result
        distances_from_far_point = nx.single_source_shortest_path_length(G_nx, farthest_from_start)
        opposite_endpoint = max(distances_from_far_point, key=distances_from_far_point.get)
        candidate_distance = distances_from_far_point[opposite_endpoint]

        # Keep the best (longest) pair found across all starting attempts
        if candidate_distance > best_distance:
            best_distance = candidate_distance
            best_endpoint_1, best_endpoint_2 = farthest_from_start, opposite_endpoint

    # Expand each endpoint into a k-hop neighborhood
    gamma_in = set(nx.single_source_shortest_path_length(G_nx, best_endpoint_1, cutoff=k_hop).keys())
    gamma_out = set(nx.single_source_shortest_path_length(G_nx, best_endpoint_2, cutoff=k_hop).keys())

    # Remove any overlap (rare, but possible if k_hop is large relative to the diameter)
    gamma_in = gamma_in - gamma_out
    gamma_out = gamma_out - gamma_in

    return list(map(int, gamma_in)), list(map(int, gamma_out)), best_distance
# =============================================================================
# PHASE 3: White Noise Sampling Pipeline
# =============================================================================
def build_incidence_matrix(edges, n_vertices):
    """
    Build the UNSIGNED vertex-edge incidence matrix B (n x e) as a sparse CSR matrix.
    B[i, e] = +1 if vertex i is EITHER endpoint of edge e
    Both endpoints of every edge get +1 -- this avoids implying any
    physical direction on the (undirected) graph's edges.

    Using sparse format avoids memory issues for large graphs — a
    dense (11174 x 23409) matrix requires ~2.09 GB RAM whereas the
    sparse version requires only a few MB.

    Parameters
    ----------
    edges      : list of (i, j) tuples
    n_vertices : int

    Returns
    -------
    B : scipy.sparse.csr_matrix (n_vertices x n_edges)
    """
    from scipy.sparse import lil_matrix

    n_edges = len(edges)
    B = lil_matrix((n_vertices, n_edges))

    for idx, (i, j) in enumerate(edges):
        B[i, idx] = 1
        B[j, idx] = 1   # both endpoints get +1, not -1

    return B.tocsr()


def run_phase3(L_sigma, lambda_min, edges, n_vertices):
    """
    Run the full Phase 3 white noise sampling pipeline, using sparse
    Cholesky (scikit-sparse/CHOLMOD).

    Steps:
        3a. Draw edge noise:       w_e ~ N(0,1) for every edge
        3b. Map noise to vertices: f = B @ w
        3c. Solve for u:           L_sigma @ u = sqrt(lambda_min) * f

    L_sigma must be sparse (CSC). Factorized once for reuse in the MC loop.

    Returns
    -------
    u      : np.ndarray (n_vertices,) — spatially correlated field
    B      : scipy.sparse.csr_matrix — incidence matrix (reuse in MC loop)
    factor : sksparse.cholmod.Factor — cached factorization (reuse via .solve_A())
    """

    B = build_incidence_matrix(edges, n_vertices)
    factor = sparse_cholesky(L_sigma)

    w = np.random.randn(len(edges))
    f = B @ w
    u = factor.solve_A(np.sqrt(lambda_min) * f)

    print("✓ Phase 3 complete: correlated field u generated")
    print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")
    print()
    return u, B, factor


# =============================================================================
# PHASE 4: Build Random Edge Permeability
# =============================================================================

def compute_permeability(u, edges):
    """
    Convert vertex field u into edge permeabilities k.
    Vectorized — returns a plain array in the same order as `edges`,
    matching the order expected by build_weighted_laplacian and extract_qoi.

    Parameters
    ----------
    u     : np.ndarray (n_vertices,) — spatially correlated field from Phase 3
    edges : list of (i, j) tuples

    Returns
    -------
    k_vals : np.ndarray (n_edges,) — edge permeabilities, k_e > 0
    """
    edges_arr = np.array(edges)
    i_arr = edges_arr[:, 0]
    j_arr = edges_arr[:, 1]

    k_vals = np.exp((u[i_arr] + u[j_arr]) / 2)

    return k_vals


# =============================================================================
# PHASE 5a: Build Weighted Laplacian — Sparse CSR
# =============================================================================

def build_weighted_laplacian(edges, n_vertices, k):
    """
    Build the weighted graph Laplacian L_k from edge permeabilities.

    L_k = sum_{e=(i,j)} k_e * b_e * b_e^T

    Each edge (i,j) with permeability k_e contributes:
        L_k[i,i] += k_e,  L_k[j,j] += k_e,
        L_k[i,j] -= k_e,  L_k[j,i] -= k_e

    Built in COO format for efficiency (fast bulk construction), then
    converted to CSR for downstream solves.

    Parameters
    ----------
    edges      : list of (i, j) tuples
    n_vertices : int
    k          : np.ndarray (n_edges,) — edge permeabilities from Phase 4,
                 in the SAME ORDER as `edges` (see compute_permeability)

    Returns
    -------
    L_k : scipy.sparse.csr_matrix (n_vertices x n_vertices)
    """
    edges_arr = np.array(edges)
    i_arr = edges_arr[:, 0]
    j_arr = edges_arr[:, 1]

    k_vals = np.asarray(k)

    if len(k_vals) != len(edges):
        raise ValueError(
            f"k has {len(k_vals)} values but edges has {len(edges)} — "
            f"k must be in the same order as edges (see compute_permeability)."
        )
    if np.any(k_vals <= 0):
        raise ValueError(
            f"Found {np.sum(k_vals <= 0)} non-positive permeability value(s) — "
            f"all k_e must be > 0 for a physically valid Darcy flow problem."
        )

    row_idx = np.concatenate([i_arr, j_arr, i_arr, j_arr])
    col_idx = np.concatenate([i_arr, j_arr, j_arr, i_arr])
    diag_data = np.concatenate([k_vals, k_vals])
    offdiag_data = np.concatenate([-k_vals, -k_vals])
    data = np.concatenate([diag_data, offdiag_data])
    L_k = coo_matrix((data, (row_idx, col_idx)),
                      shape=(n_vertices, n_vertices)).tocsr()

    row_sums = np.array(L_k.sum(axis=1)).flatten()
    if not np.allclose(row_sums, 0, atol=1e-8):
        raise ValueError(
            f"L_k row sums are not zero (max deviation: {np.abs(row_sums).max():.2e}) — "
            f"this violates the graph Laplacian structural invariant."
        )
    if (L_k != L_k.T).nnz > 0:
        raise ValueError("L_k is not symmetric — check edge list for duplicate/conflicting entries.")

    return L_k





# =============================================================================
# PHASE 5b: Solve Darcy Flow
# Two options: direct sparse (spsolve) or AMG preconditioned PCG
# spsolve is faster for small-medium graphs (<5000 nodes)
# AMG is better for very large graphs (>50000 nodes)
# =============================================================================

def solve_darcy(L_k, n_vertices, gamma_in, gamma_out, p_in=1.0, p_out=0.0, debug=True):
    """
    Solve Darcy flow: fix pressure at gamma_in (p_in) and gamma_out (p_out),
    solve for pressure at all interior vertices.

    Parameters
    ----------
    L_k       : scipy.sparse.csr_matrix (n x n) — weighted Laplacian
    n_vertices: int
    gamma_in  : list of int — inlet vertex indices (p = p_in)
    gamma_out : list of int — outlet vertex indices (p = p_out)
    p_in      : float — inlet pressure (default 1.0)
    p_out     : float — outlet pressure (default 0.0)
    debug     : bool — if True, run NaN/Inf and maximum-principle sanity checks

    Returns
    -------
    p : np.ndarray (n_vertices,) — pressure at every vertex
    """
    boundary = set(gamma_in) | set(gamma_out)
    interior = np.array([v for v in range(n_vertices) if v not in boundary])

    if len(interior) == 0:
        raise ValueError("No interior vertices — gamma_in and gamma_out cover the entire graph.")

    p_boundary = np.zeros(n_vertices)
    for v in gamma_in:
        p_boundary[v] = p_in
    for v in gamma_out:
        p_boundary[v] = p_out

    L_interior = L_k[interior, :][:, interior]
    rhs        = -L_k[interior, :][:, list(boundary)] @ p_boundary[list(boundary)]

    p_interior = spsolve(L_interior, rhs)

    if debug:
        if np.any(np.isnan(p_interior)) or np.any(np.isinf(p_interior)):
            raise ValueError(
                "solve_darcy: NaN/Inf in interior pressures — check L_interior for "
                "singularity (possible disconnected interior region)."
            )

    p = np.zeros(n_vertices)
    p[interior] = p_interior
    for v in gamma_in:
        p[v] = p_in
    for v in gamma_out:
        p[v] = p_out

    if debug:
        lo, hi = min(p_in, p_out), max(p_in, p_out)
        tol = 1e-8
        if p[interior].min() < lo - tol or p[interior].max() > hi + tol:
            raise ValueError(
                f"solve_darcy: pressure out of physical bounds "
                f"[min={p[interior].min():.6f}, max={p[interior].max():.6f}] — "
                f"expected range [{lo}, {hi}]."
            )

    return p


def solve_darcy_amg(L_k, n_vertices, gamma_in, gamma_out, p_in=1.0, p_out=0.0):
    """
    Solve Darcy flow using AMG preconditioned conjugate gradient.
    Recommended for very large graphs (>50,000 nodes).
    Falls back to spsolve if CG does not converge.

    Parameters
    ----------
    L_k       : scipy.sparse.csr_matrix (n x n) — weighted Laplacian
    n_vertices: int
    gamma_in  : list of int — inlet vertex indices (p = p_in)
    gamma_out : list of int — outlet vertex indices (p = p_out)
    p_in      : float — inlet pressure (default 1.0)
    p_out     : float — outlet pressure (default 0.0)

    Returns
    -------
    p : np.ndarray (n_vertices,) — pressure at every vertex
    """
    p = np.zeros(n_vertices)
    for v in gamma_in:
        p[v] = p_in
    for v in gamma_out:
        p[v] = p_out

    boundary = set(gamma_in) | set(gamma_out)
    interior = np.array([v for v in range(n_vertices) if v not in boundary])

    L_interior = L_k[interior, :][:, interior].tocsr()
    rhs        = -L_k[interior, :][:, list(boundary)] @ p[list(boundary)]

    ml = pyamg.ruge_stuben_solver(L_interior)
    M  = ml.aspreconditioner()
    p_interior, info = cg(L_interior, rhs, M=M, rtol=1e-10)

    if info != 0:
        p_interior = spsolve(L_interior, rhs)

    p[interior] = p_interior
    return p


# =============================================================================
# PHASE 6: Quantity of Interest
# =============================================================================

def extract_qoi(p, k, edges, gamma_out):
    """
    Compute Q = sum of k_e * |p_i - p_j| for all edges incident to gamma_out.
    Absolute value ensures Q >= 0 regardless of edge orientation convention.
    By conservation of flow, Q equals total flux entering through gamma_in.

    Parameters
    ----------
    p         : np.ndarray (n_vertices,) — from Phase 5
    k         : dict of {(i,j): k_e} — from Phase 4
    edges     : list of (i, j) tuples
    gamma_out : list of int — outlet vertex indices

    Returns
    -------
    Q : float — total outlet flux
    """
    gamma_out_set = set(gamma_out)
    Q = 0.0
    for (i, j) in edges:
        k_e = k.get((i,j), k.get((j,i), 1.0))
        if i in gamma_out_set or j in gamma_out_set:
            Q += k_e * abs(p[i] - p[j])
    return Q


# =============================================================================
# PHASE 7: Monte Carlo Loop
# =============================================================================

def monte_carlo_loop_tqdm(L_sigma, lambda_min, edges, n_vertices,
                          gamma_in, gamma_out, edge_weights=None,
                          N=1000, use_amg=False, debug=True):
    """
    Run the full Monte Carlo Darcy flow pipeline (Phases 3-6) N times.

    Parameters
    ----------
    ... (existing params)
    debug : bool — if True (default), runs sanity checks each sample:
            (1) NaN/Inf check on the solved interior pressures,
            (2) maximum-principle bounds check (p in [0,1]).
            Set to False to skip these checks for a faster production run
            once correctness has been validated on a given dataset.
    """

    # precompute once
    B         = build_incidence_matrix(edges, n_vertices)
    cho_cache = cho_factor(L_sigma)
    Q_samples = np.zeros(N)
    # precompute edge arrays for vectorized permeability
    edges_arr = np.array(edges)
    i_arr     = edges_arr[:, 0]
    j_arr     = edges_arr[:, 1]
    # precompute COO structure for L_k — same every step, only data changes
    row_idx = np.concatenate([i_arr, j_arr, i_arr, j_arr])
    col_idx = np.concatenate([i_arr, j_arr, j_arr, i_arr])
    # precompute boundary/interior split once
    boundary   = set(gamma_in) | set(gamma_out)
    interior   = np.array([v for v in range(n_vertices)
                           if v not in boundary])
    p_boundary = np.zeros(n_vertices)
    for v in gamma_in:
        p_boundary[v] = 1.0
    for v in gamma_out:
        p_boundary[v] = 0.0
    gamma_out_set = set(gamma_out)

    with tqdm(total=N, desc="Monte Carlo", unit="sample") as pbar:
        for idx in range(N):
            # Phase 3
            w = np.random.randn(len(edges))
            f = B @ w
            u = cho_solve(cho_cache, np.sqrt(lambda_min) * f)
            # Phase 4 — vectorized
            k_vals = np.exp((u[i_arr] + u[j_arr]) / 2)
            # Phase 5a — build L_k using precomputed structure
            diag_data = np.concatenate([k_vals, k_vals])
            offdiag_data = np.concatenate([-k_vals, -k_vals])
            data = np.concatenate([diag_data, offdiag_data])
            L_k  = coo_matrix((data, (row_idx, col_idx)),
                               shape=(n_vertices, n_vertices)).tocsr()

            # Phase 5b — solve
            L_interior = L_k[interior, :][:, interior]
            rhs        = -L_k[interior, :][:, list(boundary)] @ p_boundary[list(boundary)]

            p_interior = spsolve(L_interior, rhs)

            if debug:
                if np.any(np.isnan(p_interior)) or np.any(np.isinf(p_interior)):
                    raise ValueError(
                        f"Sample {idx}: solve produced NaN/Inf in interior pressures — "
                        f"check L_interior for singularity (possible disconnected interior region)."
                    )

            p = np.zeros(n_vertices)
            p[interior] = p_interior
            for v in gamma_in:
                p[v] = 1.0
            for v in gamma_out:
                p[v] = 0.0

            if debug:
                tol = 1e-8
                if p[interior].min() < -tol or p[interior].max() > 1 + tol:
                    raise ValueError(
                        f"Sample {idx}: pressure out of physical bounds "
                        f"[min={p[interior].min():.6f}, max={p[interior].max():.6f}] — "
                        f"expected range [0, 1]."
                    )

            # Phase 6 — vectorized
            p_diff = np.abs(p[i_arr] - p[j_arr])
            outlet_mask = np.array([i in gamma_out_set or j in gamma_out_set
                                    for i,j in edges])
            Q_samples[idx] = np.sum(k_vals[outlet_mask] * p_diff[outlet_mask])
            pbar.update(1)
            pbar.set_postfix({"mean Q": f"{Q_samples[:idx+1].mean():.4f}"})

    print(f"\n✓ Done — {N} samples")
    print(f"  Mean Q : {Q_samples.mean():.6f}")
    print(f"  Std Q  : {Q_samples.std():.6f}")
    return Q_samples



def analyze_qoi(Q_samples, save_as=None):
    """
    Analyze and plot the distribution of Q across all Monte Carlo samples.
    Produces a histogram and convergence plot of the running mean.

    Parameters
    ----------
    Q_samples : np.ndarray (N,) — from monte_carlo_loop_tqdm
    save_as   : str or None — filename to save figure
    """
    print("\n=== Monte Carlo Results ===")
    print(f"  Samples : {len(Q_samples)}")
    print(f"  Mean Q  : {Q_samples.mean():.6f}")
    print(f"  Std Q   : {Q_samples.std():.6f}")
    print(f"  Var Q   : {Q_samples.var():.6f}")
    print(f"  Min Q   : {Q_samples.min():.6f}")
    print(f"  Max Q   : {Q_samples.max():.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

    # Histogram
    axes[0].hist(Q_samples, bins=40, color="#5bc8f5",
                 edgecolor="#0f1117", alpha=0.9)
    axes[0].axvline(Q_samples.mean(), color="red", linestyle="--",
                    linewidth=2, label=f"Mean = {Q_samples.mean():.4f}")
    axes[0].set_xlabel("Q (total flux)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of QoI")
    axes[0].legend(facecolor="#0f1117", labelcolor="white")

    # Convergence
    running_mean = np.cumsum(Q_samples) / np.arange(1, len(Q_samples)+1)
    axes[1].plot(running_mean, color="#5bc8f5", linewidth=1.5)
    axes[1].axhline(Q_samples.mean(), color="red", linestyle="--",
                    linewidth=2,
                    label=f"Final mean = {Q_samples.mean():.4f}")
    axes[1].set_xlabel("Number of samples")
    axes[1].set_ylabel("Running mean of Q")
    axes[1].set_title("Monte Carlo Convergence")
    axes[1].legend(facecolor="#0f1117", labelcolor="white")

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
        print(f"Saved as {save_as}")

    plt.show()


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_graph_2d(edges, n_vertices, edge_weights=None,
                       title="Graph", save_as=None):
    """
    2D graph visualization with degree-based node sizing and coloring.

    Parameters
    ----------
    edges        : list of (i, j) tuples
    n_vertices   : int
    edge_weights : dict or None
    title        : str
    save_as      : str or None — filename to save figure
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    if edge_weights:
        for (i, j) in edges:
            w = edge_weights.get((i,j), edge_weights.get((j,i), 1.0))
            G.add_edge(i, j, weight=w)
    else:
        G.add_edges_from(edges)

    degrees     = dict(G.degree())
    node_sizes  = [300 + degrees[v] * 100 for v in G.nodes()]
    node_colors = [degrees[v] for v in G.nodes()]
    pos         = nx.spring_layout(G, seed=42, k=2)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    if edge_weights:
        weights = [G[i][j]['weight'] for i,j in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=weights,
                               edge_cmap=plt.cm.plasma, width=1.5,
                               alpha=0.6, edge_vmin=min(weights),
                               edge_vmax=max(weights))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                   norm=plt.Normalize(
                                       vmin=min(weights),
                                       vmax=max(weights)))
        cb = plt.colorbar(sm, ax=ax, shrink=0.5)
        cb.set_label("Edge Weight", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="white",
                               width=0.8, alpha=0.3)

    nc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                node_color=node_colors, cmap=plt.cm.cool,
                                alpha=0.95)
    cb2 = plt.colorbar(nc, ax=ax, shrink=0.5)
    cb2.set_label("Degree", color="white")
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")

    if n_vertices <= 100:
        nx.draw_networkx_labels(G, pos, ax=ax,
                                font_color="white", font_size=7)

    ax.set_title(title, color="white", fontsize=16, pad=20)
    ax.axis("off")
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
        print(f"Saved as {save_as}")

    plt.show()