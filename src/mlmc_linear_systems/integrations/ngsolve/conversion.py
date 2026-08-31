from scipy import sparse
import ngsolve as ng

def ng_matrix_to_csr(
    matrix: ng.BaseMatrix,
) -> sparse.csr_matrix:
    """Convert an explicit NGSolve matrix to SciPy CSR format."""
    rows, cols, vals = matrix.COO()
    height, width = matrix.shape
    return sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(height, width),
    ).tocsr()


def bilinear_form_to_csr(
    form: ng.BilinearForm,
) -> sparse.csr_matrix:
    """Convert an assembled NGSolve bilinear form to SciPy CSR format."""
    matrix = form.mat
    return ng_matrix_to_csr(matrix)
