from abc import ABC, abstractmethod
from types import MappingProxyType

import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix


class BuildSystem(ABC):
    """
    Abstract base class for the implementations related to building the linear system of equations
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of BuildSystem.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def build_linear_system(self, flownetwork):
        """
        Build a linear system of equations for the pressure and update the system matrix and right hand side in
        flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class BuildSystemSparseCsc(BuildSystem):
    """
    Class for building a sparse linear system of equations (csc_matrix).
    """

    def build_linear_system(self, flownetwork):
        """
        Fast method to build a linear system of equation. The sparse system matrix is COOrdinate format. The right
        hand side vector is a 1d-numpy array. Accounts for pressure and flow boundary conditions. The system matrix
        and right hand side are updated in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        nr_of_vs = flownetwork.nr_of_vs
        transmiss = flownetwork.transmiss
        edge_list = flownetwork.edge_list
        # Generate row, col and data arrays required to build a coo_matrix.
        # In a first step, assume a symmetrical system matrix without accounting for boundary conditions.
        # Example: Network with 2 edges and 3 vertices. edge_list = [[v1, v2], [v2, v3]], transmiss = [T1, T2]
        # row = [v1, v2, v1, v2, v2, v3, v2, v3]
        # col = [v2, v1, v1, v2, v3, v2, v2, v3]
        # data = [-T1, -T1, T1, T1, -T2, -T2, T2, T2]
        row = np.concatenate((edge_list, edge_list), axis=1).reshape(-1)
        col = np.concatenate((np.roll(edge_list, 1, axis=1), edge_list), axis=1).reshape(-1)
        data = np.vstack([transmiss] * 4).transpose()  # Assign transmissibilities on diagonals and off-diagonals.
        data[:, :2] *= -1  # set off-diagonals to negative values.
        data = data.reshape(-1)

        # Initialise the right hand side vector.
        rhs = np.zeros(nr_of_vs)

        # Account for boundary conditions.
        boundary_vertices = flownetwork.boundary_vs
        boundary_values = flownetwork.boundary_val
        boundary_types = flownetwork.boundary_type  # 1: pressure, 2: flow rate

        # Pressure boundaries.
        # Set entire rows of the system matrix that represent pressure boundary vertices to 0.
        for vid, value, type in zip(boundary_vertices, boundary_values, boundary_types):
            if type == 1:  # If is a pressure boundary
                data[row == vid] = 0.
        # Add 1 to diagonal entries of system matrix that represent pressure boundary vertices.
        row = np.append(row, boundary_vertices[boundary_types == 1])
        col = np.append(col, boundary_vertices[boundary_types == 1])
        data = np.append(data, np.ones(np.size(boundary_vertices[boundary_types == 1])))

        # Assign pressure boundary value to right hand side vector.
        rhs[boundary_vertices[boundary_types == 1]] = boundary_values[boundary_types == 1]
        # Assign flow rate boundary value to right hand side vector.
        rhs[boundary_vertices[boundary_types == 2]] = boundary_values[boundary_types == 2]  # assign flow source term to rhs
        # Build the system matrix and assign the right hand side vector.
        flownetwork.system_matrix = csc_matrix((data, (row, col)), shape=(nr_of_vs, nr_of_vs))
        flownetwork.rhs = rhs

        diagnose_linear_system(flownetwork.system_matrix, flownetwork.rhs)


def diagnose_linear_system(A, b, *, tol=1e-20, verbose=True):
    """
    Diagnose why A x = b may have no (unique) solution or produce NaNs.

    Parameters
    ----------
    A : scipy.sparse matrix (square)
    b : ndarray
    tol : float
        Tolerance for rank / zero detection
    verbose : bool

    Returns
    -------
    report : dict
    """
    report = {}
    n, m = A.shape

    # -------------------------
    # 1. Basic sanity checks
    # -------------------------
    report["shape"] = (n, m)
    report["nnz"] = A.nnz
    report["finite_A"] = np.all(np.isfinite(A.data))
    report["finite_b"] = np.all(np.isfinite(b))

    # -------------------------
    # 2. Zero rows / columns
    # -------------------------
    row_sum = np.array(np.abs(A).sum(axis=1)).ravel()
    col_sum = np.array(np.abs(A).sum(axis=0)).ravel()

    zero_rows = np.where(row_sum < tol)[0]
    zero_cols = np.where(col_sum < tol)[0]

    report["zero_rows"] = zero_rows
    report["zero_cols"] = zero_cols

    # -------------------------
    # 3. Diagonal check
    # -------------------------
    diag = A.diagonal()
    small_diag = np.where(np.abs(diag) < tol)[0]
    report["small_diagonal"] = small_diag

    # -------------------------
    # 4. Symmetry check (for Laplacian-like systems)
    # -------------------------
    if n <= 2000:
        asym = A - A.T
        report["symmetry_error"] = np.max(np.abs(asym.data)) if asym.nnz else 0.0
    else:
        report["symmetry_error"] = None

    # -------------------------
    # 5. Rank deficiency (cheap estimate)
    # -------------------------
    rank_estimate = None
    try:
        if n <= 200:
            rank_estimate = np.linalg.matrix_rank(A.toarray(), tol=tol)
    except Exception:
        pass

    report["rank_estimate"] = rank_estimate
    if rank_estimate is not None:
        report["rank_deficiency"] = n - rank_estimate

    # -------------------------
    # 6. Test solve & residual
    # -------------------------
    try:
        x = spla.spsolve(A, b)
        residual = A @ x - b
        report["solve_success"] = True
        report["solution_has_nan"] = np.any(np.isnan(x))
        report["residual_norm"] = np.linalg.norm(residual)
    except Exception as e:
        report["solve_success"] = False
        report["exception"] = str(e)

    # -------------------------
    # 7. Print human-readable diagnosis
    # -------------------------
    if verbose:
        print("\n====== Linear System Diagnosis ======")
        print(f"Shape: {n} x {m}, nnz = {A.nnz}")
        print(f"Finite A: {report['finite_A']}, Finite b: {report['finite_b']}")

        if len(zero_rows):
            print(f"[!] Zero rows (< {tol}) detected: {zero_rows[:10]}{'...' if len(zero_rows)>10 else ''}")
        if len(zero_cols):
            print(f"[!] Zero cols (< {tol}) detected: {zero_cols[:10]}{'...' if len(zero_cols)>10 else ''}")

        if len(small_diag):
            print(f"[!] Near-zero diagonal entries {small_diag.shape[0]}: {small_diag[:10]}")

        if report["symmetry_error"] not in (None, 0.0):
            print(f"[!] Symmetry error max |A - Aᵀ| = {report['symmetry_error']:.2e}")

        if rank_estimate is not None:
            print(f"Rank estimate: {rank_estimate}/{n}")
            if rank_estimate < n:
                print(f"[!] Rank deficient by {n - rank_estimate}")

        if report.get("solve_success"):
            if report["solution_has_nan"]:
                print("[!] Solution contains NaN")
            print(f"Residual norm ||Ax-b|| = {report['residual_norm']:.2e}")
        else:
            print(f"[!] Solve failed: {report.get('exception')}")

        print("====================================\n")

    return report
