import cholespy
import numpy as np
import torch

from scipy.sparse import spmatrix, csr_matrix, identity


def get_matrix_precondition(a: csr_matrix, perturb: float, use_gpu: bool):
    device = torch.device("cuda" if use_gpu else "cpu")
    # Convert scipy sparse matrix to torch sparse matrix
    a_coo = a.tocoo()
    values = a_coo.data
    indices = np.vstack((a_coo.row, a_coo.col))
    a_torch = torch.sparse_coo_tensor(indices=indices, values=values, size=a_coo.shape,
                                      dtype=torch.float64, device=device).to_sparse_csr()

    # Create the incomplete Cholesky factorization
    #   Add beta*I to diagonals of A to ensure positive definiteness
    a_aug = a + identity(a.shape[0]) * perturb
    a_coo = a_aug.tocoo()
    rows = torch.tensor(a_coo.row, device=device)
    cols = torch.tensor(a_coo.col, device=device)
    data = torch.tensor(a_coo.data, device=device)
    solver = cholespy.CholeskySolverD(a_coo.shape[0], rows, cols, data, cholespy.MatrixType.COO)
    return a_torch, solver


def hybrid_conjugate_gradient(
        a_torch: torch.Tensor,
        x0: np.ndarray,
        b: np.ndarray,
        solver: cholespy.CholeskySolverD,
        tol: float,
        use_gpu: bool
):
    device = torch.device("cuda" if use_gpu else "cpu")
    x = torch.tensor(x0, device=device)
    b = torch.tensor(b, device=device)
    r = b - (a_torch @ x)
    # Initial guess leads to convergence
    rs_old = torch.dot(r, r)
    if torch.sqrt(rs_old) < tol:
        return x.cpu().numpy(), 0

    # One iteration of preconditioned conjugate gradient
    z = torch.ones_like(r, device=device)
    solver.solve(r, z)
    p = z.clone().detach()

    a_p = a_torch @ p
    r_z = torch.dot(r, z)
    alpha = r_z / torch.dot(p, a_p)
    x += alpha * p
    r -= alpha * a_p
    if torch.sqrt(torch.dot(r, r)) < tol:
        return x.cpu().numpy(), 0

    # Perform regular conjugate gradient method
    rs_old = torch.dot(r, r)
    r = b - (a_torch @ x)
    p = r.clone().detach()
    max_iter = len(b) * 10
    for i in range(max_iter):
        a_p = (a_torch @ p)
        alpha = rs_old / torch.dot(p, a_p)
        x += alpha * p
        r -= alpha * a_p
        rs_new = torch.dot(r, r)
        if torch.sqrt(torch.dot(r, r)) < tol:
            return x.cpu().numpy(), 0
        else:
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
    return x.cpu().numpy(), 1


def move_to_gpu(a: spmatrix, x0: np.ndarray, b: np.ndarray):
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    """ Move sparse matrix and vectors to GPU """
    a = csr_matrix(a)
    x0 = cp.array(x0)
    b = cp.array(b)
    return a, x0, b


def conjugate_gradient(a: spmatrix, x0: np.ndarray, b: np.ndarray, tol: float, use_gpu: bool):
    """ Solve Ax = b using conjugate gradient method """
    if use_gpu:
        a, x, b = move_to_gpu(a=a, x0=x0, b=b)
    max_iter = len(b) * 10
    x = x0.copy()
    r = b - a.dot(x)

    # Initial guess leads to convergence
    rs_old = np.dot(r, r)
    if np.sqrt(rs_old) < tol:
        return x, 0

    p = r.copy()

    for i in range(max_iter):
        a_p = a.dot(p)
        alpha = rs_old / np.dot(p, a_p)
        x += alpha * p
        r -= alpha * a_p
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            return x, 0
        else:
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
    return x, 1
