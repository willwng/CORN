from typing import Callable, Optional

import numpy as np
from scipy.sparse import spmatrix, csr_matrix


def conjugate_gradient(a: spmatrix, x0: np.ndarray, b: np.ndarray, tol: float):
    """ Solve Ax = b using conjugate gradient method """
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


def line_search(f: Callable[[np.ndarray], float], x: np.ndarray, d: np.ndarray) -> float:
    """
    Find the step size `alpha` that minimizes f(x + alpha * d)
    """
    alpha = 1.0
    f0 = f(x)
    while f(x + alpha * d) > f0:
        alpha *= 0.5
    return alpha


def armijo_line_search(f: Callable[[np.ndarray], float], x: np.ndarray, d: np.ndarray, c: float = 1e-4) -> float:
    """
    Find the step size `alpha` that satisfies the Armijo condition
    """
    alpha = 1.0
    f0 = f(x)
    d_2 = np.dot(d, d)
    while f(x + alpha * d) > f0 + c * alpha * d_2:
        alpha *= 0.5
    return alpha


def non_linear_conjugate_gradient(
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        df: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable[[np.ndarray], csr_matrix]] = None,
        atol: float = 1e-8
):
    """
    Solve min f(x) using nonlinear conjugate gradient method
    """
    max_iter = len(x0) * 100

    x = x0.copy()
    g = df(x)
    d = -g

    g_old = g.copy()
    d_old = d.copy()

    # Take an initial step
    alpha = armijo_line_search(f, x, d)
    x += alpha * d

    for i in range(max_iter):
        # Calculate the steepest descent direction
        g = df(x)
        if np.linalg.norm(g) < atol:
            return x, 0

        # If hessian is provided, compute beta using Daniel formula
        #   Otherwise, compute beta according to Polak-Ribiere
        if hess is not None:
            h = hess(x)
            beta = (g.T @ (h @ d_old)) / (d_old.T @ (h @ d_old.T))
        else:
            y = g - g_old
            beta = np.dot(g, y) / np.dot(g_old, g_old)

        # Update the conjugate direction
        d = -g + beta * d_old

        # Update the position, after line search
        alpha = armijo_line_search(f, x, d)
        x += alpha * d

        d_old = d
        g_old = g

    return x, 1
