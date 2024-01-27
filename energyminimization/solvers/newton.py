from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix

from energyminimization.solvers.conjugate_gradient import conjugate_gradient, line_search


def newton(
        x: np.ndarray,
        df: Callable[[np.ndarray], np.ndarray],
        hess: Callable[[np.ndarray], csr_matrix]
):
    g = df(x)
    h = hess(x)
    # Solve for s = -H^{-1}g
    s, info = conjugate_gradient(h, np.zeros_like(g), -g, tol=1e-8, use_gpu=False)
    return s, g


def back_tracking_newton(
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        df: Callable[[np.ndarray], np.ndarray],
        hess: Callable[[np.ndarray], csr_matrix],
        atol: float = 1e-8
):
    x = x0.copy()
    f0 = f(x)
    alpha = 1.0
    c1 = 1e-4
    max_iter = len(x0) * 100

    s, g = newton(x, df, hess)
    for _ in range(max_iter):
        x_new = x + alpha * s
        # Check Armijo condition
        g_dot_s = np.dot(g, s)
        if f(x_new) <= f0 + c1 * alpha * g_dot_s:
            x = x_new
            # Reset step size and recompute Newton step
            alpha = 1.0
            s, g = newton(x, df, hess)
            f0 = f(x)

            # Check for convergence
            if np.linalg.norm(g) < atol:
                return x, 0

        else:
            alpha *= 0.5

    return x, 1

