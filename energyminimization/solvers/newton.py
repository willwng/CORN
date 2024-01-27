from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix

from energyminimization.solvers.conjugate_gradient import conjugate_gradient, line_search


def newton(
        x0: np.ndarray,
        df: Callable[[np.ndarray], np.ndarray],
        hess: Callable[[np.ndarray], csr_matrix]
):
    g = df(x0)
    h = hess(x0)
    s, _ = conjugate_gradient(h, np.ones_like(g), -g, tol=1e-8, use_gpu=False)
    return s, g


def line_search_newton(
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        df: Callable[[np.ndarray], np.ndarray],
        hess: Callable[[np.ndarray], csr_matrix]
):
    x = x0.copy()
    max_iter = len(x0) * 100

    for _ in range(max_iter):
        s, g = newton(x, df, hess)
        alpha = line_search(f, x, s)
        x += alpha * s
        if np.linalg.norm(g) < 1e-5:
            return x, 0

    return x, 1
