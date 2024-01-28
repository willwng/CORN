from typing import Callable

import numpy as np
from scipy.optimize import line_search
from scipy.sparse import csr_matrix

from energyminimization.solvers.conjugate_gradient import conjugate_gradient


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


def line_search_newton_cg(fun, x0, x_tol=1e-5, jac=None, hess=None):
    x = x0.copy()
    max_iter = len(x0) * 100
    cg_max_iter = len(x0) * 20

    f_old = fun(x)
    f_old_old = None

    for k in range(max_iter):
        g = jac(x)
        g_mag = np.add.reduce(np.abs(g))
        # Stopping criteria for conjugate gradient methods
        tol = np.min([0.5, np.sqrt(g_mag)]) * g_mag
        float64eps = np.finfo(np.float64).eps

        z = np.zeros_like(x)
        r, d, p = g.copy(), -g, -g
        rs = np.dot(r, r)
        rs_old = rs

        A = hess(x)
        p = np.zeros_like(x)

        for j in range(cg_max_iter):
            if np.add.reduce(np.abs(r)) <= tol:
                p = z
                break
            dBd = np.dot(d, (A.dot(d)))
            # Curvature is too small
            if 0 <= dBd <= 3 * float64eps:
                break
            elif dBd <= 0:
                if j == 0:
                    p = -g
                    break
                else:
                    p = z
                    break

            alpha = rs / dBd
            r += alpha * (A.dot(d))
            z += alpha * d

            rs = np.dot(r, r)
            beta = rs / rs_old
            d = -r + beta * d

            rs_old = rs
        else:
            return x, 1
        alpha, _, gc, f_old, f_old_old, _ = line_search(fun, jac, x, p, g, f_old, f_old_old)
        update = alpha * p
        x += update
        if np.add.reduce(np.abs(update)) < x_tol:
            return x, 0
    return x, 1

