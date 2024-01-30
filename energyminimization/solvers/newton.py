"""
This file contains several numerical optimization methods that are Newton or Quasi-Newton.
"""
from typing import Callable

import numpy as np
import torch
from scipy.optimize import line_search
from scipy.sparse import csr_matrix, diags, linalg

from energyminimization.solvers.conjugate_gradient import conjugate_gradient, get_matrix_precondition


def inexact_modified_chol(A: csr_matrix):
    """
    Inexact modified Cholesky factorization
    """
    # Compute T = diag(||Ae1||, ||Ae2||, ..., ||Aen||), where ei is the ith coordinate vector
    T_diag = linalg.norm(A, axis=1)
    T_sqrt_inv_diags = 1.0 / np.sqrt(T_diag)
    T_sqrt_inv = diags(T_sqrt_inv_diags)

    A_tilde = T_sqrt_inv @ A @ T_sqrt_inv
    beta = linalg.norm(A_tilde)
    # Compute a shift to ensure positive definiteness
    A_diag = A.diagonal()
    if np.min(A_diag) > 0:
        alpha = 0
    else:
        alpha = beta / 2

    for k in range(100):
        try:
            a_torch, solver = get_matrix_precondition(A_tilde, perturb=alpha, use_gpu=False)
            return a_torch, solver
        except ValueError:
            alpha = max(2 * alpha, beta / 2)

    return None, None


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


def line_search_newton_cg(fun, x0, jac, hess, x_tol=1e-8, pre: bool = False):
    """
    Inner loop for Newton-CG
    """
    x = x0.copy()
    max_iter = len(x0) * 100
    cg_max_iter = len(x0) * 20

    f_old = fun(x)
    f_old_old = None

    for k in range(max_iter):
        g, A = jac(x), hess(x)

        # Stopping criteria for conjugate gradient methods
        g_mag = np.add.reduce(np.abs(g))
        tol = np.min([0.5, np.sqrt(g_mag)]) * g_mag

        # Conjugate gradient steps to find step direction
        z0 = np.zeros_like(x)
        r, d, p = g.copy(), -g, -g
        if pre:
            # A, M = get_matrix_precondition(A, perturb=1e-8, use_gpu=False)
            _, M = inexact_modified_chol(A)
            p, cg_info = pre_inner_newton_cg(z0=z0, g=g, r=r, d=d, A=A, M=M, tol=tol, cg_max_iter=cg_max_iter)
            p = p.cpu().numpy()
        else:
            p, cg_info = inner_newton_cg(z0=z0, g=g, r=r, d=d, A=A, tol=tol, cg_max_iter=cg_max_iter)

        if cg_info == 1:  # CG did not converge
            return x, 2

        # Line search
        alpha, _, gc, f_old, f_old_old, _ = line_search(fun, jac, x, p, g, f_old, f_old_old)
        if alpha is None:
            return x, 3
        update = alpha * p
        x += update
        if np.add.reduce(np.abs(update)) < x_tol:
            return x, 0
    return x, 1


def inner_newton_cg(z0, g, r, d, A, tol, cg_max_iter):
    """
    Inner loop for Newton-CG, solves for the search direction p in the linear system
        Ap = -g, where A is the Hessian and g is the gradient.
    """
    z = z0.copy()
    float64eps = np.finfo(np.float64).eps

    rs_old = np.dot(r, r)
    for j in range(cg_max_iter):
        # Check for convergence
        if np.add.reduce(np.abs(r)) <= tol:
            return z, 0

        # Curvature is small
        dBd = np.dot(d, (A.dot(d)))
        if 0 <= dBd <= 3 * float64eps:
            return z, 0
        # z is both a descent direction and a direction of non-positive curvature
        elif dBd <= 0:
            if j == 0:
                return -g, 0
            else:
                return z, 0

        # Continue iterating
        alpha = rs_old / dBd
        r += alpha * (A.dot(d))
        z += alpha * d

        rs_new = np.dot(r, r)
        beta = rs_new / rs_old
        d = -r + beta * d

        rs_old = rs_new
    else:
        return z0, 1


def get_boundaries_intersections(z, d, trust_radius):
    """
    Solve the scalar quadratic equation ||z + t d|| == trust_radius.
    This is like a line-sphere intersection.
    Return the two values of t, sorted from low to high.
    """
    a = np.dot(d, d)
    b = 2 * np.dot(z, d)
    c = np.dot(z, z) - trust_radius ** 2
    sqrt_discriminant = np.sqrt(b * b - 4 * a * c)
    aux = b + np.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    return sorted([ta, tb])


def cg_steihaug(fun, x0, jac, hess, max_iter, tol, trust_radius):
    """
    Inner loop for Newton-CG, with trust region constraint
    """
    z = np.zeros_like(x0)
    r = jac(x0)
    d = -r

    if np.add.reduce(np.abs(r)) <= tol:
        return z, 0

    A = hess(x0)
    rs = np.dot(r, r)
    rs_old = rs

    for j in range(max_iter):
        dBd = np.dot(d, (A.dot(d)))
        # Direction of non-positive curvature
        if dBd <= 0:
            ta, tb = get_boundaries_intersections(z, d, trust_radius)
            pa = z + ta * d
            pb = z + tb * d
            if fun(x0 + pa) < fun(x0 + pb):
                return pa, 1
            else:
                return pb, 1

        alpha = rs / dBd
        z += alpha * d
        # Check if we have reached the trust region boundary
        if np.add.reduce(np.abs(z)) >= trust_radius:
            _, tau = get_boundaries_intersections(z, d, trust_radius)
            p = z + tau * d
            return p, 1

        r += alpha * (A.dot(d))
        rs_new = np.dot(r, r)
        if np.add.reduce(np.abs(r)) <= tol:
            return z, 0

        beta = rs_new / rs_old
        d = -r + beta * d
        rs_old = rs_new
    else:
        return z, 1


def pre_inner_newton_cg(z0, g, r, d, A, M, tol, cg_max_iter):
    """
    Inner loop for Newton-CG, with preconditioning
    """
    device = "cpu"
    # z is the resulting step direction
    z = torch.tensor(z0.copy(), device=device)
    r = torch.tensor(r, device=device)
    d = torch.tensor(d, device=device)
    float64eps = np.finfo(np.float64).eps

    rs_old = torch.dot(r, r)
    for j in range(cg_max_iter):
        if torch.sum(torch.abs(r)) <= tol:
            return z, 0
        dBd = d @ (A @ d)
        # Curvature is small
        if 0 <= dBd <= 3 * float64eps:
            return z, 0
        # z is both a descent direction and a direction of non-positive curvature
        elif dBd <= 0:
            if j == 0:
                return -g, 0
            else:
                return z, 0

        y = torch.ones_like(r)
        M.solve(r, y)
        # y = M @ r

        alpha = torch.dot(r, y) / dBd
        r += alpha * (A @ d)
        z += alpha * d

        rs_new = torch.dot(r, r)
        beta = rs_new / rs_old
        d = -r + beta * d

        rs_old = rs_new
    else:
        return torch.tensor(z0.copy(), device=device), 1
