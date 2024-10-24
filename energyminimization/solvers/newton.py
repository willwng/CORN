"""
This file contains several numerical optimization methods that are Newton or Quasi-Newton.
"""
import numpy as np
from scipy.optimize import line_search


def line_search_newton_cg(fun, x0, jac, hess, x_tol=1e-8):
    """
    Newton-CG method, which uses conjugate gradient to solve for the search direction.
    """
    x = x0.copy()
    max_iter = len(x0) * 100
    cg_max_iter = len(x0) * 20

    f_old = fun(x)
    f_old_old = None

    for k in range(max_iter):
        g, A = jac(x), hess(x)

        # Stopping criteria for conjugate gradient methods
        g_mag = np.linalg.norm(g)
        tol = np.min([0.5, np.sqrt(g_mag)]) * g_mag

        # Conjugate gradient steps to find step direction
        z0 = np.zeros_like(x)
        r, d, p = g.copy(), -g, -g
        p, cg_info = inner_newton_cg(z0=z0, g=g, r=r, d=d, A=A, tol=tol, cg_max_iter=cg_max_iter)

        if cg_info == 1:  # CG did not converge
            return x, 2

        # Line search
        alpha, _, gc, f_old, f_old_old, _ = line_search(fun, jac, x, p, g, f_old, f_old_old)
        if alpha is None:
            return x, 3
        update = alpha * p
        x += update

        # Termination condition
        if np.linalg.norm(update) < x_tol:
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
        if np.linalg.norm(r) < tol:
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


def compute_tau(z, d, trust_radius):
    """
    Solve the scalar quadratic equation ||z + t d|| == trust_radius.
    This is like a line-sphere intersection.
    Return the two values of t, ta and tb, such that ta <= tb.
    """
    a = np.dot(d, d)
    b = 2 * np.dot(z, d)
    c = np.dot(z, z) - trust_radius ** 2

    sqrt_discriminant = np.sqrt(b * b - 4 * a * c)
    aux = b + np.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    return sorted([ta, tb])


def tr_predict(f0, p, g, A):
    """
    Predict the value of the objective function at the next step
    """
    return f0 + np.dot(g, p) + 0.5 * np.dot(p, A.dot(p))


def tr_solve(f0, x, g, A, max_iter, tol, trust_radius):
    """
    Use conjugate gradient to solve for the search direction p in the trust region.
    """
    # Check if we are already at a minimum
    z0 = np.zeros_like(x)
    if np.linalg.norm(g) <= tol:
        return z0, 0

    # Initialize
    z = z0
    r = g.copy()
    d = -r

    # Iterate to solve for the search direction
    for j in range(max_iter):
        Ad = A.dot(d)
        dAd = np.dot(d, Ad)
        # Direction of non-positive curvature
        if dAd <= 0:
            ta, tb = compute_tau(z, d, trust_radius)
            pa, pb = z + ta * d, z + tb * d
            # Choose the direction with the lowest predicted objective function value
            if tr_predict(f0, pa, g, A) < tr_predict(f0, pb, g, A):
                return pa, 1
            else:
                return pb, 1

        r_squared = np.dot(r, r)
        alpha = r_squared / dAd
        z_new = z + alpha * d
        # Move back to the boundary of the trust region
        if np.linalg.norm(z_new) >= trust_radius:
            # We require tau >= 0
            _, tau = compute_tau(z, d, trust_radius)
            p = z + tau * d
            return p, 1

        # Update residual, check for convergence
        r_new = r + alpha * Ad
        r_squared_new = np.dot(r_new, r_new)
        if np.sqrt(r_squared_new) < tol:
            return z_new, 0
        beta = r_squared_new / r_squared
        d_new = -r_new + beta * d

        z = z_new
        r = r_new
        d = d_new

    else:
        return z, 2


def trust_region_newton_cg(fun, x0, jac, hess, g_tol=1e-8):
    # --- Initialize trust-region ---
    trust_radius = 1.0
    max_trust_radius = 1000.0
    eta = 0.15

    max_iter = len(x0) * 100
    cg_max_iter = len(x0) * 200

    x = x0.copy()
    f_old = fun(x)

    for k in range(max_iter):
        g = jac(x)
        g_mag = np.linalg.norm(g)
        # Termination condition
        if g_mag < g_tol:
            return x, 0

        # Try an initial step and trusting it (conjugate gradient to solve the subproblem)
        A = hess(x)
        cg_tol = min(0.5, np.sqrt(g_mag)) * g_mag
        p, hit_constraint = tr_solve(f0=f_old, x=x, g=g, A=A, max_iter=cg_max_iter, tol=cg_tol,
                            trust_radius=trust_radius)
        # Did not find a valid step
        if hit_constraint == 2:
            return x, 2
        x_new = x + p
        f_new = fun(x_new)

        # Actual reduction and predicted reduction
        df = f_old - f_new
        # Predicted reduction by the quadratic model
        df_pred = -(np.dot(g, p) + 0.5 * np.dot(p, A.dot(p)))

        # Updating trust region radius
        rho = df / df_pred
        if rho < 0.25:  # poor prediction, reduce trust radius
            trust_radius *= 0.25
        elif rho > 0.75 and hit_constraint:  # good step and on the boundary
            trust_radius = min(2 * trust_radius, max_trust_radius)

        # Accept step
        if rho > eta:
            x = x_new
            f_old = f_new

    return x, 1
