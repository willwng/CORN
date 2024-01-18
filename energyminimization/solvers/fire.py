from typing import Callable

import numpy as np


def md_integrate(v: np.ndarray, x: np.ndarray, force: np.ndarray, df: Callable[[np.ndarray], np.ndarray],
                 alpha: float, dt: float):
    velocity = v + 0.5 * dt * force
    velocity = (1 - alpha) * velocity + alpha * force * np.linalg.norm(velocity) / np.linalg.norm(force)
    x = x + dt * velocity
    force = -df(x)
    velocity = velocity + 0.5 * dt * force
    return x, velocity, force


def optimize_fire(x0: np.ndarray, df: Callable[[np.ndarray], np.ndarray], atol: float = 1e-8):
    # Initialize parameters
    alpha0 = 0.1
    n_delay = 5
    f_inc = 1.1
    f_dec = 0.5
    fa = 0.99
    dt = 0.002
    dt_max = 10 * dt
    dt_min = 0.02 * dt
    alpha = alpha0
    n_pos = 0

    # Termination conditions
    max_iter = x0.size * 100
    sqrt_dof = np.sqrt(x0.size)

    # Initialize x, velocity, and force
    x = x0.copy()
    velocity = np.zeros_like(x)
    force = -df(x)

    for _ in range(max_iter):
        power = np.dot(force, velocity)
        if power > 0:
            n_pos += 1
            if n_pos > n_delay:
                dt = min(dt * f_inc, dt_max)
                alpha *= fa
        else:
            n_pos = 0
            dt = max(dt * f_dec, dt_min)
            alpha = alpha0
            velocity = np.zeros_like(x)

        x, velocity, force = md_integrate(v=velocity, x=x, force=force, df=df, alpha=alpha, dt=dt)
        if (np.linalg.norm(force) / sqrt_dof) < atol:
            return x, 0

    return x, 1
