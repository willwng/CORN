from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix


def test_gradient_hessian(
        x0: np.ndarray,
        f: Callable[[np.ndarray], float],
        df: Callable[[np.ndarray], np.ndarray],
        hess: Callable[[np.ndarray], csr_matrix],
):
    """
    Use finite differences to test implementations of the gradient and hessian of a function
    """
    n_tests = 500
    # Results contain the relative error of the finite difference approximation
    results_df = np.zeros(n_tests) # gradient
    results_d2f = np.zeros(n_tests) # hessian
    results_d2f2 = np.zeros((n_tests, x0.size)) # hessian using finite differences of gradient (assuming df is correct)

    for i in range(n_tests):
        x = x0
        u = np.random.rand(x0.size)  # Pick a random displacement, normalize it
        u /= np.linalg.norm(u)
        h = 1e-3  # Step size

        # Evaluate the function at x, x + h*u, x - h*u
        f0 = f(x)
        fp = f(x + h * u)
        fm = f(x - h * u)
        # Finite difference approximation of the gradient
        df_test = (fp - fm) / (2 * h)
        # Finite difference approximation of the hessian
        d2f_test = (fp - 2 * f0 + fm) / (h ** 2)

        # Obtain our implementation of the gradient and hessian
        df0 = df(x)
        h0 = hess(x)

        # Finite difference approximation of the hessian using the gradient implementation
        d2f_test2 = (df(x + h * u) - df(x - h * u)) / (2 * h)

        # Compute the relative error
        results_df[i] = np.abs(df_test - (df0.T @ u)) / (np.linalg.norm(df0.T @ u))
        results_d2f[i] = np.abs(d2f_test - (u.T @ h0 @ u)) / (np.linalg.norm(u.T @ h0 @ u))
        results_d2f2[i] = np.linalg.norm(d2f_test2 - (h0 @ u)) / (np.linalg.norm(h0 @ u))

    print(f"df_avg: {np.mean(results_df)}, df_max: {np.max(results_df)}")
    print(f"d2f_avg: {np.mean(results_d2f)}, d2f_max: {np.max(results_d2f)}")
    print(f"d2f2_avg: {np.mean(results_d2f2)}, d2f2_max: {np.max(results_d2f2)}")


# plot sparsity of hessian
def plot_sparsity(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    return ax
