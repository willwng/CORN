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
    n_tests = 500
    results_df = np.zeros(n_tests)
    results_d2f = np.zeros(n_tests)
    for i in range(n_tests):
        x = np.random.rand(x0.size)
        u = np.random.rand(x0.size)
        h = 1e-4
        fp = f(x + h * u)
        fm = f(x - h * u)
        f0 = f(x)
        df0 = df(x)
        h0 = hess(x)

        df_test = (fp - fm) / (2 * h)
        d2f_test = (fp - 2 * f0 + fm) / (h ** 2)
        results_df[i] = np.abs(df_test - (df0.T @ u)) / (np.linalg.norm(df0.T @ u))
        results_d2f[i] = np.abs(d2f_test - (u.T @ h0 @ u)) / (np.linalg.norm(u.T @ h0 @ u))

    print(f"df_avg: {np.mean(results_df)}, df_max: {np.max(results_df)}")
    print(f"d2f_avg: {np.mean(results_d2f)}, d2f_max: {np.max(results_d2f)}")


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
