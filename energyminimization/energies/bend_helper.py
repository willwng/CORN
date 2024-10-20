"""
Helper for bend energy calculations
see stretch_helper.py
"""
import numpy as np
from scipy.sparse import csr_matrix


def add_entry(rows, cols, data, i, r, c, v):
    rows[i], cols[i], data[i] = r, c, v


def generate_hessian(
        active_pi_indices: np.ndarray,
        dxx: np.ndarray,
        dyy: np.ndarray,
        dxy: np.ndarray,
        n: int
):
    j, i, k = active_pi_indices[:, 0], active_pi_indices[:, 1], active_pi_indices[:, 2]
    bond1_idx, pi_idx, sign = active_pi_indices[:, 3], active_pi_indices[:, 5], active_pi_indices[:, 6]
    num_pi_bonds = np.max(pi_idx) + 1
    rows, cols = np.zeros(36 * num_pi_bonds, dtype=np.uint32), np.zeros(36 * num_pi_bonds, dtype=np.uint32)
    data = np.zeros(36 * num_pi_bonds, dtype=np.float64)

    entry_idx = 36 * pi_idx
    x_i, x_j, x_k = 2 * i, 2 * j, 2 * k
    y_i, y_j, y_k = 2 * i + 1, 2 * j + 1, 2 * k + 1

    add_entry(rows, cols, data, entry_idx, x_j, x_j, 4 * dxx)
    add_entry(rows, cols, data, entry_idx + 1, y_j, y_j, 4 * dyy)
    add_entry(rows, cols, data, entry_idx + 2, x_i, x_i, dxx)
    add_entry(rows, cols, data, entry_idx + 3, y_i, y_i, dyy)
    add_entry(rows, cols, data, entry_idx + 4, x_k, x_k, dxx)
    add_entry(rows, cols, data, entry_idx + 5, y_k, y_k, dyy)
    # Derivative df^2/d_(x1,y1)
    add_entry(rows, cols, data, entry_idx + 6, x_j, y_j, -4 * dxy)
    add_entry(rows, cols, data, entry_idx + 7, y_j, x_j, -4 * dxy)
    add_entry(rows, cols, data, entry_idx + 8, x_i, y_i, -dxy)
    add_entry(rows, cols, data, entry_idx + 9, y_i, x_i, -dxy)
    add_entry(rows, cols, data, entry_idx + 10, x_k, y_k, -dxy)
    add_entry(rows, cols, data, entry_idx + 11, y_k, x_k, -dxy)
    # Derivative df^2/d_(x1,x2or3)
    add_entry(rows, cols, data, entry_idx + 12, x_j, x_i, -2 * dxx)
    add_entry(rows, cols, data, entry_idx + 13, x_i, x_j, -2 * dxx)
    add_entry(rows, cols, data, entry_idx + 14, x_j, x_k, -2 * dxx)
    add_entry(rows, cols, data, entry_idx + 15, x_k, x_j, -2 * dxx)
    add_entry(rows, cols, data, entry_idx + 16, x_i, x_k, dxx)
    add_entry(rows, cols, data, entry_idx + 17, x_k, x_i, dxx)
    # Derivative df^2/d_(y1,y2or3)
    add_entry(rows, cols, data, entry_idx + 18, y_j, y_i, -2 * dyy)
    add_entry(rows, cols, data, entry_idx + 19, y_i, y_j, -2 * dyy)
    add_entry(rows, cols, data, entry_idx + 20, y_j, y_k, -2 * dyy)
    add_entry(rows, cols, data, entry_idx + 21, y_k, y_j, -2 * dyy)
    add_entry(rows, cols, data, entry_idx + 22, y_i, y_k, dyy)
    add_entry(rows, cols, data, entry_idx + 23, y_k, y_i, dyy)
    # Derivative df^2/d_(x1,y2or3)
    add_entry(rows, cols, data, entry_idx + 24, x_j, y_i, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 25, y_i, x_j, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 26, x_j, y_k, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 27, y_k, x_j, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 28, x_i, y_j, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 29, y_j, x_i, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 30, x_i, y_k, -dxy)
    add_entry(rows, cols, data, entry_idx + 31, y_k, x_i, -dxy)
    add_entry(rows, cols, data, entry_idx + 32, x_k, y_j, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 33, y_j, x_k, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 34, x_k, y_i, -dxy)
    add_entry(rows, cols, data, entry_idx + 35, y_i, x_k, -dxy)

    return csr_matrix((data, (rows, cols)), shape=(n, n))

def generate_jacobian(
        active_pi_indices: np.ndarray,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        n: int
) -> np.ndarray:
    gradient = np.zeros(n)
    vertex, edge1, edge2 = active_pi_indices[:, 0], active_pi_indices[:, 1], active_pi_indices[:, 2]
    np.add.at(gradient, 2 * vertex, -2 * grad_x)
    np.add.at(gradient, 2 * vertex + 1, 2 * grad_y)
    # Force applied to the edge nodes (should be the same since they are 'rotating' around the center pivot)
    np.add.at(gradient, 2 * edge1, grad_x)
    np.add.at(gradient, 2 * edge1 + 1, -grad_y)
    np.add.at(gradient, 2 * edge2, grad_x)
    np.add.at(gradient, 2 * edge2 + 1, -grad_y)
    return gradient
