"""
A helper script for generating Hessian and Jacobian matrices for stretch energy
    - Linear and nonlinear stretching have the same "structure" for these
"""
import numpy as np
from scipy.sparse import csr_matrix


def add_entry(rows, cols, data, i, r, c, v):
    rows[i], cols[i], data[i] = r, c, v


def generate_hessian(
        active_bond_indices: np.ndarray,
        dxx: np.ndarray,
        dyy: np.ndarray,
        dxy: np.ndarray,
        n: int
):
    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, -1]
    # The number of entries be equal to 16 * num_bonds
    #   due to 4 (second order derivatives) and 12 (6*2 second partial derivatives)
    num_bonds = np.max(idx) + 1
    bond_idx = 16 * idx
    # Storing the row, column, and data values for each entry
    rows, cols = np.zeros(16 * num_bonds, dtype=np.uint32), np.zeros(16 * num_bonds, dtype=np.uint32)
    data = np.zeros(16 * num_bonds, dtype=np.float64)

    x_1, x_2 = 2 * i, 2 * j
    y_1, y_2 = 2 * i + 1, 2 * j + 1
    add_entry(rows, cols, data, bond_idx, x_1, x_1, dxx)
    add_entry(rows, cols, data, bond_idx + 1, y_1, y_1, dyy)
    add_entry(rows, cols, data, bond_idx + 2, x_2, x_2, dxx)
    add_entry(rows, cols, data, bond_idx + 3, y_2, y_2, dyy)

    # Derivative df^2/(d_x1,d_y1) and df^2/(d_x2,d_y2)
    add_entry(rows, cols, data, bond_idx + 4, x_1, y_1, dxy)
    add_entry(rows, cols, data, bond_idx + 5, y_1, x_1, dxy)
    add_entry(rows, cols, data, bond_idx + 6, x_2, y_2, dxy)
    add_entry(rows, cols, data, bond_idx + 7, y_2, x_2, dxy)

    # Derivative df^2/(d_x1,d_x2) and df^2/(d_y1,d_y2)
    add_entry(rows, cols, data, bond_idx + 8, x_1, x_2, -dxx)
    add_entry(rows, cols, data, bond_idx + 9, x_2, x_1, -dxx)
    add_entry(rows, cols, data, bond_idx + 10, y_1, y_2, -dyy)
    add_entry(rows, cols, data, bond_idx + 11, y_2, y_1, -dyy)

    # Derivative df^2/(d_x1,d_y2) and df^2/(d_x2,d_y1)
    add_entry(rows, cols, data, bond_idx + 12, x_1, y_2, -dxy)
    add_entry(rows, cols, data, bond_idx + 13, y_2, x_1, -dxy)
    add_entry(rows, cols, data, bond_idx + 14, x_2, y_1, -dxy)
    add_entry(rows, cols, data, bond_idx + 15, y_1, x_2, -dxy)

    return csr_matrix((data, (rows, cols)), shape=(n, n))


def generate_jacobian(
        active_bond_indices: np.ndarray,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        n: int
) -> np.ndarray:
    gradient = np.zeros(n)
    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
    np.add.at(gradient, 2 * i, grad_x)
    np.add.at(gradient, 2 * i + 1, grad_y)
    np.add.at(gradient, 2 * j, -grad_x)
    np.add.at(gradient, 2 * j + 1, -grad_y)

    return gradient
