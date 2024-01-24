"""
Central forces with non-linear effects (uses full form of 1/2k(l-l0)^2)
"""
from typing import Tuple

import numpy as np
import energyminimization.matrix_helper as pos


def get_bond_lengths(u_matrix: np.ndarray, r_matrix: np.ndarray, length_matrix: np.ndarray,
                     active_bond_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, 3]
    # c_ij points from j to i based on their current positions
    c_ij = np.subtract(r_matrix[idx], np.subtract(u_matrix[i, :], u_matrix[j, :]))
    distances = np.linalg.norm(c_ij, axis=1) - length_matrix[idx]
    return distances, c_ij


def get_nl_stretch_jacobian(u_matrix: np.ndarray, stretch_mod: float, r_matrix: np.ndarray, length_matrix: np.ndarray,
                            active_bond_indices: np.ndarray) -> np.ndarray:
    n = u_matrix.size
    gradient = np.zeros(n)

    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, 3]
    d, c_ij = get_bond_lengths(u_matrix=u_matrix, r_matrix=r_matrix, length_matrix=length_matrix,
                               active_bond_indices=active_bond_indices)

    # Direction of force
    c_ij_hat = c_ij / np.linalg.norm(c_ij, axis=1, keepdims=True)
    gradient_c = -stretch_mod * d
    grad_x = np.multiply(gradient_c, c_ij_hat[:, 0])
    grad_y = np.multiply(gradient_c, c_ij_hat[:, 1])

    np.add.at(gradient, 2 * i, grad_x)
    np.add.at(gradient, 2 * i + 1, grad_y)
    np.add.at(gradient, 2 * j, -grad_x)
    np.add.at(gradient, 2 * j + 1, -grad_y)

    return gradient


def get_nonlinear_stretch_energy(
        stretch_mod: float,
        u_node_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> float:
    u_node_matrix = u_node_matrix.reshape(-1, 2)

    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
    hor_pbc, top_pbc, idx = active_bond_indices[:, 2], active_bond_indices[:, 3], active_bond_indices[:, 4]
    u_matrix = u_node_matrix[i, :] - u_node_matrix[j, :]
    c_matrix = r_matrix[idx, :] - u_matrix[idx, :]
    d_matrix = np.linalg.norm(c_matrix, axis=1) - active_bond_lengths[idx]
    energy = 0.5 * stretch_mod * np.square(d_matrix)
    return float(np.sum(energy))
