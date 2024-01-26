"""
Central forces with non-linear effects (uses full form of 1/2k(l-l0)^2)
"""
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix


def compute_lengths(
        u_node_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_node_matrix = u_node_matrix.reshape(-1, 2)
    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, -1]

    # c_ij points from j to i based on their current positions
    c_matrix = np.subtract(r_matrix[idx], np.subtract(u_node_matrix[i, :], u_node_matrix[j, :]))
    # length of each bond
    length_matrix = np.linalg.norm(c_matrix, axis=1)
    # (l - l_0) for each bond
    d_matrix = length_matrix - active_bond_lengths[idx]
    return c_matrix, length_matrix, d_matrix


def add_entry(rows, cols, data, i, r, c, v):
    rows[i], cols[i], data[i] = r, c, v


def get_nonlinear_stretch_hessian(
        stretch_mod: float,
        u_node_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
):
    c_matrix, length_matrix, d_matrix = compute_lengths(u_node_matrix, r_matrix, active_bond_indices,
                                                        active_bond_lengths)

    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, -1]
    # The number of entries be equal to num_bonds * 16 (since for each bond,
    #   there are 4 (2*2) node dimensions)
    # Take 4 (second order derivatives) and 12 (6*2 second partial derivatives)
    num_bonds = np.max(idx) + 1
    rows, cols = np.zeros(16 * num_bonds, dtype=np.uint32), np.zeros(16 * num_bonds, dtype=np.uint32)
    data = np.zeros(16 * num_bonds, dtype=np.float64)

    bond_idx = 16 * idx
    x_1, x_2 = 2 * i, 2 * j
    y_1, y_2 = 2 * i + 1, 2 * j + 1
    nd = np.zeros(u_node_matrix.size)
    return csr_matrix((data, (rows, cols)), shape=(nd, nd))


def get_nonlinear_stretch_jacobian(
        stretch_mod: float,
        u_node_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> np.ndarray:
    gradient = np.zeros(u_node_matrix.size)
    c_matrix, length_matrix, d_matrix = compute_lengths(u_node_matrix, r_matrix, active_bond_indices,
                                                        active_bond_lengths)

    # Gradient is equal to -a * (l - l_0) / l * c_ij
    grad_factor = -stretch_mod * np.divide(d_matrix, length_matrix)
    grad = np.multiply(grad_factor.reshape(-1, 1), c_matrix)

    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
    grad_x, grad_y = grad[:, 0], grad[:, 1]
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
    """
    Calculates the energy of the system due to the non-linear (full form) stretching of the bonds
    :param stretch_mod: Stretch modulus
    :param u_node_matrix: Node displacement matrix
    :param r_matrix: matrix of bond vectors for ORIGINAL configuration
    :param active_bond_indices: information associated with active bonds
    :param active_bond_lengths: ORIGINAL lengths of active bonds
    """

    c_matrix, length_matrix, d_matrix = compute_lengths(u_node_matrix, r_matrix, active_bond_indices,
                                                        active_bond_lengths)
    energy = 0.5 * stretch_mod * np.square(d_matrix)
    return float(np.sum(energy))
