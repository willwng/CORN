"""
Central forces with non-linear effects (uses full form of 1/2k(l-l0)^2)
"""
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix

import energyminimization.energies.stretch_helper as sh


def compute_lengths(
        u_node_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the vector between each node (c_ij), the length of each bond, and the difference between the
        current length and the rest length
    :param u_node_matrix: Node displacement matrix
    :param r_matrix: matrix of bond vectors for ORIGINAL configuration
    :param active_bond_indices: information associated with active bonds
    :param active_bond_lengths: ORIGINAL lengths of active bonds
    :return: c_matrix (contains list of c_ij for each bond), length_matrix (length of each bond), d_matrix
    """
    u_node_matrix = u_node_matrix.reshape(-1, 2)
    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, -1]

    # c_ij points from j to i based on their current positions, length_matrix is the length of c_ij, d is (l-l_0)
    c_matrix = np.subtract(r_matrix[idx], np.subtract(u_node_matrix[i, :], u_node_matrix[j, :]))
    length_matrix = np.linalg.norm(c_matrix, axis=1)
    d_matrix = length_matrix - active_bond_lengths[idx]
    return c_matrix, length_matrix, d_matrix


def get_nonlinear_stretch_hessian(
        stretch_mod: float,
        u_node_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> csr_matrix:
    # Compute the current vectors associated with the position
    c_matrix, length_matrix, d_matrix = compute_lengths(u_node_matrix, r_matrix, active_bond_indices,
                                                        active_bond_lengths)

    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, -1]
    # Second derivative of energy with respect to same variable
    c_ij_sq = np.square(c_matrix)
    init_bond_lengths = active_bond_lengths[idx].reshape(-1, 1)
    curr_bond_lengths = length_matrix.reshape(-1, 1)
    # Formula is a * (1 - ((l0 * (c2_ij[!i]) / l^3))
    dd = stretch_mod * (1 - np.divide(np.multiply(init_bond_lengths, c_ij_sq), np.power(curr_bond_lengths, 3)))
    dxx, dyy = dd[:, 1], dd[:, 0]
    # Formula is a * (l0 * (c_ij[0] * c_ij[1]) / l^3)
    c_ij_prod = np.prod(c_matrix, axis=1).flatten()
    dxy = stretch_mod * np.divide(np.multiply(init_bond_lengths.flatten(), c_ij_prod),
                                  np.power(curr_bond_lengths.flatten(), 3))

    n_dof = u_node_matrix.size
    return sh.generate_hessian(active_bond_indices=active_bond_indices, dxx=dxx, dyy=dyy, dxy=dxy, n=n_dof)


def get_nonlinear_stretch_jacobian(
        stretch_mod: float,
        u_node_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> np.ndarray:
    c_matrix, length_matrix, d_matrix = compute_lengths(u_node_matrix, r_matrix, active_bond_indices,
                                                        active_bond_lengths)

    # Gradient is equal to -(a / l) * (l - l_0) * c_ij
    grad_factor = -stretch_mod * np.divide(d_matrix, length_matrix)
    grad = np.multiply(grad_factor.reshape(-1, 1), c_matrix)
    grad_x, grad_y = grad[:, 0], grad[:, 1]

    n_dof = u_node_matrix.size
    return sh.generate_jacobian(active_bond_indices=active_bond_indices, grad_x=grad_x, grad_y=grad_y, n=n_dof)


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

    _, _, d_matrix = compute_lengths(u_node_matrix, r_matrix, active_bond_indices, active_bond_lengths)
    energy = 0.5 * stretch_mod * np.square(d_matrix)
    return float(np.sum(energy))
