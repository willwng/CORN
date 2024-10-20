"""
Same as stretch_nonlinear.py, but uses positions instead of displacements and r matrices
"""
import numpy as np
from scipy.sparse import csr_matrix

import energyminimization.energies.stretch_helper as sh


def compute_lengths(
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the vector between each node (c_ij), the length of each bond, and the difference between the
        current length and the rest length
    :return: c_ji (contains vectors from node j to i for each bond), length_ji (length of each bond),
        d_ji (difference between current length and rest length)
    """
    pos_matrix = pos_matrix.reshape(-1, 2)
    # Indices of the nodes for each bond
    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
    # c_ij points from j to i based on their current positions
    c_ji = pos_matrix[j] - pos_matrix[i]
    # Handle periodic boundary conditions
    hor_pbc, top_pbc = active_bond_indices[:, 2], active_bond_indices[:, 3]
    c_ji[np.where(hor_pbc == 1), 0] += corrections[0]
    c_ji[np.where(top_pbc == 1), 1] += corrections[1]
    # Lengths and differences
    idx = active_bond_indices[:, -1]
    length_ji = np.linalg.norm(c_ji, axis=1)
    d_ji = length_ji - active_bond_lengths[idx]
    return c_ji, length_ji, d_ji


def get_nonlinear_stretch_hessian(
        stretch_mod: float,
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> csr_matrix:
    # Compute the current vectors associated with the position
    c_ji, l_ji, d_matrix = compute_lengths(pos_matrix=pos_matrix, active_bond_indices=active_bond_indices,
                                           corrections=corrections, active_bond_lengths=active_bond_lengths)

    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, -1]
    # Second derivative of energy with respect to same variable
    c_ij_sq = np.square(c_ji)
    init_bond_lengths = active_bond_lengths[idx].reshape(-1, 1)
    curr_bond_lengths = l_ji.reshape(-1, 1)
    # Formula is a * (1 - ((l0 * (c2_ij[!i]) / l^3)), where !i is y if i is x and vice versa
    dd = stretch_mod * (1 - np.divide(np.multiply(init_bond_lengths, c_ij_sq), np.power(curr_bond_lengths, 3)))
    dxx, dyy = dd[:, 1], dd[:, 0]
    # Formula is a * (l0 * (c_ij[0] * c_ij[1]) / l^3)
    c_ij_prod = np.prod(c_ji, axis=1).flatten()
    dxy = stretch_mod * np.divide(np.multiply(init_bond_lengths.flatten(), c_ij_prod),
                                  np.power(curr_bond_lengths.flatten(), 3))

    n_dof = pos_matrix.size
    return sh.generate_hessian(active_bond_indices=active_bond_indices, dxx=dxx, dyy=dyy, dxy=dxy, n=n_dof)


def get_nonlinear_stretch_jacobian(
        stretch_mod: float,
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> np.ndarray:
    c_ji, l_ji, d_matrix = compute_lengths(pos_matrix=pos_matrix, active_bond_indices=active_bond_indices,
                                           corrections=corrections, active_bond_lengths=active_bond_lengths)

    # Gradient is equal to -alpha * ((l - l_0) / l) * c_ij. Compute the pre-factor
    grad_factor = -stretch_mod * np.divide(d_matrix, l_ji)
    # Then multiply the pre-factor by the c_ij vectors
    grad = np.multiply(grad_factor.reshape(-1, 1), c_ji)
    # The derivative with respect to x has only the x component. Same for y
    grad_x, grad_y = grad[:, 0], grad[:, 1]

    n_dof = pos_matrix.size
    return sh.generate_jacobian(active_bond_indices=active_bond_indices, grad_x=grad_x, grad_y=grad_y, n=n_dof)


def get_full_stretch_energy(
        stretch_mod: float,
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> float:
    """
    Calculates the energy of the system due to the non-linear (full form) stretching of the bonds
    :param stretch_mod: Stretch modulus
    :param pos_matrix: positions of the nodes (shape: (n, 2))
    :param corrections: [correction_x, correction_y] corrections for periodic boundary conditions (shape: (2,))
    :param active_bond_indices: information associated with active bonds
    :param active_bond_lengths: ORIGINAL lengths of active bonds
    """

    _, _, d_ji = compute_lengths(pos_matrix=pos_matrix, active_bond_indices=active_bond_indices,
                                 corrections=corrections, active_bond_lengths=active_bond_lengths)
    energy = 0.5 * stretch_mod * np.square(d_ji)
    return float(np.sum(energy))
