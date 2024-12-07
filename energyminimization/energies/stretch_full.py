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
    :return:
        - c_ij: vector between nodes j and i for each bond
        - length_ij: lengths of c_ij
        - d_ij: difference between current length and rest length
    """
    pos_matrix = pos_matrix.reshape(-1, 2)
    # Indices of the nodes for each bond
    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
    # c_ij points from i to j based on their current positions
    c_ij = pos_matrix[j] - pos_matrix[i]
    # Handle periodic boundary conditions
    hor_pbc, top_pbc = active_bond_indices[:, 2], active_bond_indices[:, 3]
    c_ij[np.where(hor_pbc == 1), 0] += corrections[0]
    c_ij[np.where(top_pbc == 1), 1] += corrections[1]
    # Lengths and differences
    idx = active_bond_indices[:, -1]
    length_ij = np.linalg.norm(c_ij, axis=1)
    d_ij = length_ij - active_bond_lengths[idx]
    return c_ij, length_ij, d_ij


def get_nonlinear_stretch_hessian(
        stretch_mod: np.ndarray,
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> csr_matrix:
    # Compute the current vectors associated with the position
    c_ij, l_ij, d_ij = compute_lengths(pos_matrix=pos_matrix, active_bond_indices=active_bond_indices,
                                       corrections=corrections, active_bond_lengths=active_bond_lengths)

    idx = active_bond_indices[:, -1]
    l0 = active_bond_lengths[idx]
    # Helpful values for computing second derivatives
    c_ij_sq = np.square(c_ij)
    c_ij_prod = np.prod(c_ij, axis=1)  # c_ij[0] * c_ij[1]
    l0_div_l3 = np.divide(l0, np.power(l_ij, 3))  # l0 / l^3

    # Formula for d^2E/d(x,y)^2 and alpha * (1 - ((l0 * (c2_ij[!i]) / l^3)), where !i is y if i is x and vice versa
    dd = np.multiply(stretch_mod[:, np.newaxis], (1 - np.multiply(l0_div_l3[:, np.newaxis], c_ij_sq)))
    dxx, dyy = dd[:, 1], dd[:, 0]
    # Formula for d^2E/dxdy is alpha * (l0 * (c_ij[0] * c_ij[1]) / l^3)
    dxy = np.multiply(stretch_mod, np.multiply(l0_div_l3, c_ij_prod))

    n_dof = pos_matrix.size
    return sh.generate_hessian(active_bond_indices=active_bond_indices, dxx=dxx, dyy=dyy, dxy=dxy, n=n_dof)


def get_nonlinear_stretch_jacobian(
        stretch_mod: np.ndarray,
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_bond_indices: np.ndarray,
        active_bond_lengths: np.ndarray
) -> np.ndarray:
    c_ij, l_ij, d_ij = compute_lengths(pos_matrix=pos_matrix, active_bond_indices=active_bond_indices,
                                       corrections=corrections, active_bond_lengths=active_bond_lengths)

    # Gradient is equal to -alpha * ((l - l_0) / l) * c_ij. First compute the pre-factor
    grad_factor = -np.multiply(stretch_mod, np.divide(d_ij, l_ij))[:, np.newaxis]
    # Then multiply the pre-factor by the c_ij vectors
    grad = np.multiply(grad_factor, c_ij)
    # The derivative with respect to x and y
    grad_x, grad_y = grad[:, 0], grad[:, 1]

    n_dof = pos_matrix.size
    return sh.generate_jacobian(active_bond_indices=active_bond_indices, grad_x=grad_x, grad_y=grad_y, n=n_dof)


def get_full_stretch_energy(
        stretch_mod: np.ndarray,
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

    _, _, d_ij = compute_lengths(pos_matrix=pos_matrix, active_bond_indices=active_bond_indices,
                                 corrections=corrections, active_bond_lengths=active_bond_lengths)
    # E = 1/2 * alpha * (l - l_0)^2
    energy = 0.5 * np.multiply(stretch_mod, np.square(d_ij))
    return np.sum(energy)
