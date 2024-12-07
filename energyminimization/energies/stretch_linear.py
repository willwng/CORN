"""
stretch_linear.py is used to calculate the energy stored in a spring network from central forces
"""

import numpy as np
from scipy.sparse import csr_matrix

import energyminimization.energies.stretch_helper as sh


def get_stretch_hessian(
        n: int,
        stretch_mod: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray
) -> csr_matrix:
    """
    Returns the Hessian matrix associated with stretching
    (Currently uses a sparse matrix to save on memory usage
    and potentially increase speed).

    :param n: n is the number of nodes (size of hessian is d*n x d*n)
    :param stretch_mod: alpha - stretching modulus
    :param r_matrix: matrix containing unit vectors between each node.
        Example: r_matrix[idx] returns the unit vector [r_x, r_y] for bond with index [idx]
    :param active_bond_indices: List containing indices of [i, j]
    :return: Hessian matrix size (n, n)
    """
    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, -1]
    r_ij = r_matrix[idx]
    r_x, r_y = r_ij[:, 0], r_ij[:, 1]
    # Derivative df^2/d_x1^2 and df^2/d_y1^2
    dxx = np.multiply(stretch_mod, np.square(r_x))
    dyy = np.multiply(stretch_mod, np.square(r_y))
    dxy = np.multiply(stretch_mod, r_x * r_y)
    n_dof = 2 * n
    return sh.generate_hessian(active_bond_indices=active_bond_indices, dxx=dxx, dyy=dyy, dxy=dxy, n=n_dof)


def get_stretch_jacobian(
        stretch_mod: np.ndarray,
        u_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray
) -> np.ndarray:
    """
    Returns the gradient [[fx_1, fy_1], [fx2, fy2]...] for stretching forces
    :param stretch_mod: alpha - stretch modulus
    :param u_matrix: matrix containing displacement vectors for each node
        compared to initial position
        Example: u_matrix[i] returns u_i
    :param r_matrix: matrix containing unit vectors between each node.
        Example: r_matrix[i][j] returns the
    unit vector [r_x, r_y] between nodes i and j
    :param active_bond_indices: List containing indices of [i, j]
    :return: gradient for bending forces for each node
    :rtype: shape (2n,) vector where the forces are [fx_1, fy_1, fx_2, fy_2]
    """
    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, -1]
    r_ij = r_matrix[idx]
    gradient_c = np.multiply(stretch_mod,
                             np.einsum("nk, nk -> n", np.subtract(u_matrix[i, :], u_matrix[j, :]), r_ij))
    grad_x = np.multiply(gradient_c, r_ij[:, 0])
    grad_y = np.multiply(gradient_c, r_ij[:, 1])

    n_dof = u_matrix.size
    return sh.generate_jacobian(active_bond_indices=active_bond_indices, grad_x=grad_x, grad_y=grad_y, n=n_dof)


def get_stretch_energies(
        stretch_mod: np.ndarray,
        u_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray
) -> np.ndarray:
    """
    Returns the energy stretching forces

    :param stretch_mod: stretching modulus (alpha)
    :param u_matrix: displacement field matrix
    :type u_matrix: Shape (n, 2) matrix
    :param r_matrix: matrix containing unit vectors between each node
    :type r_matrix: Shape (n, n, 2) matrix
    :param active_bond_indices: List containing indices of [i, j]
    :type active_bond_indices: Shape (# bonds, 2) matrix
    """
    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, -1]
    r_ij = r_matrix[idx]
    energy = 0.5 * np.multiply(stretch_mod, np.square(
        np.einsum("nk, nk -> n", np.subtract(u_matrix[i, :], u_matrix[j, :]), r_ij)))
    return energy


def get_stretch_energy(
        stretch_mod: float,
        u_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray
) -> float:
    return float(np.sum(get_stretch_energies(stretch_mod=stretch_mod,
                                             u_matrix=u_matrix,
                                             r_matrix=r_matrix,
                                             active_bond_indices=active_bond_indices)))
