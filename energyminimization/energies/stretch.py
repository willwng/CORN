"""
stretch.py is used to calculate the energy stored in a spring network from central forces
"""

import numpy as np
from scipy.sparse import csr_matrix


def add_entry(rows, cols, data, i, r, c, v):
    rows[i], cols[i], data[i] = r, c, v


def get_stretch_hessian(
        n: int,
        stretch_mod: float,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray
):
    """
    Returns the Hessian matrix associated with stretching
    (Currently uses a sparse matrix to save on memory usage
    and potentially increase speed).

    :param n: n is the number of nodes (size of hessian is d*n x d*n)
    :type n: int
    :param stretch_mod: alpha - stretching modulus
    :type stretch_mod: float
    :param r_matrix: matrix containing unit vectors between each node.
        Example: r_matrix[i][j] returns the
    unit vector [r_x, r_y] between nodes i and j
    :type r_matrix: Shape (n, n, 2) matrix
    :param active_bond_indices: List containing indices of [i, j]
    :type active_bond_indices: Shape (# bonds, 2) matrix
    :return: Hessian matrix size (n, n)
    """
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
    r_ij = r_matrix[idx]
    r_x, r_y = r_ij[:, 0], r_ij[:, 1]
    # Derivative df^2/d_x1^2 and df^2/d_y1^2
    dxx = stretch_mod * np.square(r_x)
    dyy = stretch_mod * np.square(r_y)
    dxy = stretch_mod * r_x * r_y
    add_entry(rows, cols, data, bond_idx, x_1, x_1, dxx)
    add_entry(rows, cols, data, bond_idx + 1, y_1, y_1, dyy)
    # Derivative df^2/d_x2^2 and df^2/d_y2^2
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
    return csr_matrix((data, (rows, cols)), shape=(2 * n, 2 * n))


def get_stretch_jacobian(
        stretch_mod: float,
        u_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray
) -> np.ndarray:
    """
    Returns the gradient [[fx_1, fy_1], [fx2, fy2]...] for stretching forces
    :param stretch_mod: alpha - stretch modulus
    :type stretch_mod: float
    :param u_matrix: matrix containing displacement vectors for each node
        compared to initial position
        Example: u_matrix[i] returns u_i
    :type u_matrix: Shape (n, 2) matrix
    :param r_matrix: matrix containing unit vectors between each node.
        Example: r_matrix[i][j] returns the
    unit vector [r_x, r_y] between nodes i and j
    :type r_matrix: Shape (n, n, 2) matrix
    :param active_bond_indices: List containing indices of [i, j]
    :type active_bond_indices: Shape (# bonds, 2) matrix
    :return: gradient for bending forces for each node
    :rtype: shape (2n,) vector where the forces are [fx_1, fy_1, fx_2, fy_2]
    """

    n = u_matrix.size
    gradient = np.zeros(n)

    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, -1]
    r_ij = r_matrix[idx]
    gradient_c = stretch_mod * np.einsum("nk, nk -> n", np.subtract(u_matrix[i, :], u_matrix[j, :]), r_ij)
    grad_x = np.multiply(gradient_c, r_ij[:, 0])
    grad_y = np.multiply(gradient_c, r_ij[:, 1])

    np.add.at(gradient, 2 * i, grad_x)
    np.add.at(gradient, 2 * i + 1, grad_y)
    np.add.at(gradient, 2 * j, -grad_x)
    np.add.at(gradient, 2 * j + 1, -grad_y)

    return gradient


def get_stretch_energies(
        stretch_mod: float,
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
    energy = 0.5 * stretch_mod * np.square(
        np.einsum("nk, nk -> n", np.subtract(u_matrix[i, :], u_matrix[j, :]), r_ij))
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
