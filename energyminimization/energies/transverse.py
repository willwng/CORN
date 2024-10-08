"""
transverse.py is used to calculate the energy and the forces from transverse
"""

import numpy as np
from scipy.sparse import csr_matrix


def add_entry(rows, cols, data, i, r, c, v):
    rows[i], cols[i], data[i] = r, c, v


def get_transverse_hessian(
        n: int,
        tran_mod: float,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray
) -> csr_matrix:
    """
    Returns the Hessian matrix associated with transverse
    (Currently uses a sparse matrix to save on memory usage
    and potentially increase speed).

    :param n: n is the number of nodes (size of hessian is d*n x d*n)
    :type n: int
    :param tran_mod: mu - transverse modulus
    :type tran_mod: float
    :param r_matrix: matrix containing unit vectors between each node.
        Example: r_matrix[i][j] returns the
    unit vector [r_x, r_y] between nodes i and j
    :type r_matrix: Shape (n, n, 2) matrix
    :param active_bond_indices: List containing indices of [i, j]
    :type active_bond_indices: Shape (# bonds, 2) matrix
    :return: Hessian matrix size (n, n)
    """
    num_bonds = len(active_bond_indices)
    # The number of entries be equal to num_bonds * 16 (since for each bond,
    # there are 4 (2*2) node dimensions)
    # Take 4 (second order derivatives) and 12 (6*2 second partial derivatives)
    rows, cols = np.zeros(16 * num_bonds, dtype=np.uint32), np.zeros(16 * num_bonds, dtype=np.uint32)
    data = np.zeros(16 * num_bonds, dtype=np.float64)
    i, j, idx = active_bond_indices[:, 0], active_bond_indices[:, 1], active_bond_indices[:, 3]
    bond_idx = 16 * idx
    x_1, x_2 = 2 * i, 2 * j
    y_1, y_2 = 2 * i + 1, 2 * j + 1
    r_ij = r_matrix[idx]
    r_x, r_y = r_ij[:, 0], r_ij[:, 1]
    # Derivative df^2/d_x1^2 and df^2/d_y1^2
    add_entry(rows, cols, data, bond_idx, x_1, x_1, -tran_mod * (r_x ** 2 - 1))
    add_entry(rows, cols, data, bond_idx, y_1, y_1, -tran_mod * (r_y ** 2 - 1))
    # Derivative df^2/d_x2^2 and df^2/d_y2^2
    add_entry(rows, cols, data, bond_idx, x_2, x_2, -tran_mod * (r_x ** 2 - 1))
    add_entry(rows, cols, data, bond_idx, y_2, y_2, -tran_mod * (r_y ** 2 - 1))
    # Derivative df^2/(d_x1,d_y1) and df^2/(d_x2,d_y2)
    add_entry(rows, cols, data, bond_idx, x_1, y_1, -tran_mod * r_x * r_y)
    add_entry(rows, cols, data, bond_idx, x_2, y_2, -tran_mod * r_x * r_y)
    # Derivative df^2/(d_x1,d_x2) and df^2/(d_y1,d_y2)
    add_entry(rows, cols, data, bond_idx, x_1, x_2, tran_mod * (r_x ** 2 - 1))
    add_entry(rows, cols, data, bond_idx, y_1, y_2, tran_mod * (r_y ** 2 - 1))
    # Derivative df^2/(d_x1,d_y2) and df^2/(d_x2,d_y1)
    add_entry(rows, cols, data, bond_idx, x_1, y_2, tran_mod * r_x * r_y)
    add_entry(rows, cols, data, bond_idx, x_2, y_1, tran_mod * r_x * r_y)
    return csr_matrix((data, (rows, cols)), shape=(2 * n, 2 * n))


def get_transverse_jacobian(
        n: int,
        tran_mod: float,
        u_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray
) -> np.ndarray:
    """
    Returns the gradient [[fx_1, fy_1], [fx2, fy2]...] for transverse forces
    :param n: number of nodes times number of dimensions (2 * n)
    :type n: int
    :param tran_mod: mu - transverse modulus
    :type tran_mod: float
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
    gradient = np.zeros(n)

    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
    gradient_c = tran_mod * np.einsum("nk, nk -> n", u_matrix[i, :] - u_matrix[j, :], r_matrix[i, j])
    r_ij = r_matrix[i, j]
    grad_x = tran_mod * u_matrix[i, j, 0] - gradient_c * r_ij[:, 0]
    grad_y = tran_mod * u_matrix[i, j, 1] - gradient_c * r_ij[:, 1]
    np.add.at(gradient, 2 * i, grad_x)
    np.add.at(gradient, 2 * i + 1, grad_y)
    np.add.at(gradient, 2 * j, -grad_x)
    np.add.at(gradient, 2 * j + 1, -grad_y)

    return gradient


def get_transverse_energy(
        tran_mod: float,
        u_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray
) -> float:
    """
    Returns the energy transverse forces

    :param tran_mod: mu - transverse modulus
    :type tran_mod: float
    :param u_matrix: displacement field matrix
    :type u_matrix: Shape (n, 2) matrix
    :param r_matrix: matrix containing unit vectors between each node
    :type r_matrix: Shape (n, n, 2) matrix
    :return: energy associated with the pi bond and the gradient associated
        with each node x, y
    :param active_bond_indices: List containing indices of [i, j]
    :type active_bond_indices: Shape (# bonds, 2) matrix
    """
    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
    energy = 0.5 * tran_mod * (np.einsum("nk", "nk -> n", u_matrix[i, :] - u_matrix[j, :],
                                         u_matrix[i, :] - u_matrix[j, :]) - np.square(
        np.einsum("nk, nk -> n", u_matrix[i, j], r_matrix[i, j])))
    return energy
