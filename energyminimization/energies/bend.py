"""
bend.py is used to calculate the energy and the forces from bending (pi bonds)
"""

import numpy as np
from scipy.sparse import csr_matrix


def add_entry(rows, cols, data, i, r, c, v):
    rows[i], cols[i], data[i] = r, c, v


def get_bend_hessian(
        n: int,
        bend_mod: float,
        r_matrix: np.ndarray,
        active_pi_indices,
) -> csr_matrix:
    """
    Returns the Hessian matrix associated with bending (Currently uses a sparse
        matrix to save on memory usage and potentially increase speed).

    :param n: n is the number of nodes (size of hessian is d*n x d*n)
    :type n: int
    :param bend_mod: kappa - bending modulus
    :type bend_mod: float
    :param r_matrix: matrix containing unit vectors between each node.
        Example: r_matrix[i][j] returns the
    :param active_pi_indices: List containing indices of [vertex (j), i, k]
    :type active_pi_indices: Shape (# pi-bonds, 3) matrix
    unit vector [r_x, r_y] between nodes i and j
    :type r_matrix: Shape (n, n, 2) matrix
    :return: Hessian matrix size (n, n)
    """
    # The number of entries be equal to num_bonds * 36
    #   (since for each bond, there are 6 (3*2) node dimensions)
    # 6 (second order derivatives) and 30 (15*2 second partial derivatives)

    j, i, k = active_pi_indices[:, 0], active_pi_indices[:, 1], active_pi_indices[:, 2]
    idx, sign = active_pi_indices[:, 3], active_pi_indices[:, -1]
    num_pi_bonds = np.max(idx) + 1
    rows, cols = np.zeros(36 * num_pi_bonds, dtype=np.uint32), np.zeros(36 * num_pi_bonds, dtype=np.uint32)
    data = np.zeros(36 * num_pi_bonds, dtype=np.float64)

    pi_bond_idx = 36 * idx
    x_i, x_j, x_k = 2 * i, 2 * j, 2 * k
    y_i, y_j, y_k = 2 * i + 1, 2 * j + 1, 2 * k + 1
    if np.shape(sign) != np.shape(r_matrix[idx]):
        sign = sign[:, np.newaxis]
    r_ji = np.multiply(sign, r_matrix[idx])
    r_x, r_y = r_ji[:, 0], r_ji[:, 1]
    # Derivative df^2/d_x1^2 and df^2/d_y1^2
    dxx = bend_mod * np.square(r_y)
    dyy = bend_mod * np.square(r_x)
    dxy = bend_mod * r_x * r_y
    add_entry(rows, cols, data, pi_bond_idx, x_j, x_j, 4 * dxx)
    add_entry(rows, cols, data, pi_bond_idx + 1, y_j, y_j, 4 * dyy)
    add_entry(rows, cols, data, pi_bond_idx + 2, x_i, x_i, dxx)
    add_entry(rows, cols, data, pi_bond_idx + 3, y_i, y_i, dyy)
    add_entry(rows, cols, data, pi_bond_idx + 4, x_k, x_k, dxx)
    add_entry(rows, cols, data, pi_bond_idx + 5, y_k, y_k, dyy)
    # Derivative df^2/d_(x1,y1)
    add_entry(rows, cols, data, pi_bond_idx + 6, x_j, y_j, -4 * dxy)
    add_entry(rows, cols, data, pi_bond_idx + 7, y_j, x_j, -4 * dxy)
    add_entry(rows, cols, data, pi_bond_idx + 8, x_i, y_i, -dxy)
    add_entry(rows, cols, data, pi_bond_idx + 9, y_i, x_i, -dxy)
    add_entry(rows, cols, data, pi_bond_idx + 10, x_k, y_k, -dxy)
    add_entry(rows, cols, data, pi_bond_idx + 11, y_k, x_k, -dxy)
    # Derivative df^2/d_(x1,x2or3)
    add_entry(rows, cols, data, pi_bond_idx + 12, x_j, x_i, -2 * dxx)
    add_entry(rows, cols, data, pi_bond_idx + 13, x_i, x_j, -2 * dxx)
    add_entry(rows, cols, data, pi_bond_idx + 14, x_j, x_k, -2 * dxx)
    add_entry(rows, cols, data, pi_bond_idx + 15, x_k, x_j, -2 * dxx)
    add_entry(rows, cols, data, pi_bond_idx + 16, x_i, x_k, dxx)
    add_entry(rows, cols, data, pi_bond_idx + 17, x_k, x_i, dxx)
    # Derivative df^2/d_(y1,y2or3)
    add_entry(rows, cols, data, pi_bond_idx + 18, y_j, y_i, -2 * dyy)
    add_entry(rows, cols, data, pi_bond_idx + 19, y_i, y_j, -2 * dyy)
    add_entry(rows, cols, data, pi_bond_idx + 20, y_j, y_k, -2 * dyy)
    add_entry(rows, cols, data, pi_bond_idx + 21, y_k, y_j, -2 * dyy)
    add_entry(rows, cols, data, pi_bond_idx + 22, y_i, y_k, dyy)
    add_entry(rows, cols, data, pi_bond_idx + 23, y_k, y_i, dyy)
    # Derivative df^2/d_(x1,y2or3)
    add_entry(rows, cols, data, pi_bond_idx + 24, x_j, y_i, 2 * dxy)
    add_entry(rows, cols, data, pi_bond_idx + 25, y_i, x_j, 2 * dxy)
    add_entry(rows, cols, data, pi_bond_idx + 26, x_j, y_k, 2 * dxy)
    add_entry(rows, cols, data, pi_bond_idx + 27, y_k, x_j, 2 * dxy)
    add_entry(rows, cols, data, pi_bond_idx + 28, x_i, y_j, 2 * dxy)
    add_entry(rows, cols, data, pi_bond_idx + 29, y_j, x_i, 2 * dxy)
    add_entry(rows, cols, data, pi_bond_idx + 30, x_i, y_k, -dxy)
    add_entry(rows, cols, data, pi_bond_idx + 31, y_k, x_i, -dxy)
    add_entry(rows, cols, data, pi_bond_idx + 32, x_k, y_j, 2 * dxy)
    add_entry(rows, cols, data, pi_bond_idx + 33, y_j, x_k, 2 * dxy)
    add_entry(rows, cols, data, pi_bond_idx + 34, x_k, y_i, -dxy)
    add_entry(rows, cols, data, pi_bond_idx + 35, y_i, x_k, -dxy)

    return csr_matrix((data, (rows, cols)), shape=(2 * n, 2 * n))


def get_bend_jacobian(
        bend_mod: float,
        u_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_pi_indices: np.ndarray
) -> np.ndarray:
    """
    Returns the gradient [[fx_1, fy_1], [fx2, fy2]...] for bending forces

    :param bend_mod: kappa - bending modulus
    :type bend_mod: float
    :param r_matrix: matrix containing unit vectors between each node.
        Example: r_matrix[i][j] returns the
    unit vector [r_x, r_y] between nodes i and j
    :type r_matrix: Shape (n, n, 2) matrix
    :param u_matrix: matrix containing displacement vectors for each node
        compared to initial position
    Example: u_matrix[i] returns u_i
    :type u_matrix: Shape (n, 2) matrix
    :param active_pi_indices: List containing indices of [vertex (j), i, k]
    :type active_pi_indices: Shape (# pi-bonds, 3) matrix
    :return: gradient for bending forces for each node
    :rtype: shape (2n,) vector where the forces are [fx_1, fy_1, fx_2, fy_2]
    """
    n = u_matrix.size
    gradient = np.zeros(n)
    vertex, edge1, edge2 = active_pi_indices[:, 0], active_pi_indices[:, 1], active_pi_indices[:, 2]
    idx, sign = active_pi_indices[:, 3], active_pi_indices[:, 4]
    # Get the coefficient of the gradient
    r_ji = r_matrix[idx] * sign[:, np.newaxis]
    gradient_c = -bend_mod * np.cross((2 * u_matrix[vertex, :] - u_matrix[edge1, :] - u_matrix[edge2, :]), r_ji)
    # Force applied to the vertex node
    grad_x = np.multiply(gradient_c, r_ji[:, 1])
    grad_y = np.multiply(gradient_c, r_ji[:, 0])

    np.add.at(gradient, 2 * vertex, -2 * grad_x)
    np.add.at(gradient, 2 * vertex + 1, 2 * grad_y)
    # Force applied to the edge nodes (should be the same since they are 'rotating' around the center pivot)
    np.add.at(gradient, 2 * edge1, grad_x)
    np.add.at(gradient, 2 * edge1 + 1, -grad_y)
    np.add.at(gradient, 2 * edge2, grad_x)
    np.add.at(gradient, 2 * edge2 + 1, -grad_y)
    return gradient


def get_bend_energy(
        bend_mod: float,
        u_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_pi_indices: np.ndarray
) -> float:
    """
    get_bend_energy returns the energy and the gradient associated with the
        bending forces
    :param bend_mod: bending modulus (kappa)
    :type bend_mod: float
    :param u_matrix: displacement field matrix
    :type u_matrix: Shape (n, 2) matrix
    :param r_matrix: matrix containing unit vectors between each node
    :type r_matrix: Shape (n, n, 2) matrix
    :param active_pi_indices: List containing indices of [vertex (j), i, k]
    :type active_pi_indices: Shape (# pi-bonds, 3) matrix
    :return: energy associated with the pi bond and the gradient associated
        with each node x, y
    :rtype: float
    """
    j, i, k = active_pi_indices[:, 0], active_pi_indices[:, 1], active_pi_indices[:, 2]
    idx, sign = active_pi_indices[:, 3], active_pi_indices[:, 4]
    r_ji = r_matrix[idx] * sign[:, np.newaxis]
    energy = 0.5 * bend_mod * np.square(
        np.cross((2 * u_matrix[j, :] - u_matrix[i, :] - u_matrix[k, :]), r_ji))
    return np.sum(energy)
