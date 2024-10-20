"""
bend_linear.py is used to calculate the energy and the forces from bending (pi bonds)
"""

import numpy as np
from scipy.sparse import csr_matrix

import energyminimization.energies.bend_helper as bh


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
    bond1_idx, pi_idx, sign = active_pi_indices[:, 3], active_pi_indices[:, 5], active_pi_indices[:, 6]
    # Get the index of bond 1
    if np.shape(sign) != np.shape(r_matrix[bond1_idx]):
        sign = sign[:, np.newaxis]
    r_ji = np.multiply(sign, r_matrix[bond1_idx])
    r_x, r_y = r_ji[:, 0], r_ji[:, 1]
    # Derivative df^2/d_x1^2 and df^2/d_y1^2
    dxx = bend_mod * np.square(r_y)
    dyy = bend_mod * np.square(r_x)
    dxy = bend_mod * r_x * r_y

    n_dof = 2 * n
    return bh.generate_hessian(active_pi_indices=active_pi_indices, dxx=dxx, dyy=dyy, dxy=dxy, n=n_dof)


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
    n_dof = u_matrix.size
    vertex, edge1, edge2 = active_pi_indices[:, 0], active_pi_indices[:, 1], active_pi_indices[:, 2]
    idx, sign = active_pi_indices[:, 3], active_pi_indices[:, 4]
    # Get the coefficient of the gradient
    r_ji = r_matrix[idx] * sign[:, np.newaxis]
    gradient_c = -bend_mod * np.cross((2 * u_matrix[vertex, :] - u_matrix[edge1, :] - u_matrix[edge2, :]), r_ji)
    # Force applied to the vertex node
    grad_x = np.multiply(gradient_c, r_ji[:, 1])
    grad_y = np.multiply(gradient_c, r_ji[:, 0])
    return bh.generate_jacobian(active_pi_indices=active_pi_indices, grad_x=grad_x, grad_y=grad_y, n=n_dof)


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
