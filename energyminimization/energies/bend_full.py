"""
Bending forces with non-linear effects (uses full form of 1/2 κ(theta-theta0)^2)
"""

import numpy as np
from scipy.sparse import csr_matrix

import energyminimization.energies.bend_helper as bh


def compute_angles(
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_pi_indices: np.ndarray,
        orig_pi_angles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the vector between each node (c_ij), the length of each bond, and the difference between the
        current length and the rest length
    :return: c_ji (contains vectors from node j to i for each bond), length_ji (length of each bond),
        d_ji (difference between current length and rest length)
    """
    pos_matrix = pos_matrix.reshape(-1, 2)
    j, i, k = active_pi_indices[:, 0], active_pi_indices[:, 1], active_pi_indices[:, 2]
    bond1_idx, bond2_idx, pi_idx = active_pi_indices[:, 3], active_pi_indices[:, 4], active_pi_indices[:, 5]
    hor_pbc1, top_pbc1 = active_pi_indices[:, 7], active_pi_indices[:, 8]
    hor_pbc2, top_pbc2 = active_pi_indices[:, 9], active_pi_indices[:, 10]
    # Vectors out of the vertex node
    c_ji = pos_matrix[j] - pos_matrix[i]
    c_jk = pos_matrix[j] - pos_matrix[k]
    # Handle periodic boundary conditions
    c_ji[np.where(hor_pbc1 == 1), 0] += corrections[0]
    c_ji[np.where(top_pbc1 == 1), 1] += corrections[1]
    c_jk[np.where(hor_pbc2 == 1), 0] += corrections[0]
    c_jk[np.where(top_pbc2 == 1), 1] += corrections[1]

    theta_jik = np.arctan2(np.cross(c_ji, c_jk), np.einsum('ij,ij->i', c_ji, c_jk))
    theta_jik[theta_jik < 0] += 2 * np.pi
    # Differences in theta
    d_jik = theta_jik - orig_pi_angles[pi_idx]
    return theta_jik, d_jik


def compute_angles(
        u_node_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_pi_indices: np.ndarray,
) -> np.ndarray:
    """
    Computes the angle for each pi bond
    """
    u_node_matrix = u_node_matrix.reshape(-1, 2)
    j, i, k = active_pi_indices[:, 0], active_pi_indices[:, 1], active_pi_indices[:, 2]
    bond1_idx, bond2_idx, pi_idx = active_pi_indices[:, 3], active_pi_indices[:, 4], active_pi_indices[:, 5]

    # c_ij points from j to i based on their current positions
    c_ij = np.subtract(r_matrix[bond1_idx], np.subtract(u_node_matrix[i, :], u_node_matrix[j, :]))
    c_kj = np.subtract(r_matrix[bond2_idx], np.subtract(u_node_matrix[j, :], u_node_matrix[k, :]))

    theta = np.arctan2(np.cross(c_ij, c_kj), np.dot(c_ij, c_kj))
    return theta


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
    bond1_idx, pi_idx, sign = active_pi_indices[:, 3], active_pi_indices[:, 4], active_pi_indices[:, 5]
    num_pi_bonds = np.max(pi_idx) + 1
    rows, cols = np.zeros(36 * num_pi_bonds, dtype=np.uint32), np.zeros(36 * num_pi_bonds, dtype=np.uint32)
    data = np.zeros(36 * num_pi_bonds, dtype=np.float64)

    entry_idx = 36 * pi_idx
    x_i, x_j, x_k = 2 * i, 2 * j, 2 * k
    y_i, y_j, y_k = 2 * i + 1, 2 * j + 1, 2 * k + 1

    # Get the index of bond 1
    if np.shape(sign) != np.shape(r_matrix[bond1_idx]):
        sign = sign[:, np.newaxis]
    r_ji = np.multiply(sign, r_matrix[bond1_idx])
    r_x, r_y = r_ji[:, 0], r_ji[:, 1]
    # Derivative df^2/d_x1^2 and df^2/d_y1^2
    dxx = bend_mod * np.square(r_y)
    dyy = bend_mod * np.square(r_x)
    dxy = bend_mod * r_x * r_y
    add_entry(rows, cols, data, entry_idx, x_j, x_j, 4 * dxx)
    add_entry(rows, cols, data, entry_idx + 1, y_j, y_j, 4 * dyy)
    add_entry(rows, cols, data, entry_idx + 2, x_i, x_i, dxx)
    add_entry(rows, cols, data, entry_idx + 3, y_i, y_i, dyy)
    add_entry(rows, cols, data, entry_idx + 4, x_k, x_k, dxx)
    add_entry(rows, cols, data, entry_idx + 5, y_k, y_k, dyy)
    # Derivative df^2/d_(x1,y1)
    add_entry(rows, cols, data, entry_idx + 6, x_j, y_j, -4 * dxy)
    add_entry(rows, cols, data, entry_idx + 7, y_j, x_j, -4 * dxy)
    add_entry(rows, cols, data, entry_idx + 8, x_i, y_i, -dxy)
    add_entry(rows, cols, data, entry_idx + 9, y_i, x_i, -dxy)
    add_entry(rows, cols, data, entry_idx + 10, x_k, y_k, -dxy)
    add_entry(rows, cols, data, entry_idx + 11, y_k, x_k, -dxy)
    # Derivative df^2/d_(x1,x2or3)
    add_entry(rows, cols, data, entry_idx + 12, x_j, x_i, -2 * dxx)
    add_entry(rows, cols, data, entry_idx + 13, x_i, x_j, -2 * dxx)
    add_entry(rows, cols, data, entry_idx + 14, x_j, x_k, -2 * dxx)
    add_entry(rows, cols, data, entry_idx + 15, x_k, x_j, -2 * dxx)
    add_entry(rows, cols, data, entry_idx + 16, x_i, x_k, dxx)
    add_entry(rows, cols, data, entry_idx + 17, x_k, x_i, dxx)
    # Derivative df^2/d_(y1,y2or3)
    add_entry(rows, cols, data, entry_idx + 18, y_j, y_i, -2 * dyy)
    add_entry(rows, cols, data, entry_idx + 19, y_i, y_j, -2 * dyy)
    add_entry(rows, cols, data, entry_idx + 20, y_j, y_k, -2 * dyy)
    add_entry(rows, cols, data, entry_idx + 21, y_k, y_j, -2 * dyy)
    add_entry(rows, cols, data, entry_idx + 22, y_i, y_k, dyy)
    add_entry(rows, cols, data, entry_idx + 23, y_k, y_i, dyy)
    # Derivative df^2/d_(x1,y2or3)
    add_entry(rows, cols, data, entry_idx + 24, x_j, y_i, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 25, y_i, x_j, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 26, x_j, y_k, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 27, y_k, x_j, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 28, x_i, y_j, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 29, y_j, x_i, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 30, x_i, y_k, -dxy)
    add_entry(rows, cols, data, entry_idx + 31, y_k, x_i, -dxy)
    add_entry(rows, cols, data, entry_idx + 32, x_k, y_j, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 33, y_j, x_k, 2 * dxy)
    add_entry(rows, cols, data, entry_idx + 34, x_k, y_i, -dxy)
    add_entry(rows, cols, data, entry_idx + 35, y_i, x_k, -dxy)

    return csr_matrix((data, (rows, cols)), shape=(2 * n, 2 * n))


def get_bend_jacobian(
        bend_mod: float,
        u_matrix: np.ndarray,
        r_matrix: np.ndarray,
        active_pi_indices: np.ndarray
) -> np.ndarray:
    theta, diff_theta = compute_angles(u_node_matrix=u_matrix, r_matrix=r_matrix, active_pi_indices=active_pi_indices)
    # Gradient is equal to -(kappa) * (theta - theta_0) *
    grad_x, grad_y = np.zeros(n_dof), np.zeros(n_dof)
    n_dof = u_matrix.size
    return bh.generate_jacobian(active_pi_indices=active_pi_indices, grad_x=grad_x, grad_y=grad_y, n=n_dof)


def get_bend_energy(
        bend_mod: float,
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_pi_indices: np.ndarray,
        orig_pi_angles: np.ndarray
) -> float:
    _, d_theta = compute_angles(pos_matrix=pos_matrix, corrections=corrections, active_pi_indices=active_pi_indices,
                                orig_pi_angles=orig_pi_angles)
    return 0.5 * bend_mod * np.sum(np.square(d_theta))
