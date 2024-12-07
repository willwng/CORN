"""
Bending forces with non-linear effects (uses full form of 1/2 κ(theta-theta0)^2)
"""

import numpy as np
from scipy.sparse import csr_matrix


def compute_angles(
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_pi_indices: np.ndarray,
        orig_pi_angles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes:
    - The angle between the two bonds in a pi bond
    - The difference between the current angle and the original angle
    - The vector between the vertex and first edge
    - The vector between the vertex and second edge
    """
    pos_matrix = pos_matrix.reshape(-1, 2)
    j, i, k = active_pi_indices[:, 0], active_pi_indices[:, 1], active_pi_indices[:, 2]
    bond1_idx, bond2_idx, pi_idx = active_pi_indices[:, 3], active_pi_indices[:, 4], active_pi_indices[:, 5]
    hor_pbc1, top_pbc1 = active_pi_indices[:, 8], active_pi_indices[:, 9]
    hor_pbc2, top_pbc2 = active_pi_indices[:, 10], active_pi_indices[:, 11]
    sign_i, sign_j = active_pi_indices[:, 6], active_pi_indices[:, 7]

    # Vectors out of the vertex node
    c_ji = pos_matrix[i] - pos_matrix[j]
    c_jk = pos_matrix[k] - pos_matrix[j]
    # Handle periodic boundary conditions
    idx_hor_i = np.where(hor_pbc1 == 1)
    idx_top_i = np.where(top_pbc1 == 1)
    idx_hor_j = np.where(hor_pbc2 == 1)
    idx_top_j = np.where(top_pbc2 == 1)
    # The PBC correction depends on which node is the vertex (determined by sign)
    c_ji[idx_hor_i, 0] += sign_i[idx_hor_i] * corrections[0]
    c_ji[idx_top_i, 1] += sign_i[idx_top_i] * corrections[1]
    c_jk[idx_hor_j, 0] += sign_j[idx_hor_j] * corrections[0]
    c_jk[idx_top_j, 1] += sign_j[idx_top_j] * corrections[1]

    theta_jik = np.arctan2(np.cross(c_ji, c_jk), np.einsum('ij,ij->i', c_ji, c_jk))
    theta_jik[theta_jik < 0] += 2 * np.pi
    # Differences in theta
    d_jik = theta_jik - orig_pi_angles[pi_idx]
    return theta_jik, d_jik, c_ji, c_jk


def add_entry(rows, cols, data, i, r, c, v):
    rows[i], cols[i], data[i] = r, c, v


def get_bend_hessian(
        bend_mod: np.ndarray,
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_pi_indices: np.ndarray,
        orig_pi_angles: np.ndarray,
) -> csr_matrix:
    # Initialize the hessian matrix
    pi_idx = active_pi_indices[:, 5]
    num_pi_bonds = np.max(pi_idx) + 1
    rows, cols = np.zeros(36 * num_pi_bonds, dtype=np.uint32), np.zeros(36 * num_pi_bonds, dtype=np.uint32)
    data = np.zeros(36 * num_pi_bonds, dtype=np.float64)
    theta, diff_theta, c_ji, c_jk = compute_angles(pos_matrix=pos_matrix, corrections=corrections,
                                                   active_pi_indices=active_pi_indices, orig_pi_angles=orig_pi_angles)

    # The second partials are: kappa * [(theta - theta_0) * d^2_theta/d_dof^2 + d_theta/d_dof * d_theta/d_dof]
    # We first compute the helper values needed for d_theta/d_dof
    length_ji = np.linalg.norm(c_ji, axis=1)
    length_jk = np.linalg.norm(c_jk, axis=1)
    t_ji = np.divide(c_ji, np.square(length_ji)[: , np.newaxis])
    t_jk = np.divide(c_jk, np.square(length_jk)[: , np.newaxis])
    # Compute d_theta/d_dof
    d_theta_djx = t_ji[:, 1] - t_jk[:, 1]
    d_theta_dix = -t_ji[:, 1]
    d_theta_dkx = t_jk[:, 1]
    # Partials of theta with respect to y
    d_theta_djy = -t_ji[:, 0] + t_jk[:, 0]
    d_theta_diy = t_ji[:, 0]
    d_theta_dky = -t_jk[:, 0]
    # Squared values of d_theta/d_dof
    d_theta_djx_sq = np.square(d_theta_djx)
    d_theta_dix_sq = np.square(d_theta_dix)
    d_theta_dkx_sq = np.square(d_theta_dkx)
    d_theta_djy_sq = np.square(d_theta_djy)
    d_theta_diy_sq = np.square(d_theta_diy)
    d_theta_dky_sq = np.square(d_theta_dky)

    # Now compute the second partials of d^2_theta/d_dof^2
    # First compute some helper values
    l_ji_sq = np.square(length_ji)
    l_jk_sq = np.square(length_jk)
    l_ji_fourth = np.power(length_ji, 4)
    l_jk_fourth = np.power(length_jk, 4)
    cji_prod = np.prod(c_jk, axis=1)
    cjk_prod = np.prod(c_ji, axis=1)
    cji_sq = np.square(c_ji)
    cjk_sq = np.square(c_jk)
    cji_x_plus_y = c_ji[:, 0] + c_ji[:, 1]
    cjk_x_plus_y = c_jk[:, 0] + c_jk[:, 1]
    cji_x_minus_y = c_ji[:, 0] - c_ji[:, 1]
    cjk_x_minus_y = c_jk[:, 0] - c_jk[:, 1]
    # We're ready!
    d2_theta_djx_jx = 2 * (cjk_prod / l_jk_fourth - cji_prod / l_ji_fourth)
    d2_theta_djx_jy = (1 / l_ji_sq) - (1 / l_jk_sq) + (2 * cjk_sq[:, 1] / l_jk_fourth) - (2 * cji_sq[:, 1] / l_ji_fourth)
    d2_theta_djx_ix = 2 * (cji_prod / l_ji_fourth)
    d2_theta_djx_iy = -(cji_x_plus_y * cji_x_minus_y) / l_ji_fourth
    d2_theta_djx_kx = -2 * (cjk_prod / l_jk_fourth)
    d2_theta_djx_ky = (cjk_x_plus_y * cjk_x_minus_y) / l_jk_fourth

    d2_theta_djy_jy = 2 * (cji_prod / l_ji_fourth - cjk_prod / l_jk_fourth)
    d2_theta_djy_ix = -(cji_sq[:, 0] - cji_sq[:, 1]) / l_ji_fourth
    d2_theta_djy_iy = -2 * cji_prod / l_ji_fourth
    d2_theta_djy_kx = (cjk_sq[:, 0] - cjk_sq[:, 1]) / l_jk_fourth
    d2_theta_djy_ky = 2 * cjk_prod / l_jk_fourth

    d2_theta_dix_ix = -2 * cji_prod / l_ji_fourth
    d2_theta_dix_iy = (cji_sq[:, 0] - cji_sq[:, 1]) / l_ji_fourth
    d2_theta_dix_kx = np.zeros_like(d2_theta_dix_ix)
    d2_theta_dix_ky = np.zeros_like(d2_theta_dix_ix)

    d2_theta_diy_iy = 2 * cji_prod / l_ji_fourth
    d2_theta_diy_kx = np.zeros_like(d2_theta_dix_ix)
    d2_theta_diy_ky = np.zeros_like(d2_theta_dix_ix)

    d2_theta_dkx_kx = 2 * cjk_prod / l_jk_fourth
    d2_theta_dkx_ky = -(cjk_sq[:, 0] - cjk_sq[:, 1]) / l_jk_fourth

    d2_theta_dky_ky = -2 * cjk_prod / l_jk_fourth

    # Finally, compute the hessian terms
    d2e_djx_jx = bend_mod * (diff_theta * d2_theta_djx_jx + d_theta_djx_sq)
    d2e_djx_jy = bend_mod * (diff_theta * d2_theta_djx_jy + d_theta_djx * d_theta_djy)
    d2e_djx_ix = bend_mod * (diff_theta * d2_theta_djx_ix + d_theta_djx * d_theta_dix)
    d2e_djx_iy = bend_mod * (diff_theta * d2_theta_djx_iy + d_theta_djx * d_theta_diy)
    d2e_djx_kx = bend_mod * (diff_theta * d2_theta_djx_kx + d_theta_djx * d_theta_dkx)
    d2e_djx_ky = bend_mod * (diff_theta * d2_theta_djx_ky + d_theta_djx * d_theta_dky)

    d2e_djy_jx = d2e_djx_jy
    d2e_djy_jy = bend_mod * (diff_theta * d2_theta_djy_jy + d_theta_djy_sq)
    d2e_djy_ix = bend_mod * (diff_theta * d2_theta_djy_ix + d_theta_djy * d_theta_dix)
    d2e_djy_iy = bend_mod * (diff_theta * d2_theta_djy_iy + d_theta_djy * d_theta_diy)
    d2e_djy_kx = bend_mod * (diff_theta * d2_theta_djy_kx + d_theta_djy * d_theta_dkx)
    d2e_djy_ky = bend_mod * (diff_theta * d2_theta_djy_ky + d_theta_djy * d_theta_dky)

    d2e_dix_jx = d2e_djx_ix
    d2e_dix_jy = d2e_djy_ix
    d2e_dix_ix = bend_mod * (diff_theta * d2_theta_dix_ix + d_theta_dix_sq)
    d2e_dix_iy = bend_mod * (diff_theta * d2_theta_dix_iy + d_theta_dix * d_theta_diy)
    d2e_dix_kx = bend_mod * (diff_theta * d2_theta_dix_kx + d_theta_dix * d_theta_dkx)
    d2e_dix_ky = bend_mod * (diff_theta * d2_theta_dix_ky + d_theta_dix * d_theta_dky)

    d2e_diy_jx = d2e_djx_iy
    d2e_diy_jy = d2e_djy_iy
    d2e_diy_ix = d2e_dix_iy
    d2e_diy_iy = bend_mod * (diff_theta * d2_theta_diy_iy + d_theta_diy_sq)
    d2e_diy_kx = bend_mod * (diff_theta * d2_theta_diy_kx + d_theta_diy * d_theta_dkx)
    d2e_diy_ky = bend_mod * (diff_theta * d2_theta_diy_ky + d_theta_diy * d_theta_dky)

    d2e_dkx_jx = d2e_djx_kx
    d2e_dkx_jy = d2e_djy_kx
    d2e_dkx_ix = d2e_dix_kx
    d2e_dkx_iy = d2e_diy_kx
    d2e_dkx_kx = bend_mod * (diff_theta * d2_theta_dkx_kx + d_theta_dkx_sq)
    d2e_dkx_ky = bend_mod * (diff_theta * d2_theta_dkx_ky + d_theta_dkx * d_theta_dky)

    d2e_dky_jx = d2e_djx_ky
    d2e_dky_jy = d2e_djy_ky
    d2e_dky_ix = d2e_dix_ky
    d2e_dky_iy = d2e_diy_ky
    d2e_dky_kx = d2e_dkx_ky
    d2e_dky_ky = bend_mod * (diff_theta * d2_theta_dky_ky + d_theta_dky_sq)

    j, i, k = active_pi_indices[:, 0], active_pi_indices[:, 1], active_pi_indices[:, 2]
    j_x, j_y, i_x, i_y, k_x, k_y = 2 * j, 2 * j + 1, 2 * i, 2 * i + 1, 2 * k, 2 * k + 1
    entry_idx = 36 * pi_idx # 36 entries per pi bond (such that no entries overlap)

    add_entry(rows, cols, data, entry_idx, j_x, j_x, d2e_djx_jx)
    add_entry(rows, cols, data, entry_idx + 1, j_x, j_y, d2e_djx_jy)
    add_entry(rows, cols, data, entry_idx + 2, j_x, i_x, d2e_djx_ix)
    add_entry(rows, cols, data, entry_idx + 3, j_x, i_y, d2e_djx_iy)
    add_entry(rows, cols, data, entry_idx + 4, j_x, k_x, d2e_djx_kx)
    add_entry(rows, cols, data, entry_idx + 5, j_x, k_y, d2e_djx_ky)

    add_entry(rows, cols, data, entry_idx + 6, j_y, j_x, d2e_djy_jx)
    add_entry(rows, cols, data, entry_idx + 7, j_y, j_y, d2e_djy_jy)
    add_entry(rows, cols, data, entry_idx + 8, j_y, i_x, d2e_djy_ix)
    add_entry(rows, cols, data, entry_idx + 9, j_y, i_y, d2e_djy_iy)
    add_entry(rows, cols, data, entry_idx + 10, j_y, k_x, d2e_djy_kx)
    add_entry(rows, cols, data, entry_idx + 11, j_y, k_y, d2e_djy_ky)

    add_entry(rows, cols, data, entry_idx + 12, i_x, j_x, d2e_dix_jx)
    add_entry(rows, cols, data, entry_idx + 13, i_x, j_y, d2e_dix_jy)
    add_entry(rows, cols, data, entry_idx + 14, i_x, i_x, d2e_dix_ix)
    add_entry(rows, cols, data, entry_idx + 15, i_x, i_y, d2e_dix_iy)
    add_entry(rows, cols, data, entry_idx + 16, i_x, k_x, d2e_dix_kx)
    add_entry(rows, cols, data, entry_idx + 17, i_x, k_y, d2e_dix_ky)

    add_entry(rows, cols, data, entry_idx + 18, i_y, j_x, d2e_diy_jx)
    add_entry(rows, cols, data, entry_idx + 19, i_y, j_y, d2e_diy_jy)
    add_entry(rows, cols, data, entry_idx + 20, i_y, i_x, d2e_diy_ix)
    add_entry(rows, cols, data, entry_idx + 21, i_y, i_y, d2e_diy_iy)
    add_entry(rows, cols, data, entry_idx + 22, i_y, k_x, d2e_diy_kx)
    add_entry(rows, cols, data, entry_idx + 23, i_y, k_y, d2e_diy_ky)

    add_entry(rows, cols, data, entry_idx + 24, k_x, j_x, d2e_dkx_jx)
    add_entry(rows, cols, data, entry_idx + 25, k_x, j_y, d2e_dkx_jy)
    add_entry(rows, cols, data, entry_idx + 26, k_x, i_x, d2e_dkx_ix)
    add_entry(rows, cols, data, entry_idx + 27, k_x, i_y, d2e_dkx_iy)
    add_entry(rows, cols, data, entry_idx + 28, k_x, k_x, d2e_dkx_kx)
    add_entry(rows, cols, data, entry_idx + 29, k_x, k_y, d2e_dkx_ky)

    add_entry(rows, cols, data, entry_idx + 30, k_y, j_x, d2e_dky_jx)
    add_entry(rows, cols, data, entry_idx + 31, k_y, j_y, d2e_dky_jy)
    add_entry(rows, cols, data, entry_idx + 32, k_y, i_x, d2e_dky_ix)
    add_entry(rows, cols, data, entry_idx + 33, k_y, i_y, d2e_dky_iy)
    add_entry(rows, cols, data, entry_idx + 34, k_y, k_x, d2e_dky_kx)
    add_entry(rows, cols, data, entry_idx + 35, k_y, k_y, d2e_dky_ky)

    n_dof = pos_matrix.size
    return csr_matrix((data, (rows, cols)), shape=(n_dof, n_dof))


def get_bend_jacobian(
        bend_mod: np.ndarray,
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_pi_indices: np.ndarray,
        orig_pi_angles: np.ndarray,
) -> np.ndarray:
    gradient = np.zeros(pos_matrix.size)
    theta, diff_theta, c_ji, c_jk = compute_angles(pos_matrix=pos_matrix, corrections=corrections,
                                                   active_pi_indices=active_pi_indices, orig_pi_angles=orig_pi_angles)
    # Gradient is equal to kappa * (theta - theta_0) * d_theta/d_dof
    grad_factor = np.multiply(bend_mod, diff_theta)[: , np.newaxis]
    # Temporary variables useful for the gradient
    length_ji = np.linalg.norm(c_ji, axis=1)
    length_jk = np.linalg.norm(c_jk, axis=1)
    t_ji = np.divide(c_ji, np.square(length_ji)[:, np.newaxis])
    t_jk = np.divide(c_jk, np.square(length_jk)[: , np.newaxis])
    # Multiply be the gradient factor so we don't have to later (save some redundant computation)
    g_ji = np.multiply(grad_factor, t_ji)
    g_jk = np.multiply(grad_factor, t_jk)
    # Partials of theta with respect to x (times the gradient factor gives the correct partials of energy)
    de_djx = -g_ji[:, 1] + g_jk[:, 1]
    de_dix = g_ji[:, 1]
    de_dkx = -g_jk[:, 1]
    # Partials of theta with respect to y
    de_djy = g_ji[:, 0] - g_jk[:, 0]
    de_diy = -g_ji[:, 0]
    de_dky = g_jk[:, 0]

    # Update the gradient
    j, i, k = active_pi_indices[:, 0], active_pi_indices[:, 1], active_pi_indices[:, 2]
    jx, jy, ix, iy, kx, ky = 2 * j, 2 * j + 1, 2 * i, 2 * i + 1, 2 * k, 2 * k + 1
    np.add.at(gradient, jx, de_djx)
    np.add.at(gradient, jy, de_djy)
    np.add.at(gradient, ix, de_dix)
    np.add.at(gradient, iy, de_diy)
    np.add.at(gradient, kx, de_dkx)
    np.add.at(gradient, ky, de_dky)
    return gradient


def get_bend_energy(
        bend_mod: np.ndarray,
        pos_matrix: np.ndarray,
        corrections: np.ndarray,
        active_pi_indices: np.ndarray,
        orig_pi_angles: np.ndarray
) -> float:
    # Compute (theta - theta_0) for each bond
    _, d_theta, _, _ = compute_angles(pos_matrix=pos_matrix, corrections=corrections,
                                      active_pi_indices=active_pi_indices,
                                      orig_pi_angles=orig_pi_angles)
    # Energy is equal to 1/2 * kappa * (theta - theta_0)^2 for each bond
    energy = 0.5 * np.multiply(bend_mod, np.square(d_theta))
    return np.sum(energy)
