"""
matrix_helper.py is used to create necessary matrices for the minimization algorithms
"""
from typing import List, Tuple, Dict

import numpy as np
from scipy.sparse import csr_matrix

from energyminimization.energies.bend import get_bend_hessian
from energyminimization.energies.stretch import get_stretch_hessian
from energyminimization.energies.transverse import get_transverse_hessian
from energyminimization.transformations import transform_pos_matrix
from lattice.abstract_lattice import AbstractLattice

# Caching results when fetching the position matrix for a lattice
lattice_to_pos: Dict[AbstractLattice, np.ndarray] = {}


def create_pos_matrix(lattice: AbstractLattice) -> np.ndarray:
    """
    Used to create a shape (n, 2) matrix (n is number of nodes) used to denote
        (x,y) position of the *rest position* each node
    Example:
    [[0, 5],
     [2,5]]
    is used to show that node 1 is at (0,5) and node 2 is at (2,5)

    :param lattice: lattice object to create matrix
    :type lattice: Object that inherits AbstractLattice
    :return: Shape (n, 2) matrix denoting positions of initial nodes
    """
    # Use pre-computed result
    if lattice in lattice_to_pos:
        return lattice_to_pos[lattice]
    # Create new matrix
    nodes = lattice.get_nodes()
    num_nodes = len(nodes)
    pos_matrix = np.zeros((num_nodes, 2))
    for node in nodes:
        n_pos = node.get_xy()
        n_id = node.get_id()
        pos_matrix[n_id][0] = n_pos[0]
        pos_matrix[n_id][1] = n_pos[1]
    # Store matrix
    lattice_to_pos[lattice] = pos_matrix
    return pos_matrix


def create_r_matrix(
        pos_vector: np.ndarray,
        active_bond_indices: np.ndarray,
        lattice: AbstractLattice,
        normalize: bool
) -> np.ndarray:
    """
    Returns the matrix containing vectors between each node.
        Example: r_matrix[i][j] returns the
    unit vector [r_x, r_y] between nodes i and j

    :param pos_vector: Position of each node (will be reshaped) in order of
        [x_1, y_1, x_2, ...]
    :type pos_vector: Must contain 2n elements
    :param active_bond_indices: List containing indices of [i, j]
    :type active_bond_indices: Shape (# bonds, 2) matrix
    :param lattice: Lattice object
    :param normalize: Whether to normalize the vectors to unit vectors
    :type normalize: bool
    :return: Shape (num_bonds, 2) matrix
    """
    pos_matrix = pos_vector.reshape((-1, 2))
    # Change in x from node to node
    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]

    # Periodic boundary conditions require correction
    hor_pbc, top_pbc, idx = active_bond_indices[:, 2], active_bond_indices[:, 3], active_bond_indices[:, 4]

    idx_hor = idx[np.where(hor_pbc == 1)]
    idx_top = idx[np.where(top_pbc == 1)]
    correction = np.zeros((len(active_bond_indices), 2))
    correction[idx_hor, 0] = lattice.get_length()
    correction[idx_top, 1] = lattice.get_height() + lattice.height_increment
    r_matrix = pos_matrix[j, :] - pos_matrix[i, :] + correction

    # Debugging: verify that r_matrix is correct
    # verify_r_matrix(pos_vector, r_matrix, lattice, active_bond_indices)

    if normalize:
        norm_factor = np.linalg.norm(r_matrix, axis=1, keepdims=True)
        r_matrix = r_matrix / norm_factor

    return r_matrix


def verify_r_matrix(
        pos_vector: np.ndarray,
        r_matrix: np.ndarray,
        lattice: AbstractLattice,
        active_bond_indices: np.ndarray,
        is_generic: bool = False
) -> None:
    """
    Verifies that the r_matrix is correct
    """
    pos_matrix = pos_vector.reshape((-1, 2))
    for i, j, hor, top, idx in active_bond_indices:
        i_pos = pos_matrix[i]
        j_pos = pos_matrix[j]
        r_ij = j_pos - i_pos
        if hor:
            assert i_pos[0] > j_pos[0]
            i_pos = i_pos - np.array([lattice.get_length(), 0])
            r_ij = j_pos - i_pos
        if top:
            assert i_pos[1] > j_pos[1]
            assert i_pos[1] == lattice.get_height()
            i_pos = i_pos - np.array([0, lattice.get_height() + lattice.height_increment])
            r_ij = j_pos - i_pos

        assert np.allclose(r_ij, r_matrix[idx])
        if not is_generic:
            assert np.allclose(np.linalg.norm(r_ij), 1)

    return


def create_correction_matrix(lattice: AbstractLattice, init_pos: np.ndarray,
                             trans_matrix: np.ndarray,
                             all_bond_indices: np.ndarray) -> np.ndarray:
    """
    create_correction_matrix returns a matrix that ensures the displacements along PBCs are accounted for
    """
    init_pos = init_pos.reshape((-1, 2))
    correction_matrix = np.zeros_like(init_pos)

    i, j = all_bond_indices[:, 0], all_bond_indices[:, 1]
    hor_pbc, top_pbc = all_bond_indices[:, 2], all_bond_indices[:, 3]

    # Apply transformation matrix to every vector in position matrix
    trans_positions = transform_pos_matrix(pos_matrix=init_pos, transformation_matrix=trans_matrix)
    u_matrix = trans_positions - init_pos

    # Find the "imaginary" position of the i nodes (greater x and/or y positions)
    imag_pos = init_pos.copy()
    i_hor = i[np.where(hor_pbc == 1)]
    i_vert = i[np.where(top_pbc == 1)]
    i_correction = i[np.where((hor_pbc == 1) | (top_pbc == 1))]
    imag_pos[i_hor, 0] -= lattice.get_length()
    imag_pos[i_vert, 1] -= (lattice.get_height() + lattice.height_increment)

    # Compute where the imaginary positions are after applying the transformation
    trans_imag_pos = (trans_matrix @ imag_pos.T).T
    u_imag_matrix = trans_imag_pos - imag_pos

    # Correction is just the difference between the imaginary displacement and the real displacement
    correction_matrix[i_correction, :] = u_imag_matrix[i_correction, :] - u_matrix[i_correction, :]

    return correction_matrix.flatten()


def create_u_matrix(pos_matrix: np.ndarray, init_pos: np.ndarray) -> np.ndarray:
    """
    Returns the displacement of each node compared to the init_pos
    Example: u_matrix[i] returns u_i which contains [u_xi, u_yi]

    :param pos_matrix: position of the current nodes
    :type: Shape (n, 2)
    :param init_pos: initial position matrix of the nodes
    :type: Shape (n, 2)
    :return: Shape (n, 2) matrix containing displacement field
    """
    return (pos_matrix - init_pos).reshape((-1, 2))


class KMatrixResult:
    """
    Wrapper class for obtaining the K matrices and computing values (e.g., quadratic forms)
    """
    k_stretch: csr_matrix
    k_bend: csr_matrix
    k_transverse: csr_matrix
    k_total: csr_matrix

    def __init__(self, k_stretch: csr_matrix, k_bend: csr_matrix, k_transverse: csr_matrix):
        self.k_stretch = k_stretch
        self.k_bend = k_bend
        self.k_transverse = k_transverse
        valid_matrices = filter(lambda x: x is not None, [k_stretch, k_bend, k_transverse])
        self.k_total = sum(valid_matrices)

    def compute_quad_forms(self, x: np.ndarray) -> Tuple[List[float], float]:
        """
        Computes the quadratic form 1/2 x^T K x
        """
        result = [0.5 * x.T @ mat @ x if mat is not None else 0.0 for mat in
                  [self.k_stretch, self.k_bend, self.k_transverse]]
        return result, sum(result)

    def apply_projection(self, projection: csr_matrix):
        def project(mat: csr_matrix):
            return projection @ mat @ projection.T if mat is not None else None

        self.k_stretch = project(self.k_stretch)
        self.k_bend = project(self.k_bend)
        self.k_transverse = project(self.k_transverse)
        valid_matrices = filter(lambda x: x is not None, [self.k_stretch, self.k_bend, self.k_transverse])
        self.k_total = sum(valid_matrices)


def get_k_matrices(
        n: int,
        r_matrix: np.ndarray,
        stretch_mod: float,
        bend_mod: float,
        tran_mod: float,
        active_bond_indices: np.ndarray,
        active_pi_indices: np.ndarray
) -> KMatrixResult:
    """
    Returns the K matrices (hessians) for the energy minimization
    """
    k_stretch, k_bend, k_transverse = None, None, None
    if stretch_mod != 0:
        k_stretch = get_stretch_hessian(n=n, r_matrix=r_matrix, stretch_mod=stretch_mod,
                                        active_bond_indices=active_bond_indices)
    if bend_mod != 0:
        k_bend = get_bend_hessian(n=n, r_matrix=r_matrix, bend_mod=bend_mod, active_pi_indices=active_pi_indices)

    if tran_mod != 0:
        k_transverse = get_transverse_hessian(n=n, r_matrix=r_matrix, tran_mod=bend_mod,
                                              active_bond_indices=active_bond_indices)
    return KMatrixResult(k_stretch=k_stretch, k_bend=k_bend, k_transverse=k_transverse)
