"""
matrix_helper.py is used to create necessary matrices for the minimization algorithms
"""
from typing import List, Tuple, Dict

import numpy as np
from scipy.sparse import csr_matrix, diags

from energyminimization.energies.bend import get_bend_hessian
from energyminimization.energies.stretch import get_stretch_hessian
from energyminimization.energies.transverse import get_transverse_hessian
from lattice.abstract_lattice import AbstractLattice

lattice_to_pos: Dict[AbstractLattice, np.ndarray] = {}


def create_pos_matrix(lattice: AbstractLattice) -> np.ndarray:
    """
    Used to create a shape (n, 2) matrix (n is number of nodes) used to denote
        (x,y) position of each node
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
    nodes = lattice.get_nodes()
    num_nodes = len(nodes)
    pos_matrix = np.zeros((num_nodes, 2))
    for node in nodes:
        n_pos = node.get_xy()
        n_id = node.get_id()
        pos_matrix[n_id][0] = n_pos[0]
        pos_matrix[n_id][1] = n_pos[1]
    lattice_to_pos[lattice] = pos_matrix
    return pos_matrix


def get_transformation_matrices(gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the stretch x, stretch y, shear, dilate transformation matrices
    """
    stretch_x = np.array([[1 + gamma, 0], [0, 1]])
    stretch_y = np.array([[1, 0], [0, 1 + gamma]])
    shear = np.array([[1, gamma], [gamma, 1]])
    dilate = np.array([[1 + gamma, 0], [0, 1 + gamma]])
    return stretch_x, stretch_y, shear, dilate


def transform_pos_matrix(pos_matrix: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Returns the transformed position matrix
    """
    pos_matrix = pos_matrix.reshape((-1, 2))
    return (transformation_matrix @ pos_matrix.T).T


def shear_pos_matrix(
        lattice: AbstractLattice,
        pos_matrix: np.ndarray,
        hor: float,
        comp: float,
        shear: bool,
) -> np.ndarray:
    """
    shear_pos_matrix returns the position matrix after lattice has been
        compressed and sheared (based on parallelogram)
    Note: does not mutate the lattice object

    :param lattice: lattice object
    :type lattice: Object that inherits AbstractLattice
    :param pos_matrix: Matrix denoting initial positions of each node
    :type pos_matrix: Shape (n, 2)
    :param hor: Percent shear strain
    :type hor: float
    :param comp: Percent of maximum height that the lattice should be
        compressed to
    :type comp: float
    :param shear: Whether to shear or stretch the lattice
    :return: A new (n, 2) matrix denoting positions of nodes after shearing
    """
    nodes = lattice.get_nodes()
    sheared_pos_matrix = pos_matrix.copy()
    half_height = max(node.y for node in nodes) / 2
    min_x = min(node.get_xy()[0] for node in nodes)
    for node in nodes:
        n_pos = node.get_xy()
        n_id = node.get_id()
        if shear:
            sheared_pos_matrix[n_id][0] += hor * (n_pos[1] - half_height)
        else:
            sheared_pos_matrix[n_id][0] += hor * (sheared_pos_matrix[n_id][0] - min_x)
        sheared_pos_matrix[n_id][1] += comp * (sheared_pos_matrix[n_id][1])
    return sheared_pos_matrix


def create_boundary_matrix(
        lattice: AbstractLattice, pos_matrix: np.ndarray
) -> List[List[None]]:
    """
    Creates the bounds used in minimization. Each node is given an upper and
    lower bound in the form [low, up]
    for each dimension (x, y). Top and bottom nodes should be stationary

    :param lattice: lattice object
    :type lattice: Object that inherits AbstractLattice
    :param pos_matrix: position matrix of the current lattice
        (does not have to be initial)
    :type pos_matrix: shape (n, 2)
    :return: A 2-element [low, high] array for each node and dimension.
    [[low_x1, high_x1], [low_y1, high_y1], [low_x2, high_x2], ...]
    """
    nodes = lattice.get_nodes()
    n = len(nodes)
    bounds = [[None, None]] * (2 * n)
    for node in nodes:
        n_id = node.get_id()
        if node.is_boundary():
            bounds[2 * n_id] = [pos_matrix[n_id][0], pos_matrix[n_id][0]]
            bounds[2 * n_id + 1] = [pos_matrix[n_id][1], pos_matrix[n_id][1]]
    return bounds


def create_r_matrix(
        pos_vector: np.ndarray, edge_matrix: np.ndarray, active_bond_indices: np.ndarray, normalize: bool
) -> np.ndarray:
    """
    Returns the matrix containing vectors between each node.
        Example: r_matrix[i][j] returns the
    unit vector [r_x, r_y] between nodes i and j

    :param pos_vector: Position of each node (will be reshaped) in order of
        [x_1, y_1, x_2, ...]
    :type pos_vector: Must contain 2n elements
    :param edge_matrix: matrix used to update positions of the edge nodes
        (so edge bonds are not overly long)
    :type edge_matrix: Shape (n_bonds, 2)
    :param active_bond_indices: List containing indices of [i, j]
    :type active_bond_indices: Shape (# bonds, 2) matrix
    :param normalize: Whether to normalize the vectors to unit vectors
    :type normalize: bool
    :return: Shape (n, n, 2) matrix
    """
    pos_matrix = pos_vector.reshape((-1, 2))
    # Change in x from node to node
    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
    r_matrix = pos_matrix[j, :] - pos_matrix[i, :]
    r_matrix -= edge_matrix  # Subtract the x values for edge bonds
    if normalize:
        norm_factor = np.linalg.norm(r_matrix, axis=1, keepdims=True)
        r_matrix = r_matrix / norm_factor
    return r_matrix


def create_length_matrix(
        pos_vector: np.ndarray, edge_matrix: np.ndarray, active_bond_indices: np.ndarray
) -> np.ndarray:
    """
    Matrix representing the rest lengths of each bond (similar to r_matrix)
    """
    pos_matrix = pos_vector.reshape((-1, 2))
    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
    r_matrix = pos_matrix[j, :] - pos_matrix[i, :]
    r_matrix -= edge_matrix
    length_matrix = np.linalg.norm(r_matrix, axis=1, keepdims=True)
    return length_matrix.flatten()


def create_edge_matrix(lattice: AbstractLattice, init_pos: np.ndarray, active_bond_indices: np.ndarray) -> np.ndarray:
    """
    create_edge_matrix is used to create a matrix that ensures the edge bonds are not overly long. Used
    by subtracting this matrix from r_matrix

    :param lattice: lattice object
    :param init_pos: initial position matrix of the nodes
    :type: Shape (# active_bonds, 2)
    :param active_bond_indices: List containing indices of [i, j]
    :type active_bond_indices: Shape (# bonds, 2) matrix
    """
    n = len(active_bond_indices)
    edge_matrix = np.zeros((n, 2))

    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
    hor_pbc, top_pbc = active_bond_indices[:, 2], active_bond_indices[:, 3]

    # # Horizontal periodic boundary conditions
    # x_i, x_j = init_pos[i, 0], init_pos[j, 0]
    # e1 = np.where((hor_pbc == 1) & (x_i > x_j))
    # e2 = np.where((hor_pbc == 1) & (x_i <= x_j))
    # assert e2[0].size == 0  # x position of node_i should always be greater than node_j
    # edge_matrix[e1, 0] = -lattice.get_length()
    #
    # # Vertical periodic boundary conditions
    # y_i, y_j = init_pos[i, 1], init_pos[j, 1]
    # e1 = np.where((top_pbc == 1) & (y_i > y_j))
    # e2 = np.where((top_pbc == 1) & (y_i <= y_j))
    # assert e2[0].size == 0  # y position of node_i should always be greater than node_j
    # edge_matrix[e1, 1] = -lattice.get_height() - lattice.height_increment
    return edge_matrix


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


def get_projection_matrices(lattice: AbstractLattice) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """
    Computes the projection matrices as described in the correlated networks paper
    p_r_to_n is the projection matrix from the non-boundary nodes to full n dimensions
    p_n_to_r is the opposite of p_r_to_n
    i_boundary is the selection operator for boundary nodes
    """
    r_nodes = [node for node in lattice.get_nodes() if not node.is_boundary()]
    n, r = len(lattice.get_nodes()), len(r_nodes)
    # Arrays to create the sparse projection matrix
    rows, cols = np.zeros(2 * r, dtype=np.uint32), np.zeros(2 * r, dtype=np.uint32)
    data = np.ones(2 * r, dtype=np.uint8)
    # Selection operator for boundary nodes (1 if corresponding to boundary node, 0 otherwise)
    i_boundary_lookup = np.zeros(2 * n, dtype=np.uint8)
    i = 0  # Iteration number for nodes in R
    for node in lattice.get_nodes():
        if not node.is_boundary():
            rows[i], cols[i] = 2 * node.get_id(), 2 * i
            rows[r + i], cols[r + i] = 2 * node.get_id() + 1, 2 * i + 1
            i += 1
        else:
            i_boundary_lookup[2 * node.get_id()] = 1
            i_boundary_lookup[2 * node.get_id() + 1] = 1
    i_boundary = diags(i_boundary_lookup, dtype=np.uint8, shape=(2 * n, 2 * n))
    p_r_to_n = csr_matrix((data, (rows, cols)), dtype=np.uint8, shape=(2 * n, 2 * r))
    return p_r_to_n, p_r_to_n.T.tocsr(), i_boundary


def get_box_projection(lattice: AbstractLattice) -> Tuple[csr_matrix, csr_matrix]:
    """
    Computes the projection matrix for only nodes within the box
    """
    length, height = max([node.x for node in lattice.get_nodes()]), max([node.y for node in lattice.get_nodes()])
    box_l, box_r = 0.1 * length, 0.9 * length
    box_b, box_t = 0.1 * height, 0.9 * height
    box_nodes = [node for node in lattice.get_nodes() if box_l <= node.x <= box_r and box_b <= node.y <= box_t]
    n, r = len(lattice.get_nodes()), len(box_nodes)
    # Arrays to create the sparse projection matrix
    rows, cols = np.zeros(2 * r, dtype=np.uint32), np.zeros(2 * r, dtype=np.uint32)
    data = np.ones(2 * r, dtype=np.uint8)
    i = 0
    for node in lattice.get_nodes():
        x, y = node.get_xy()
        # Inside the box
        if box_l <= x <= box_r and box_b <= y <= box_t:
            rows[i], cols[i] = 2 * node.get_id(), 2 * i
            rows[r + i], cols[r + i] = 2 * node.get_id() + 1, 2 * i + 1
            i += 1
    p_box_to_n = csr_matrix((data, (rows, cols)), dtype=np.uint8, shape=(2 * n, 2 * r))
    return p_box_to_n, p_box_to_n.T.tocsr()


def get_force_mask(lattice: AbstractLattice) -> csr_matrix:
    """
    Computes the force mask to set forces to zero for boundary nodes
    """
    n = len(lattice.get_nodes())
    i_mask_lookup = np.zeros(2 * n, dtype=np.uint8)
    for node in lattice.get_nodes():
        if not node.is_boundary():
            i_mask_lookup[2 * node.get_id()] = 1
            i_mask_lookup[2 * node.get_id() + 1] = 1
    i_mask = diags(i_mask_lookup, dtype=np.uint8, shape=(2 * n, 2 * n))
    return i_mask


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


# def add_k_matrices(k1: KMatrixResult, k2: KMatrixResult) -> KMatrixResult:
#     def add_opt_matrix(m1: Optional[csr_matrix], m2: Optional[csr_matrix]) -> Optional[csr_matrix]:
#         if m1 is None:
#             return m2
#         elif m2 is None:
#             return m1
#         else:
#             return m1 + m2
#     return KMatrixResult(add_opt_matrix(k1.k_stretch, k2.k_stretch), add_opt_matrix(k1.k_bend, k2.k_bend),
#                          add_opt_matrix(k1.k_transverse, k2.k_transverse))


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
