import numpy as np

import energyminimization.matrix_helper as pos
from lattice.abstract_lattice import AbstractLattice


class NonAffinityResult:
    gamma: float
    gamma_x: float
    gamma_y: float

    def __init__(self, gamma, gamma_x, gamma_y):
        self.gamma = gamma
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y


def get_non_affinity(
        lattice: AbstractLattice,
        final_pos: np.ndarray,
        shear_strain: float,
) -> NonAffinityResult:
    """
    Returns the non-affine parameter
    """
    num_nodes = len(lattice.get_nodes())
    init_pos = pos.create_pos_matrix(lattice)

    # Non-affinity constants
    l_0 = 1  # Rest length of bond
    gamma_c = (1 / (num_nodes * shear_strain ** 2 * l_0 ** 2))
    u_matrix = (final_pos - init_pos).reshape(-1, 2)
    u2_matrix = np.square(np.linalg.norm(u_matrix, axis=1))

    # Non-affinity parameter, and its net x and y components
    gamma = gamma_c * np.sum(u2_matrix)
    gamma_x = gamma_c * np.sum(u_matrix[:, 0])
    gamma_y = gamma_c * np.sum(u_matrix[:, 1])
    return NonAffinityResult(gamma, gamma_x, gamma_y)
