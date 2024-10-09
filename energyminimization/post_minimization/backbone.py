import numpy as np

import energyminimization.matrix_helper as pos
from energyminimization.energies.stretch_linear import get_stretch_energies
from lattice.abstract_lattice import AbstractLattice


class BackboneResult:
    n_stretch: int
    n_bend: int
    n_tran: int

    def __init__(self, n_stretch, n_bend, n_tran):
        self.n_stretch = n_stretch
        self.n_bend = n_bend
        self.n_tran = n_tran


def get_backbone_result(
        lattice: AbstractLattice,
        stretch_mod: float,
        final_pos: np.ndarray,
        r_matrix: np.ndarray,
        active_bond_indices: np.ndarray,
        tolerance: float,
) -> BackboneResult:
    init_pos = pos.create_pos_matrix(lattice)
    # Tolerance per bond (floppy vs rigid)
    num_bonds = len(lattice.get_active_bonds())
    bond_tol = tolerance / num_bonds

    u_matrix = (final_pos - init_pos).reshape(-1, 2)
    stretch_energies = get_stretch_energies(stretch_mod=stretch_mod, u_matrix=u_matrix, r_matrix=r_matrix,
                                            active_bond_indices=active_bond_indices)
    n_stretch = np.where(stretch_energies > bond_tol)[0].size

    return BackboneResult(n_stretch=n_stretch, n_bend=0, n_tran=0)
