"""
minimize.py is used to get and minimize the energy of a lattice
"""
import time
from typing import List, Optional

import numpy as np

import energyminimization.matrix_helper as pos
import energyminimization.solvers.solver as solver
from energyminimization.post_minimization.backbone import BackboneResult, get_backbone_result
from energyminimization.post_minimization.non_affinity import get_non_affinity, NonAffinityResult
from energyminimization.transformations import Strain
from lattice.abstract_lattice import AbstractLattice


class MinimizationResult:
    init_energy: float
    final_energy: float
    sheared_pos: np.ndarray
    final_pos: np.ndarray
    individual_energies: List[float]
    non_affinity_result: NonAffinityResult
    backbone_result: BackboneResult
    reusable_results: solver.ReusableResults
    time_used: float
    info: str

    def __init__(self, init_energy, final_energy, sheared_pos, final_pos, individual_energies, reusable_results,
                 time_used, info):
        self.init_energy = init_energy
        self.final_energy = final_energy
        self.sheared_pos = sheared_pos
        self.final_pos = final_pos
        self.individual_energies = individual_energies
        # self.non_affinity_result = non_affinity_result
        # self.backbone_result = backbone_result
        self.reusable_results = reusable_results
        self.time_used = time_used
        self.info = info


def minimize(
        lattice: AbstractLattice,
        stretch_mod: float,
        bend_mod: float,
        tran_mod: float,
        strain: Strain,
        sheared_pos: np.ndarray,
        init_guess: Optional[np.ndarray],
        reusable_results: solver.ReusableResults,
        tolerance: float,
        minimization_method: solver.MinimizationType,
) -> MinimizationResult:
    """
    minimize finds the initial (sheared) and final energies of the lattice
    along with the final position. 'final' meaning the last iteration of the
    energy minimization algorithm

    :param lattice: the lattice to be minimized
    :type lattice: Instance of AbstractLattice class
    :param stretch_mod: modulus of stretching (alpha)
    :type stretch_mod: float
    :param bend_mod: bending modulus (kappa)
    :type bend_mod: float
    :param tran_mod: modulus of transverse (mu)
    :type tran_mod: float
    :param strain: strain applied to the lattice
    :param sheared_pos: position matrix containing the positions of each node
        after shearing
    :type sheared_pos: Matrix containing 2n elements (x, y for each node)
    :param init_guess: initial position guess
    :param reusable_results: objects from previous minimization
    :param tolerance: norm(residual) <= max(tol*norm(b), atol)
    :param minimization_method: type of energy minimization method to use
    """
    start_time = time.time()

    # Map from bond object to index (in active_bond_indices and r_matrix)
    bond_to_idx = {}

    # all_bond_indices[i] contains: [0: node_1 id, 1: node_2 id, 2: horizontal PBC, 3: top PBC, 4: index for r_matrix]
    all_bond_indices = np.zeros((len(lattice.get_bonds()), 5), dtype=np.int32)
    i_active = 0
    for i, bond in enumerate(lattice.get_bonds()):
        all_bond_indices[i][0] = bond.get_node1().get_id()
        all_bond_indices[i][1] = bond.get_node2().get_id()
        all_bond_indices[i][2] = bond.is_hor_pbc()
        all_bond_indices[i][3] = bond.is_top_pbc()
        if bond.exists():
            all_bond_indices[i][4] = i_active
            bond_to_idx[bond] = i_active
            i_active += 1
        else:
            all_bond_indices[i][4] = -1

    # Same as all_bond_indices but only for active bonds
    active_bond_indices = all_bond_indices[all_bond_indices[:, 4] != -1]

    # active_pi_indices[i] contains: [vertex id, edge 1 id, edge 2 id, bond 1 id, sign for r_matrix[bond id]]
    active_pi_bonds = lattice.get_active_pi_bonds()
    active_pi_indices = np.zeros((len(active_pi_bonds), 5), dtype=np.int32)
    for i, pi_bond in enumerate(active_pi_bonds):
        if pi_bond.exists():
            active_pi_indices[i][0] = pi_bond.get_vertex_node().get_id()
            active_pi_indices[i][1] = pi_bond.get_edge_nodes()[0].get_id()
            active_pi_indices[i][2] = pi_bond.get_edge_nodes()[1].get_id()
            active_pi_indices[i][3] = bond_to_idx[pi_bond.get_bond1()]
            active_pi_indices[i][4] = 1 if pi_bond.get_bond1().get_node1() == pi_bond.get_vertex_node() else -1

    # Initial position of nodes, initial unit vectors for bonds, initial energy
    init_pos = pos.create_pos_matrix(lattice)
    correction_matrix = pos.create_correction_matrix(lattice=lattice, init_pos=init_pos,
                                                     strain=strain, all_bond_indices=all_bond_indices)
    r_matrix = pos.create_r_matrix(pos_vector=init_pos, active_bond_indices=active_bond_indices, lattice=lattice,
                                   normalize=True)
    length_matrix = pos.create_r_matrix(pos_vector=init_pos, active_bond_indices=active_bond_indices, lattice=lattice,
                                        normalize=False)
    length_matrix = np.linalg.norm(length_matrix, axis=1)

    # Solve for the final relaxed position
    solve_params = solver.SolveParameters(lattice=lattice, init_pos=init_pos, sheared_pos=sheared_pos,
                                          init_guess=init_guess, r_matrix=r_matrix, correction_matrix=correction_matrix,
                                          length_matrix=length_matrix, active_bond_indices=active_bond_indices,
                                          active_pi_indices=active_pi_indices, stretch_mod=stretch_mod,
                                          bend_mod=bend_mod, tran_mod=tran_mod, tolerance=tolerance)
    solve_result = solver.solve(params=solve_params, minimization_type=minimization_method,
                                reusable_results=reusable_results)
    final_pos, individual_energies, info = solve_result.final_pos, solve_result.individual_energies, solve_result.info
    init_energy, final_energy = solve_result.init_energy, solve_result.final_energy

    # # Non-affinity parameter
    # non_affinity_result = get_non_affinity(lattice=lattice, final_pos=final_pos, shear_strain=shear_strain)
    #
    # # Backbone result
    # backbone_result = get_backbone_result(lattice=lattice, stretch_mod=stretch_mod, final_pos=final_pos,
    #                                       r_matrix=r_matrix, active_bond_indices=active_bond_indices,
    #                                       tolerance=tolerance)

    return MinimizationResult(init_energy=init_energy,
                              final_energy=final_energy,
                              sheared_pos=sheared_pos,
                              final_pos=final_pos,
                              individual_energies=individual_energies,
                              reusable_results=solve_result.reusable_results,
                              time_used=time.time() - start_time,
                              info=f"Converged with code: {info}")
