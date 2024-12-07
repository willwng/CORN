"""
This file is used to get and minimize the energy of a lattice.
    Lattice objects are converted to data types suitable for the solver (i.e., contiguous arrays).
"""
import time
from typing import List, Optional

import numpy as np

import energyminimization.matrix_helper as pos
import energyminimization.solvers.solver as solver
from energyminimization.post_minimization.backbone import BackboneResult
from energyminimization.post_minimization.non_affinity import NonAffinityResult
from energyminimization.transformations import Strain
from lattice.abstract_lattice import AbstractLattice


class MinimizationResult:
    """
    A result following a minimization
    """
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
    :param strain: strain applied to the lattice
    :param sheared_pos: position matrix containing the positions of each node
        after shearing
    :param init_guess: initial position guess
    :param reusable_results: objects from previous minimization
    :param tolerance: norm(residual) <= max(tol*norm(b), atol)
    :param minimization_method: type of energy minimization method to use
    """
    start_time = time.time()
    # --- Prepare solver input from lattice ---

    # 1. Create a tensor/array of relevant bond information
    #  [0: node_1 id, 1: node_2 id, 2: horizontal PBC, 3: top PBC, 4: index for r_matrix] for each bond
    all_bond_indices = np.zeros((len(lattice.get_bonds()), 5), dtype=np.int32)

    # Map from bond object to index (in active_bond_indices and r_matrix)
    bond_to_idx = {}
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
    ind_active = all_bond_indices[:, 4] != -1
    active_bond_indices = all_bond_indices[ind_active]

    # [vertex id, edge node 1 id, edge node 2 id, bond 1 id, bond 2 id, pi bond id, sign for r_matrix[bond id], horizontal PBC, top PBC]
    active_pi_bonds = lattice.get_active_pi_bonds()
    active_pi_indices = np.zeros((len(active_pi_bonds), 12), dtype=np.int32)
    for i, pi_bond in enumerate(active_pi_bonds):
        assert (pi_bond.exists())
        active_pi_indices[i][0] = pi_bond.get_vertex_node().get_id() # j
        active_pi_indices[i][1] = pi_bond.get_edge_nodes()[0].get_id() # i
        active_pi_indices[i][2] = pi_bond.get_edge_nodes()[1].get_id() # k
        active_pi_indices[i][3] = bond_to_idx[pi_bond.get_bond1()]
        active_pi_indices[i][4] = bond_to_idx[pi_bond.get_bond1()]
        active_pi_indices[i][5] = i
        active_pi_indices[i][6] = 1 if pi_bond.get_bond1().get_node1() == pi_bond.get_vertex_node() else -1 # sign 1
        active_pi_indices[i][7] = 1 if pi_bond.get_bond2().get_node1() == pi_bond.get_vertex_node() else -1 # sign 2
        # PBC
        active_pi_indices[i][8] = pi_bond.get_bond1().is_hor_pbc()
        active_pi_indices[i][9] = pi_bond.get_bond1().is_top_pbc()
        active_pi_indices[i][10] = pi_bond.get_bond2().is_hor_pbc()
        active_pi_indices[i][11] = pi_bond.get_bond2().is_top_pbc()


    # Stack all the moduli
    stretch_moduli = np.array([bond.stretch_mod for bond in lattice.get_bonds()], dtype=np.float64)
    tran_moduli = np.array([bond.tran_mod for bond in lattice.get_bonds()], dtype=np.float64)
    stretch_moduli = stretch_moduli[ind_active]
    tran_moduli = tran_moduli[ind_active]
    bend_moduli = np.array([pi_bond.bend_mod for pi_bond in active_pi_bonds], dtype=np.float64)

    # 2. Convert lattice positions to matrices
    # Initial position of nodes, initial unit vectors for bonds, initial energy
    init_pos = pos.create_pos_matrix(lattice)
    # Correction matrix is only needed for linear solves
    correction_matrix = pos.create_correction_matrix(lattice=lattice, init_pos=init_pos,
                                                     strain=strain, all_bond_indices=all_bond_indices)
    r_matrix = pos.create_r_matrix(pos_vector=init_pos, active_bond_indices=active_bond_indices, lattice=lattice,
                                   normalize=True)
    length_matrix = pos.create_r_matrix(pos_vector=init_pos, active_bond_indices=active_bond_indices, lattice=lattice,
                                        normalize=False)
    length_matrix = np.linalg.norm(length_matrix, axis=1)

    angle_matrix = pos.create_angle_matrix(pos_vector=init_pos, active_pi_indices=active_pi_indices, lattice=lattice)
    # 3. Solve for the final relaxed position
    solve_params = solver.SolveParameters(lattice=lattice, strain=strain, init_pos=init_pos, strained_pos=sheared_pos,
                                          init_guess=init_guess, r_matrix=r_matrix, correction_matrix=correction_matrix,
                                          length_matrix=length_matrix, angle_matrix=angle_matrix,
                                          active_bond_indices=active_bond_indices, active_pi_indices=active_pi_indices,
                                          stretch_mod=stretch_moduli, bend_mod=bend_moduli, tran_mod=tran_moduli,
                                          tolerance=tolerance)
    solve_result = solver.solve(params=solve_params, minimization_type=minimization_method,
                                reusable_results=reusable_results)

    # Return final results
    final_pos, individual_energies, info = solve_result.final_pos, solve_result.individual_energies, solve_result.info
    init_energy, final_energy = solve_result.init_energy, solve_result.final_energy

    # Uncomment below for additional post-minimization results
    # # Non-affinity parameter
    # non_affinity_result = get_non_affinity(lattice=lattice, final_pos=final_pos, shear_strain=shear_strain)
    #
    # # Backbone result
    # backbone_result = get_backbone_result(lattice=lattice, stretch_mod=stretch_mod, final_pos=final_pos,
    #                                       r_matrix=r_matrix, active_bond_indices=active_bond_indices,
    #                                       tolerance=tolerance)

    print(f" > Converged with code: {info}")
    print(f" > Time used: {np.format_float_scientific(time.time() - start_time)} s")
    print(f" > Initial E: {init_energy}, Final E: {final_energy}")

    return MinimizationResult(init_energy=init_energy,
                              final_energy=final_energy,
                              sheared_pos=sheared_pos,
                              final_pos=final_pos,
                              individual_energies=individual_energies,
                              reusable_results=solve_result.reusable_results,
                              time_used=time.time() - start_time,
                              info=f"Converged with code: {info}")
