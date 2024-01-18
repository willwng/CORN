#!/usr/bin/env python3
"""
main.py takes no command-line arguments (they are taken through Parameters.py)
To run the code:
    '(python executable) main.py'
"""
import inspect
import time
from typing import Tuple, Optional, List

import numpy as np

import energyminimization.matrix_helper as pos
import energyminimization.minimize as em
import lattice.bond_basin_helper as basin
from energyminimization.minimize import MinimizationResult
from energyminimization.solvers.solver import ReusableResults
from lattice.abstract_lattice import AbstractLattice
from lattice.generic_lattice import make_generic
from lattice.hinged_lattice import add_hinges
from lattice.lattice_factory import LatticeFactory
from parameters import Parameters
from result_handling.output_handler import OutputHandler, PickleHandler
from visualization.visualize_lattice import Visualizer


def run_minimization(
        lattice: AbstractLattice,
        shear_strain: float,
        sheared_pos: np.ndarray,
        trans_matrix: np.ndarray,
        init_guess: Optional[np.ndarray],
        reusable_results: Optional[ReusableResults],
) -> MinimizationResult:
    """
    run_minimization finds the energy minimized state of the lattice after being sheared. Returns the list of shears
    used, final energy per shear, and final minimization result

    :param lattice: the lattice object
    :param shear_strain: amount of shear to apply to the lattice
    :param sheared_pos: sheared position of the lattice
    :param trans_matrix: transformation matrix
    :param init_guess: initial position guess for the energy minimization
    :param reusable_results: reusable results from the previous minimization
    """

    minimization_result = em.minimize(
        lattice=lattice,
        stretch_mod=Parameters.stretch_mod,
        bend_mod=Parameters.bend_mod,
        tran_mod=Parameters.tran_mod,
        shear_strain=shear_strain,
        sheared_pos=sheared_pos,
        trans_matrix=trans_matrix,
        init_guess=init_guess,
        tolerance=Parameters.tolerance,
        minimization_method=Parameters.minimization_method,
        reusable_results=reusable_results
    )
    print(f" > {minimization_result.info}")
    print(f" > Time used: {np.format_float_scientific(minimization_result.time_used)} s")

    # Print and append to arrays if using slope method
    print("Initial Energy: " + str(minimization_result.init_energy),
          " Final Energy " + str(minimization_result.final_energy))

    return minimization_result


def set_lattice_bonds(lattice: AbstractLattice, p_fill: float):
    """
    Sets the lattice bond generation
    (random, correlated (add/remove), polarized (add/remove))

    :param lattice: the lattice object
    :param p_fill: desired bond occupation probability
    :param p_fill: float (0 <= p <= 1)
    """
    # (random, correlated (add/remove), polarized (add/remove))
    lattice.set_bonds(prob_fill=p_fill, strength=Parameters.pc_strength)

    # Add a node/hinge in between each bond if hinged
    if Parameters.is_hinged:
        add_hinges(lattice)

    return


def initialize_lattice(set_bonds_active: bool):
    """
    Initializes the lattice object (either by loading a pickle file or creating a new lattice)
    """
    if Parameters.load_lattice:
        lattice = LatticeFactory.load_lattice_from_pickle(
            pickle_file=Parameters.load_lattice_pickle_file,
            generation_type=Parameters.bond_generation,
            set_bonds_active=set_bonds_active
        )
        print(f"Lattice loaded from {Parameters.load_lattice_pickle_file}")
    else:
        lattice = LatticeFactory.create_lattice(
            Parameters.lattice_type,
            Parameters.lattice_length,
            Parameters.lattice_height,
            Parameters.bond_generation,
            Parameters.bend_mod != 0,
        )

    # Pre-compute bond directions
    [bond.get_direction() for bond in lattice.get_bonds()]

    # Make the lattice generic if specified
    if Parameters.is_generic:
        make_generic(lattice=lattice, rng=get_rng(), d_shift=Parameters.d_shift)

    # Naive optimization: remove all the pi-bonds if we don't require bending:
    if Parameters.bend_mod == 0:
        lattice.drop_pi_bonds()

    return lattice


def get_moduli(
        lattice: AbstractLattice, init_guesses: Optional[List[np.ndarray]]
) -> Tuple[List[float], List[MinimizationResult]]:
    if init_guesses is None:
        init_guesses = [None] * 4
    init_pos = pos.create_pos_matrix(lattice=lattice)
    area_lattice = lattice.get_length() * lattice.get_height()

    shear_strain = Parameters.hor_shear
    transformation_matrices = pos.get_transformation_matrices(gamma=shear_strain)
    # --- Compute the response to each strain ---
    moduli, min_results = [], []
    reusable_results = None
    for i, trans_matrix in enumerate(transformation_matrices):
        sheared_pos = pos.transform_pos_matrix(pos_matrix=init_pos, transformation_matrix=trans_matrix)
        minimization_result = run_minimization(lattice=lattice, init_guess=init_guesses[i], shear_strain=shear_strain,
                                               sheared_pos=sheared_pos, trans_matrix=trans_matrix,
                                               reusable_results=reusable_results)
        reusable_results = minimization_result.reusable_results
        modulus = 2 * minimization_result.final_energy / (area_lattice * Parameters.hor_shear ** 2)
        print(f"Modulus for transformation {i}: {modulus}")
        moduli.append(modulus)
        min_results.append(minimization_result)
    return moduli, min_results


def get_rng() -> np.random.Generator:
    # Get a random number generator (make new seed if required)
    seed = Parameters.random_seed
    if seed is None:
        seed = np.random.SeedSequence().entropy
    rng = np.random.default_rng(seed)
    return rng


def update_output_file(
        lattice: AbstractLattice,
        moduli: List[float],
        minimization_results: List[MinimizationResult],
        output_handler: OutputHandler,
        bond_occupation: Optional[float] = None,
):
    """
    After each minimization, update the output file and potentially create final visualizations/pickles
    """
    if bond_occupation is None:
        bond_occupation = get_bond_occupation(lattice, disp=False)
    # Write the results to the output
    output_handler.add_results(lattice=lattice, moduli=moduli, bond_occupation=bond_occupation,
                               minimization_results=minimization_results)
    shear_mod_result = minimization_results[-1]
    output_handler.create_pickle_visualizations(folder_name=str(round(bond_occupation, 6)), lattice=lattice,
                                                sheared_pos=shear_mod_result.sheared_pos,
                                                final_pos=shear_mod_result.final_pos)


def get_bond_occupation(lattice, disp: bool = False) -> float:
    # Print the number of active bonds, the occupation will also be used for the folder name
    active_bonds, total_bonds = lattice.get_bond_occupation()
    bond_occupation = active_bonds / total_bonds
    if disp:
        print(f"\n--- Number of Active Bonds: {active_bonds}, p={bond_occupation} ---")
    return bond_occupation


def run_single_minimization(
        lattice: AbstractLattice,
        prev_final_pos: Optional[List[np.ndarray]],
        output_handler: OutputHandler,
        bond_occupation: Optional[float] = None,

) -> Tuple[List[float], List[MinimizationResult]]:
    get_bond_occupation(lattice, disp=True)  # Print out the bond occupation

    moduli, minimization_results = get_moduli(lattice=lattice, init_guesses=prev_final_pos)
    update_output_file(lattice=lattice, moduli=moduli,
                       minimization_results=minimization_results, output_handler=output_handler,
                       bond_occupation=bond_occupation)
    return moduli, minimization_results


def run_remove_bond_protocol(lattice: AbstractLattice, output_handler: OutputHandler):
    """
    Removes bonds from the lattice until the shear modulus is below the tolerance
    """
    # Start with a full lattice
    set_lattice_bonds(lattice=lattice, p_fill=1.0)
    lattice.update_active_bonds()

    bond_occupation = get_bond_occupation(lattice=lattice, disp=False)

    # Remove bonds until the occupation is below the starting probability
    while bond_occupation > Parameters.starting_prob_fill_remove:
        lattice.remove_bond(num_remove=Parameters.num_remove_bonds, strength=Parameters.pc_strength)
        bond_occupation = get_bond_occupation(lattice=lattice, disp=False)

    # Run the minimization for the base lattice
    lattice.update_active_bonds()
    shear_modulus_simple, minimization_results = run_single_minimization(lattice, None, output_handler)

    # Remove bonds until the shear modulus is below the tolerance
    prev_final_pos = [result.final_pos for result in minimization_results]
    while shear_modulus_simple >= Parameters.obj_tolerance:
        for _ in range(Parameters.num_remove_bonds):
            lattice.remove_bond(num_remove=Parameters.num_remove_bonds, strength=Parameters.pc_strength)
        shear_modulus_simple, minimization_results = run_single_minimization(lattice, prev_final_pos,
                                                                             output_handler)
        # Save the final position for initial guesses on the next iteration
        prev_final_pos = [result.final_pos for result in minimization_results]
    return


def run_add_bond_protocol(lattice: AbstractLattice, output_handler: OutputHandler):
    """
    Removes bonds from the lattice until the shear modulus is below the tolerance
    """
    # Start with an empty lattice
    set_lattice_bonds(lattice=lattice, p_fill=0.0)
    lattice.update_active_bonds()
    bond_occupation = get_bond_occupation(lattice=lattice, disp=False)

    # Add bonds until the occupation is above the starting probability
    while bond_occupation < Parameters.starting_prob_fill_add:
        lattice.add_bond(num_add=Parameters.num_add_bonds, strength=Parameters.pc_strength)
        bond_occupation = get_bond_occupation(lattice=lattice, disp=False)

    # Run minimization on the base lattice and remove bonds until final probability is met
    _, minimization_results = run_single_minimization(lattice, None, output_handler)
    prev_final_pos = [result.final_pos for result in minimization_results]
    while bond_occupation < Parameters.ending_prob_fill_add:
        lattice.add_bond(num_add=Parameters.num_add_bonds, strength=Parameters.pc_strength)
        bond_occupation = get_bond_occupation(lattice=lattice, disp=False)
        _, minimization_results = run_single_minimization(lattice, prev_final_pos, output_handler)
        prev_final_pos = [result.final_pos for result in minimization_results]
    return


def run_basin_protocol(lattice: AbstractLattice, output_handler: OutputHandler, add: bool, use_basin: bool):
    """
    Assigns each bond a random number s_i in (0, 1) and increases p: when p > s_i, we add the bond i
    """

    rng = get_rng()
    # Start with an empty lattice and assign each bond a random number in (0, 1)
    set_lattice_bonds(lattice=lattice, p_fill=0.0)
    basin.assign_bond_levels(lattice=lattice, rng=rng)
    lattice.update_active_bonds()

    # The threshold of p must be less than p_max, otherwise the threshold can be larger than 1
    r = Parameters.pc_strength
    p_max = min((1 / 3) + (2 / (3 * r)), Parameters.high_prob_fill_basin)

    # Gradually increase/decrease p (fill/drain the basin) until p_max or obj_tolerance is reached
    prev_final_pos = None
    p = Parameters.low_prob_fill_basin if add else p_max

    basin.set_bonds_basin(lattice=lattice, p=p, r=r, target_direction=Parameters.target_direction)
    removal_order = basin.get_removal_order(lattice=lattice, r=r, target_direction=Parameters.target_direction)
    while True:
        # Set the bond activity based on p, or addition/removal order
        if use_basin:
            basin.set_bonds_basin(lattice=lattice, p=p, r=r, target_direction=Parameters.target_direction)
        else:
            next_bond = removal_order.pop() if add else removal_order.popleft()
            next_bond.add_bond() if add else next_bond.remove_bond()
            lattice.update_active_bonds()

        moduli, minimization_results = run_single_minimization(lattice=lattice, prev_final_pos=prev_final_pos,
                                                               output_handler=output_handler, bond_occupation=p)
        prev_final_pos = [result.final_pos for result in minimization_results]

        # Update p based on either the occupation or the basin rule
        if use_basin:
            p = p + Parameters.p_delta if add else p - Parameters.p_delta
        else:
            p = get_bond_occupation(lattice=lattice, disp=False)
        # Termination conditions for adding: reach p_max. For removing: shear modulus is below tolerance
        if add and p > p_max:
            break
        elif not add and max(moduli) < Parameters.obj_tolerance:
            break
    return


def run_p_range_protocol(lattice: AbstractLattice, output_handler: OutputHandler):
    """
    Runs the minimization for all values of prob_no_bond
    """
    for p in Parameters.prob_fills:
        lattice.set_all_bonds_active()  # Reset all the bonds in case setting assumes a full lattice
        set_lattice_bonds(lattice, p)
        lattice.update_active_bonds()
        run_single_minimization(lattice, None, output_handler)
    return


def main():
    start_time = time.time()

    # Result are taken care of by the dedicated handlers
    visualizer = Visualizer(params=Parameters.visualizer_parameters)
    pickle_handler = PickleHandler(params=Parameters.pickle_handler_parameters, visualizer=visualizer)

    output_handler = OutputHandler(parameter_path=inspect.getfile(Parameters),
                                   params=Parameters.output_handler_parameters, pickle_handler=pickle_handler)

    # Initialize the lattice object
    lattice = initialize_lattice(set_bonds_active=True)

    # Slight hacky-fix. Remove bonds with double "imaginary positions" (i.e. bonds that are both periodic in x and y)
    imag_x = set([bond.get_node1() for bond in lattice.bonds if bond.is_hor_pbc()])
    imag_y = set([bond.get_node1() for bond in lattice.bonds if bond.is_top_pbc()])
    for bond in [bond for bond in lattice.get_bonds() if bond.is_hor_pbc() or bond.is_top_pbc()]:
        node_1, node_2 = bond.get_node1(), bond.get_node2()
        if node_1 in (imag_y.difference(imag_x)) and node_2 in imag_x:
            print(f"Removed double imaginary bond: {bond.get_node1().get_id()}, {bond.get_node2().get_id()}")
            lattice.drop_bond(bond)
        if node_1 in (imag_y.intersection(imag_x)) and node_2 not in (imag_y.union(imag_x)) and node_2.get_id() != 0:
            print(f"Removed double imaginary bond: {bond.get_node1().get_id()}, {bond.get_node2().get_id()}")
            lattice.drop_bond(bond)

    # If we are generating from scratch, save the lattice object for future re-use
    # if not Parameters.load_lattice:
    #     file_name = f"{lattice.get_description()}.pickle"
    #     output_handler.create_initial_lattice_object_pickle(lattice=lattice, file_name=file_name)

    # Run the appropriate protocol
    if Parameters.bond_occupation_protocol == 1:
        run_remove_bond_protocol(lattice, output_handler)
    elif Parameters.bond_occupation_protocol == 2:
        run_add_bond_protocol(lattice, output_handler)
    elif Parameters.bond_occupation_protocol == 3:
        run_basin_protocol(lattice, output_handler, Parameters.basin_add, Parameters.use_basin)
    elif Parameters.bond_occupation_protocol == 4:
        run_p_range_protocol(lattice, output_handler)
    else:
        print(f"Unknown bond occupation protocol: {Parameters.bond_occupation_protocol}")

    print(f"Total time used: {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
