#!/usr/bin/env python3
"""
main.py takes no command-line arguments (they are taken through Parameters.py)
To run the code:
    '(python executable) main.py'
"""
import inspect

import numpy as np

import energyminimization.matrix_helper as pos
import energyminimization.minimize as em
import lattice.bond_basin_helper as basin
from energyminimization.minimize import MinimizationResult
from lattice.abstract_lattice import AbstractLattice
from lattice.generic_lattice import make_generic
from lattice.lattice_factory import LatticeFactory
from parameters import Parameters
from protocol_types import Protocol
from result_handling.output_handler import OutputHandler, VisualizationHandler
from visualization.visualize_lattice import Visualizer


def initialize_lattice(rng: np.random.Generator) -> AbstractLattice:
    lattice = LatticeFactory.create_lattice(
        lattice_type=Parameters.lattice_type,
        length=Parameters.lattice_length,
        height=Parameters.lattice_height,
    )

    # --- Initialization steps ---
    # Pre-compute bond directions
    [bond.get_direction() for bond in lattice.get_bonds()]

    # Make the lattice generic if specified
    if Parameters.is_generic:
        make_generic(lattice=lattice, rng=rng, d_shift=Parameters.d_shift)

    # Patches any potential issues with periodic boundary conditions
    lattice.patch_pbc()

    # Set the mechanical properties
    lattice.set_stretch_mod(Parameters.stretch_mod, Parameters.stretch_mod2)
    lattice.set_bend_mod(Parameters.bend_mod, Parameters.bend_mod2)
    lattice.set_tran_mod(Parameters.tran_mod, Parameters.tran_mod2)
    return lattice


def update_output_file(
        lattice: AbstractLattice,
        minimization_results: list[MinimizationResult],
        output_handler: OutputHandler,
        bond_occupation: float
):
    """
    After each minimization, update the output file and potentially create final visualizations/pickles
    """
    # Write the results to the output
    output_handler.add_results(lattice=lattice, bond_occupation=bond_occupation,
                               minimization_results=minimization_results)
    last_result = minimization_results[-1]
    output_handler.create_pickle_visualizations(folder_name=str(round(bond_occupation, 6)), lattice=lattice,
                                                sheared_pos=last_result.sheared_pos, final_pos=last_result.final_pos)


def check_terminate_protocol(minimization_results: list[MinimizationResult]) -> bool:
    """
    This function determines whether the protocol should terminate.
    Current implementation checks when all moduli are below the tolerance
    """
    area_lattice = Parameters.lattice_length * Parameters.lattice_height
    moduli = [2 * result.final_energy / (area_lattice * Parameters.gamma ** 2) for result in minimization_results]
    return all([modulus < Parameters.moduli_tolerance for modulus in moduli])


def run_strain_sweep_protocol(lattice: AbstractLattice, output_handler: OutputHandler):
    init_pos = pos.create_pos_matrix(lattice=lattice)

    # Set the number of bonds in the lattice to p * total_bonds
    r = Parameters.r_strength
    p = min((1 / 3) + (2 / (3 * r)), Parameters.prob_fill_high)  # max allowable by r, or user-set value
    p2 = min((1 / 3) + (2 / (3 * r)), Parameters.prob_fill_high2)
    if type(lattice).__name__ == "DoubleTriangularLattice":
        # For the double network, set the bonds individually
        basin.set_bonds_basin(lattice=lattice.network1, p=p, r=r, target_direction=Parameters.target_direction)
        basin.set_bonds_basin(lattice=lattice.network2, p=p2, r=r, target_direction=Parameters.target_direction)
    else:
        basin.set_bonds_basin(lattice=lattice, p=p, r=r, target_direction=Parameters.target_direction)
    lattice.update_active_bonds()

    # All types of strains, initial guesses for each one
    strains = Parameters.strains
    init_guesses = [None] * len(strains)

    # Sweep through all strain amounts
    for gamma in Parameters.gammas:
        # Run minimization
        reusable_results = None
        # --- Compute the response to each strain ---
        minimization_results = []
        for i, strain in enumerate(strains):
            strain.update_gamma(gamma)
            print(f"-- Performing strain: {strain.name} with magnitude {strain.gamma} --")
            sheared_pos = strain.apply(pos_matrix=init_pos)
            minimization_result = em.minimize(
                lattice=lattice,
                sheared_pos=sheared_pos,
                strain=strain,
                init_guess=init_guesses[i],
                tolerance=Parameters.tolerance,
                minimization_method=Parameters.minimization_method,
                reusable_results=reusable_results
            )
            minimization_results.append(minimization_result)
            # Results that can be re-used across the same lattice & p
            reusable_results = minimization_result.reusable_results
            # Initial guess for the next minimization with this strain
            init_guesses[i] = minimization_result.final_pos

        update_output_file(lattice=lattice, minimization_results=minimization_results, output_handler=output_handler,
                           bond_occupation=gamma)
    return


def run_removal_protocol(lattice: AbstractLattice, output_handler: OutputHandler):
    """
    Assigns each bond a random number s_i in (0, 1) and increases p: when p > s_i, we add the bond i
    """
    init_pos = pos.create_pos_matrix(lattice=lattice)

    # Start with an empty lattice and assign each bond a random number in (0, 1)
    lattice.set_bonds(prob_fill=0.0)
    lattice.update_active_bonds()

    # The threshold of p must be less than p_max, otherwise the threshold can be larger than 1
    r = Parameters.r_strength
    p_max = min((1 / 3) + (2 / (3 * r)), Parameters.prob_fill_high)

    # Set the occupancy to p_max and get the bond removal order
    basin.set_bonds_basin(lattice=lattice, p=p_max, r=r, target_direction=Parameters.target_direction)
    removal_order = basin.get_removal_order(lattice=lattice, r=r, target_direction=Parameters.target_direction)

    # All types of strains, initial guesses for each one
    strains = Parameters.strains
    init_guesses = [None] * len(strains)

    # Gradually increase/decrease p (fill/drain the basin) until p_max or obj_tolerance is reached
    p = p_max
    while p > 0:
        # Remove the bond next in queue
        next_bond = removal_order.popleft()
        lattice.remove_bond(next_bond)

        # Get the bond occupation (right now use the actual p value)
        active_bonds, total_bonds = lattice.get_bond_occupation()
        p = active_bonds / total_bonds
        print(f"\n--- Number of Active Bonds: {active_bonds}, p={p} ---")

        # Run minimization
        reusable_results = None
        # --- Compute the response to each strain ---
        minimization_results = []
        for i, strain in enumerate(strains):
            print(f" Performing strain: {strain.name} --")
            sheared_pos = strain.apply(pos_matrix=init_pos)
            minimization_result = em.minimize(
                lattice=lattice,
                sheared_pos=sheared_pos,
                strain=strain,
                init_guess=init_guesses[i],
                tolerance=Parameters.tolerance,
                minimization_method=Parameters.minimization_method,
                reusable_results=reusable_results
            )
            minimization_results.append(minimization_result)
            # Results that can be re-used across the same lattice & p
            reusable_results = minimization_result.reusable_results
            # Initial guess for the next minimization with this strain
            init_guesses[i] = minimization_result.final_pos

        update_output_file(lattice=lattice, minimization_results=minimization_results, output_handler=output_handler,
                           bond_occupation=p)

        # Check termination of protocol
        if check_terminate_protocol(minimization_results):
            break
    return


def main():
    # Get the random number generator (make new seed if required)
    seed = Parameters.random_seed
    rng = np.random.default_rng(seed)

    # Result are taken care of by the dedicated handlers
    visualizer = Visualizer(params=Parameters.visualizer_parameters)
    visualization_handler = VisualizationHandler(params=Parameters.pickle_handler_parameters, visualizer=visualizer)
    output_handler = OutputHandler(parameter_path=inspect.getfile(Parameters),
                                   params=Parameters.output_handler_parameters,
                                   visualization_handler=visualization_handler)

    # Initialize the lattice object
    lattice = initialize_lattice(rng=rng)
    basin.assign_bond_seeds(lattice=lattice, rng=rng)  # assign a random number/key to each bond

    # Run the appropriate protocol
    if Parameters.protocol == Protocol.BOND_REMOVAL:
        run_removal_protocol(lattice=lattice, output_handler=output_handler)
    elif Parameters.protocol == Protocol.STRAIN_SWEEP:
        run_strain_sweep_protocol(lattice=lattice, output_handler=output_handler)
    else:
        raise ValueError(f"Invalid protocol: {Parameters.protocol}")


if __name__ == "__main__":
    main()
