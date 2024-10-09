#!/usr/bin/env python3
"""
main.py takes no command-line arguments (they are taken through Parameters.py)
To run the code:
    '(python executable) main.py'
"""
import inspect
from typing import Tuple, Optional, List

import numpy as np

import energyminimization.matrix_helper as pos
import energyminimization.minimize as em
import lattice.bond_basin_helper as basin
from energyminimization.minimize import MinimizationResult
from energyminimization.solvers.solver import ReusableResults
from energyminimization.transformations import Strain
from lattice.abstract_lattice import AbstractLattice
from lattice.generic_lattice import make_generic
from lattice.lattice_factory import LatticeFactory
from parameters import Parameters
from result_handling.output_handler import OutputHandler, VisualizationHandler
from visualization.visualize_lattice import Visualizer


def run_minimization(
        lattice: AbstractLattice,
        sheared_pos: np.ndarray,
        strain: Strain,
        init_guess: Optional[np.ndarray],
        reusable_results: Optional[ReusableResults],
) -> MinimizationResult:
    """
    run_minimization finds the energy minimized state of the lattice after being sheared. Returns the list of shears
    used, final energy per shear, and final minimization result

    :param lattice: the lattice object
    :param sheared_pos: sheared position of the lattice
    :param strain: the strain applied to the lattice
    :param init_guess: initial position guess for the energy minimization
    :param reusable_results: reusable results from the previous minimization
    """

    minimization_result = em.minimize(
        lattice=lattice,
        stretch_mod=Parameters.stretch_mod,
        bend_mod=Parameters.bend_mod,
        tran_mod=Parameters.tran_mod,
        sheared_pos=sheared_pos,
        strain=strain,
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


def initialize_lattice(rng: np.random.Generator) -> AbstractLattice:
    lattice = LatticeFactory.create_lattice(
        lattice_type=Parameters.lattice_type,
        length=Parameters.lattice_length,
        height=Parameters.lattice_height,
        generate_pi_bonds=Parameters.bend_mod != 0,  # Naive optimization: only make pi bonds if bending is relevant
    )

    # --- Initialization steps ---
    # Pre-compute bond directions
    [bond.get_direction() for bond in lattice.get_bonds()]

    # Make the lattice generic if specified
    if Parameters.is_generic:
        make_generic(lattice=lattice, rng=rng, d_shift=Parameters.d_shift)

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

    return lattice


def get_moduli(
        lattice: AbstractLattice, init_guesses: Optional[List[np.ndarray]]
) -> Tuple[List[float], List[MinimizationResult]]:
    # Get the strains
    strains = Parameters.strains

    # Prepare initial guesses
    if init_guesses is None:
        init_guesses = [None] * len(strains)
    # Set after the first minimization
    reusable_results = None

    init_pos = pos.create_pos_matrix(lattice=lattice)
    area_lattice = lattice.get_length() * lattice.get_height()

    # --- Compute the response to each strain ---
    min_results = []
    for i, strain in enumerate(strains):
        sheared_pos = strain.apply(pos_matrix=init_pos)
        minimization_result = run_minimization(
            lattice=lattice,
            init_guess=init_guesses[i],
            sheared_pos=sheared_pos,
            strain=strain,
            reusable_results=reusable_results
        )
        reusable_results = minimization_result.reusable_results
        # modulus = 2 * minimization_result.final_energy / (area_lattice * Parameters.gamma ** 2)
        # print(f"Modulus for transformation {i}: {modulus}")
        min_results.append(minimization_result)
    return min_results


def update_output_file(
        lattice: AbstractLattice,
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
    output_handler.add_results(lattice=lattice, bond_occupation=bond_occupation,
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

    minimization_results = get_moduli(lattice=lattice, init_guesses=prev_final_pos)
    update_output_file(lattice=lattice, minimization_results=minimization_results, output_handler=output_handler,
                       bond_occupation=bond_occupation)
    return minimization_results


def run_basin_protocol(lattice: AbstractLattice, output_handler: OutputHandler, rng: np.random.Generator):
    """
    Assigns each bond a random number s_i in (0, 1) and increases p: when p > s_i, we add the bond i
    """

    # Start with an empty lattice and assign each bond a random number in (0, 1)
    lattice.set_bonds(prob_fill=0.0)
    basin.assign_bond_levels(lattice=lattice, rng=rng)
    lattice.update_active_bonds()

    # The threshold of p must be less than p_max, otherwise the threshold can be larger than 1
    r = Parameters.r_strength
    p_max = min((1 / 3) + (2 / (3 * r)), Parameters.prob_fill_high)

    # Gradually increase/decrease p (fill/drain the basin) until p_max or obj_tolerance is reached
    prev_final_pos = None
    p = p_max

    basin.set_bonds_basin(lattice=lattice, p=p, r=r, target_direction=Parameters.target_direction)
    removal_order = basin.get_removal_order(lattice=lattice, r=r, target_direction=Parameters.target_direction)
    while True:
        # Set the bond activity removal order
        next_bond = removal_order.popleft()
        next_bond.remove_bond()
        lattice.update_active_bonds()

        minimization_results = run_single_minimization(lattice=lattice, prev_final_pos=prev_final_pos,
                                                               output_handler=output_handler, bond_occupation=p)
        prev_final_pos = [result.final_pos for result in minimization_results]

        # Update p based on either the occupation or the basin rule
        p = get_bond_occupation(lattice=lattice, disp=False)

        # Termination conditions for adding: reach p_max. For removing: shear modulus is below tolerance
        # if max(moduli) < Parameters.moduli_tolerance:
        #     break
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

    # Run the appropriate protocol
    run_basin_protocol(lattice=lattice, output_handler=output_handler, rng=rng)


if __name__ == "__main__":
    main()
