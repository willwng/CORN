"""
Parameters.py is used for user input.
The following class contains variable assignments used in the code
"""
from typing import List, Optional
from datetime import datetime

from energyminimization.solvers.solver import MinimizationType
from result_handling.output_handler import OutputHandlerParameters
from result_handling.pickle_handler import PickleHandlerParameters
from visualization.visualize_lattice import VisualizerParameters
import numpy as np


class Parameters:
    # ----- Lattice Properties -----
    # --- Loading Lattice ---
    # If [load_lattice] is True then the lattice object will be loaded from [load_lattice_pickle_file]
    # Otherwise, a new lattice is created, and pickled into [save_lattice_pickle_file] if [create_pickle] is True
    load_lattice = False
    load_lattice_pickle_file = "prebuilt_lattices/triangular_100_101.pickle"

    # --- Generating new lattice ---
    # The following is for if the lattice is to be generated [load_lattice] = False)
    # Type of lattice:
    # (1) Kagome
    # (2) Triangular
    # (3) Square
    lattice_type: int = 2
    lattice_length: int = 50
    lattice_height: float = 50

    # Generic networks have slight displacements in node positions
    is_generic: bool = False
    d_shift: float = 0.0
    # Whether to add hinges between bonds - significantly adds to computation cost
    is_hinged: bool = False

    # ----- Bond Occupation Protocol -----
    # There are two protocols which are chosen based on [bond_occupation_protocol]:
    #   (1) Start a lattice with occupation [starting_prob_fill_remove] and remove [num_remove_bonds] bonds until the
    #       shear modulus is below [obj_tolerance]
    #   (2) Start a lattice with [starting_prob_fill_add] and add [num_add_bonds] bonds until bond occupation is
    #       [ending_prob_fill_add]
    #   (3) "Water Basin" approach: each bond is assigned a random number based on a seed. We then gradually
    #        increase p by [p_increment] and add the bond if it is greater than the seeded number
    #   (4) For each value in [prob_fills], reset the network and set the bond occupation to that value
    bond_occupation_protocol = 3
    # When we remove bonds, this is our stopping criteria: shear modulus is below this value
    obj_tolerance = 1e-8

    # --- Protocol 1 ---
    # Number of bonds to remove until shear modulus is zero
    num_remove_bonds: int = 50
    starting_prob_fill_remove: float = 1.0

    # --- Protocol 2 ---
    # Number of bonds to remove until shear modulus is zero
    num_add_bonds: int = 50
    starting_prob_fill_add: float = 0.3
    ending_prob_fill_add = 0.7

    # --- Protocol 3 ---
    random_seed: Optional[int] = 123
    # Whether to add bonds starting from [low_prob_fill_basin] or remove bonds starting from [high_prob_fill_basin]
    basin_add: bool = False
    # Whether we should fill basin with p, or simply add/remove bonds based on the order they would be added
    use_basin: bool = True
    p_delta: float = 0.01
    # If [ending_prob_fill_basin] is greater than the threshold for a thermodynamically legal network, then threshold
    low_prob_fill_basin: float = 0.5
    high_prob_fill_basin = .65

    # --- Protocol 4 ---
    prob_fills: List[float] = [1.0]

    # ----- Network generation -----
    # For oriented networks, which direction to prefer (0 = horizontal, 1 = right, 2 = left
    target_direction: int = 0

    # pc_strength is the correlation/polarized strength (0 <= pc_strength < 1)
    # Ignored if bond_generation is 0
    strengths = [1.0]
    pc_strength = strengths[0]

    # ----- Mechanical Properties -----
    # alpha, kappa, mu: stretch, bend, transverse moduli
    stretch_mod = 1
    bend_mod = 0.0
    tran_mod = 0

    # ----- Energy Minimization -----
    # Method to minimize the energy
    minimization_method: MinimizationType = MinimizationType.LINEAR_PRE
    # Minimization Tolerances: see minimization methods. For linear system this is norm(residual) <= tolerance
    tolerance = 1e-8

    # --- Lattice Shearing ---
    # Direction to shear the lattice (can be used as multiplier for [hor_shear]).
    shear_dir = 1
    # Shear strain (gamma). Each node is sheared by equation: delta_x = hor_shear * (height-lattice_height/2)
    hor_shear = 0.01

    # ----- Lattice pickling -----
    # All the parameters for handling pickling and visualization output
    pickle_handler_parameters = PickleHandlerParameters(
        create_lattice_pickle=False,
        create_final_pos_pickle=False,
        create_init_pdf=False,
        create_sheared_pdf=False,
        create_final_pdf=False,
        save_lattice_pickle_file="lattice_pickle.p",
        final_pos_pickle_file="final_pos.p",
        init_pos_pdf_file="init_pos.pdf",
        sheared_pos_pdf_file="sheared_pos.pdf",
        final_pos_pdf_file="final_pos.pdf"
    )

    # ----- Output Handling -----
    # (See OutputHandlerParameters for more details)
    today_date = datetime.now().strftime("%m-%d-%y-%H")
    run_folder_name: str = f"seed={str(random_seed)[:7]}-r={pc_strength}"

    output_handler_parameters = OutputHandlerParameters(
        inc_p=True,
        inc_shear_modulus=True,
        inc_ind_energies=True,
        inc_non_affinity=True,
        inc_bond_counts=True,
        inc_backbone_count=True,
        output_path="outputs",
        run_folder_name=run_folder_name,
        output_file="results.csv",
        save_parameters=True
    )

    # ----- Graphics and Postprocessing -----
    visualizer_parameters = VisualizerParameters(
        draw_nodes=True,
        draw_bonds=True,
        draw_pbc=True,
        node_color="black",
        bond_color="black",
        hor_shear=hor_shear
    )
