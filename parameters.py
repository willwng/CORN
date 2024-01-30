
"""
Parameters.py is used for user input.
The following class contains variable assignments used in the code
"""
from datetime import datetime
from typing import Optional

import numpy as np

from energyminimization.solvers.solver import MinimizationType
from result_handling.output_handler import OutputHandlerParameters
from result_handling.pickle_handler import PickleHandlerParameters
from visualization.visualize_lattice import VisualizerParameters


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
    lattice_length: int = 15
    lattice_height: float = 15

    # Generic networks have slight displacements in node positions
    is_generic: bool = False
    d_shift: float = 0.01

    # ----- Network generation -----
    # For oriented networks, which direction to prefer (0 = horizontal, 1 = right, 2 = left
    target_direction: int = 0
    # pc_strength is the orientation strength "r" (0 <= pc_strength < 1)
    strengths = [1.0, 1.1, 1.2, 1.3, 1.4]
    pc_strength = strengths[4]

    # ----- Bond Occupation Protocol -----
    # Random seed for reproducibility
    random_seed: Optional[int] = np.random.SeedSequence().entropy
    # Starting p_fill value
    prob_fill_high = 0.68
    # When we remove bonds, this is our stopping criteria: all the moduli are below this value
    obj_tolerance = 1e-8

    # ----- Mechanical Properties -----
    # alpha, kappa, mu: stretch, bend, transverse moduli
    stretch_mod = 1
    bend_mod = 0.0
    tran_mod = 0

    # --- Lattice Shearing ---
    # Shear strain (gamma). Each node is sheared by equation: delta_x = hor_shear * (height-lattice_height/2)
    hor_shear = 0.001

    # ----- Energy Minimization -----
    # Method to minimize the energy
    minimization_method: MinimizationType = MinimizationType.LINEAR
    # Minimization Tolerances: see minimization methods. For linear systems this is usually norm(residual) <= tolerance
    tolerance = 1e-8

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
    run_folder_name: str = f"{lattice_length}-{str(random_seed)[:7]}-{pc_strength}"

    output_handler_parameters = OutputHandlerParameters(
        inc_p=True,
        inc_shear_modulus=True,
        inc_ind_energies=False,
        inc_non_affinity=False,
        inc_bond_counts=False,
        inc_backbone_count=False,
        output_path="outputs",
        run_folder_name=run_folder_name,
        output_file="results.csv",
        save_parameters=True
    )

    # ----- Graphics and Postprocessing -----
    visualizer_parameters = VisualizerParameters(
        draw_nodes=True,
        draw_bonds=True,
        draw_pbc=False,
        node_color="black",
        bond_color="black",
        hor_shear=hor_shear
    )
