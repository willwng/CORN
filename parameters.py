"""
Parameters.py is used for user input.
The following class contains variable assignments used in the code
"""
from datetime import datetime

import numpy as np

from energyminimization.solvers.solver import MinimizationType
from energyminimization.transformations import StretchX, StretchY
from lattice.lattice_type import LatticeType
from result_handling.output_handler import OutputHandlerParameters
from result_handling.pickle_handler import VisualizationHandlerParameters
from visualization.visualize_lattice import VisualizerParameters


class Parameters:
    # Random seed for reproducibility (take first 8 digits of entropy - we don't need the full amount)
    random_seed: int = np.random.SeedSequence().entropy
    random_seed = int(str(random_seed)[0:8])

    # ----- Lattice Properties -----
    lattice_type: LatticeType = LatticeType.TRIANGULAR
    lattice_length: int = 30
    lattice_height: float = 31

    # Generic networks have slight displacements in node positions
    is_generic: bool = False
    d_shift: float = 0.01

    # ----- Network generation -----
    # For oriented networks, which direction to prefer (0 = horizontal, 1 = right, 2 = left)
    target_direction: int = 0
    # How much to prefer the target direction (1 = isotropic)
    r_strength: float = 1.0

    # ----- Bond Occupation Protocol -----
    # Starting p_fill value
    prob_fill_high: float = 0.68
    # When we remove bonds, this is our stopping criteria: all the moduli are below this value
    moduli_tolerance: float = 1e-8

    # ----- Mechanical Properties -----
    # alpha, kappa, mu: stretch, bend, transverse moduli
    stretch_mod: float = 1.0
    bend_mod: float = 0.01
    tran_mod: float = 0.0

    # --- Lattice Strain ---
    # Magnitude of the strain
    gamma: float = 0.001
    strains = [StretchX(gamma=gamma), StretchY(gamma=gamma)]

    # ----- Energy and Minimization -----
    # Method to minimize the energy
    minimization_method: MinimizationType = MinimizationType.LINEAR
    # Minimization Tolerances: see minimization methods. For linear systems this is usually norm(residual) <= tolerance
    tolerance = 1e-8

    # ----- Output Handling -----
    # (See OutputHandlerParameters for more details)
    today_date = datetime.now().strftime("%m-%d-%y-%H")
    run_folder_name: str = f"{lattice_length}-{str(random_seed)}-{r_strength}"
    output_handler_parameters = OutputHandlerParameters(
        strains=strains,
        inc_p=True,
        inc_energies=True,
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
        hor_shear=gamma
    )

    # ----- Saving Visualization Files -----
    pickle_handler_parameters = VisualizationHandlerParameters(
        create_init_pdf=False,
        create_sheared_pdf=False,
        create_final_pdf=False,
        save_lattice_pickle_file="lattice_pickle.p",
        final_pos_pickle_file="final_pos.p",
        init_pos_pdf_file="init_pos.pdf",
        sheared_pos_pdf_file="sheared_pos.pdf",
        final_pos_pdf_file="final_pos.pdf"
    )
