"""
This file is used to help handle the output file for runs
Keep anything that pertains the output file here for consistency
"""
import csv
import os
import shutil
from typing import List, Tuple, Union

import numpy as np

from energyminimization.minimize import MinimizationResult
from energyminimization.transformations import Strain
from lattice.abstract_lattice import AbstractLattice
from result_handling.result_helper import create_folder_if_not_exist, create_lattice_object_pickle
from result_handling.pickle_handler import VisualizationHandler


class OutputHandlerParameters:
    """ Parameters to be fed into the output handler """
    inc_p: bool
    inc_gamma: bool
    inc_energies: bool
    inc_ind_energies: bool
    inc_non_affinity: bool
    inc_bond_counts: bool
    inc_backbone_count: bool

    # Path of the folder to put run outputs
    output_path: str
    # Name of the folder to put run outputs ([output_path]/[folder_output]/)
    run_folder_name: str
    # Name of the output file to save each run data to (found in [folder_output])
    output_file: str
    # Whether to save these parameters within [folder_output]
    save_parameters = True

    def __init__(self, inc_p: bool, inc_p2: bool, inc_gamma: bool, inc_energies: bool, strains: list[Strain],
                 inc_ind_energies: bool, inc_non_affinity: bool, inc_bond_counts: bool, inc_backbone_count: bool,
                 output_path: str, run_folder_name: str, output_file: str, save_parameters: bool):
        self.inc_p = inc_p
        self.inc_p2 = inc_p2
        self.inc_gamma = inc_gamma
        self.inc_energies = inc_energies
        self.strains = strains
        self.inc_ind_energies = inc_ind_energies
        self.inc_non_affinity = inc_non_affinity
        self.inc_bond_counts = inc_bond_counts
        self.inc_backbone_count = inc_backbone_count
        self.output_path = output_path
        self.run_folder_name = run_folder_name
        self.output_file = output_file
        self.save_parameters = save_parameters


class OutputRow:
    """ A row in the output file """
    header: List[str] = []

    def __init__(self):
        self.header = []

    def append(self, val: any):
        self.header.append(val)

    def extend(self, ext: List[any]):
        self.header.extend(ext)

    def to_list(self):
        return self.header


class OutputHandler:
    output_file: str
    params: OutputHandlerParameters
    visualization_handler: VisualizationHandler

    def __init__(self, params: OutputHandlerParameters, parameter_path: str,
                 visualization_handler: VisualizationHandler):
        """
        Initialize the output handler
        :output_file: The name of the results file to be created
        :param params: Parameters for the output handler (see OutputHandlerParameters)
        :param parameter_path: The path of the parameters file (to main)
        """
        # Create the parent output folder if it doesn't exist
        run_folder_path = os.path.join(params.output_path, params.run_folder_name)
        create_folder_if_not_exist(params.output_path)
        create_folder_if_not_exist(run_folder_path)

        if params.save_parameters:
            shutil.copy(parameter_path, run_folder_path)

        self.output_file = os.path.join(run_folder_path, params.output_file)
        self.params = params
        self.visualization_handler = visualization_handler
        self._initialize_header()

    def get_run_folder(self) -> str:
        """ Returns the run folder path """
        return os.path.dirname(self.output_file)

    def _write_row(self, row: OutputRow):
        """ Write a row to the output file """
        with open(self.output_file, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row.to_list())
        return

    def _initialize_header(self):
        """ Initialize and write the header for the output file """
        header = OutputRow()
        if self.params.inc_gamma:
            header.append("gamma")
        if self.params.inc_p2:
            header.append("p1")
            header.append("p2")
        elif self.params.inc_p:
            header.append("p")
        if self.params.inc_energies:
            header.extend([strain.name for strain in self.params.strains])
        if self.params.inc_ind_energies:
            header.extend(["stretch", "bend", "transverse"])
        if self.params.inc_non_affinity:
            header.extend(["non-affinity", "non-affinity-x", "non-affinity-y"])
        if self.params.inc_bond_counts:
            header.extend(["horizontal", "right", "left"])
        if self.params.inc_backbone_count:
            header.extend(["back-str", "back-bend", "back-tran"])
        self._write_row(header)
        return

    def add_results(
            self,
            lattice: AbstractLattice,
            strain: Strain,
            minimization_result: MinimizationResult,
    ):
        """ Add the results of the minimization to the output file (typically one row) """
        row = OutputRow()

        # Write the results to the csv file
        if self.params.inc_gamma:
            row.append(strain.gamma)
        if self.params.inc_p2:
            assert type(lattice).__name__ == "DoubleTriangularLattice"
            p1, p2 = lattice.get_bond_occupations()
            row.extend([p1, p2])
        elif self.params.inc_p:
            active, total = lattice.get_bond_occupation()
            row.append(round(active / total, 6))
        if self.params.inc_energies:
            row.append(minimization_result.final_energy)
        if self.params.inc_ind_energies:
            row.extend(minimization_result.individual_energies)
        if self.params.inc_non_affinity:
            non_affinity = minimization_result.non_affinity_result
            row.extend([non_affinity.gamma, non_affinity.gamma_x, non_affinity.gamma_y])
        if self.params.inc_bond_counts:
            bond_directions = np.array([bond.get_direction() for bond in lattice.get_active_bonds()])
            directions, counts = np.unique(bond_directions, return_counts=True)
            # Subtract off the number of boundary bonds (should not be counted)
            row.extend(counts)
        if self.params.inc_backbone_count:
            backbone_result = minimization_result.backbone_result
            row.extend([backbone_result.n_stretch, backbone_result.n_bend, backbone_result.n_tran])

        self._write_row(row)
        return

    def create_pickle_visualizations(
            self,
            lattice: AbstractLattice,
            strain: Strain,
            sheared_pos: np.ndarray,
            final_pos: np.ndarray):
        """ Create pickle files for the lattice and minimization result """
        # Folder structure is {p}/{strain}-{gamma}
        p_folder_name = ""
        if type(lattice).__name__ == "DoubleTriangularLattice":
            p1, p2 = lattice.get_bond_occupations()
            p_folder_name += f"p1-{round(p1, 6)}_p2-{round(p2, 6)}"
        else:
            active, total = lattice.get_bond_occupation()
            p_folder_name += f"p-{round(active / total, 6)}"
        p_folder_name = os.path.join(self.get_run_folder(), p_folder_name)

        if self.visualization_handler.requires_visualization():
            create_folder_if_not_exist(p_folder_name)

        folder_name = os.path.join(p_folder_name, f"{strain.name}-{round(strain.gamma, 5)}")
        self.visualization_handler.create_pickle_visualizations(folder=folder_name, lattice=lattice,
                                                                sheared_pos=sheared_pos, final_pos=final_pos)
        return

    def create_initial_lattice_object_pickle(self, lattice: AbstractLattice, file_name: str):
        """ Create an initial pickle of the lattice object """
        create_lattice_object_pickle(lattice=lattice, pickle_path=os.path.join(self.get_run_folder(), file_name))

    def output_seed_file(self, seed_file_name, seed: int):
        """ Writes a seed (integer) to a file """
        seed_out_path = os.path.join(self.get_run_folder(), seed_file_name)
        with open(seed_out_path, "w") as f:
            f.write(str(seed))
            f.close()
        return
