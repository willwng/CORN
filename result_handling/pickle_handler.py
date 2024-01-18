import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from lattice.abstract_lattice import AbstractLattice
from result_handling.result_helper import create_folder_if_not_exist, create_lattice_object_pickle
from visualization.energy_density import create_voronoi
from visualization.visualize_lattice import Visualizer
import energyminimization.matrix_helper as pos


class PickleHandlerParameters:
    # Whether to create a pickle file of the lattice object, final position matrix
    create_lattice_pickle: bool
    create_final_pos_pickle: bool

    # Whether to create a visualization of the following positions
    create_init_pdf: bool
    create_sheared_pdf: bool
    create_final_pdf: bool

    # Paths for saving the lattice object pickle, final position matrix pickle, and the pdfs
    save_lattice_pickle_file: str
    final_pos_pickle_file: str
    init_pos_pdf_file: str
    sheared_pos_pdf_file: str
    final_pos_pdf_file: str

    def __init__(self, create_lattice_pickle: bool, create_final_pos_pickle: bool, create_init_pdf: bool,
                 create_sheared_pdf: bool, create_final_pdf: bool, save_lattice_pickle_file: str,
                 final_pos_pickle_file: str, init_pos_pdf_file: str, sheared_pos_pdf_file: str,
                 final_pos_pdf_file: str):
        self.create_lattice_pickle = create_lattice_pickle
        self.create_final_pos_pickle = create_final_pos_pickle
        self.create_init_pdf = create_init_pdf
        self.create_sheared_pdf = create_sheared_pdf
        self.create_final_pdf = create_final_pdf
        self.save_lattice_pickle_file = save_lattice_pickle_file
        self.final_pos_pickle_file = final_pos_pickle_file
        self.init_pos_pdf_file = init_pos_pdf_file
        self.sheared_pos_pdf_file = sheared_pos_pdf_file
        self.final_pos_pdf_file = final_pos_pdf_file


class PickleHandler:
    """ Handles pickling and visualization of the lattice and other objects """
    params: PickleHandlerParameters
    visualizer: Visualizer

    def __init__(self, params: PickleHandlerParameters, visualizer: Visualizer):
        self.params = params
        self.visualizer = visualizer

    def _requires_folder(self) -> bool:
        return any([self.params.create_lattice_pickle, self.params.create_init_pdf, self.params.create_sheared_pdf,
                    self.params.create_final_pdf, self.params.create_final_pos_pickle])

    def _requires_visualization(self) -> bool:
        return any([self.params.create_init_pdf, self.params.create_sheared_pdf, self.params.create_final_pdf])

    def create_pickle_visualizations(
            self, folder: str,
            lattice: AbstractLattice,
            sheared_pos: np.ndarray,
            final_pos: np.ndarray,
    ) -> None:
        """
        Creates the output folder along with the lattice visualizations and pickles will be saved in
        :param folder: The folder to save the outputs in
        :param lattice: the lattice object
        :type lattice: instance of AbstractLattice or a class that inherits it
        :param sheared_pos: sheared position matrix/vector of lattice
        :param final_pos: final position matrix/vector of lattice
        """
        # Create the folder if required (and it doesn't exist)
        if self._requires_folder():
            create_folder_if_not_exist(folder)

        # Pickle of the lattice object (includes all the data needed to recreate the same lattice)
        if self.params.create_lattice_pickle:
            create_lattice_object_pickle(
                lattice=lattice,
                pickle_path=os.path.join(folder, self.params.save_lattice_pickle_file)
            )
        # Pickle the final output position
        if self.params.create_final_pos_pickle:
            pickle.dump(final_pos, open(os.path.join(folder, f"{self.params.final_pos_pickle_file}"), "wb"))
        # Visualizations of initial, sheared, final states
        if self.params.create_init_pdf:
            init_pos = pos.create_pos_matrix(lattice)
            self.visualizer.visualize(lattice, init_pos.ravel(),
                                      filename=os.path.join(folder, self.params.init_pos_pdf_file))
        if self.params.create_sheared_pdf:
            self.visualizer.visualize(lattice, sheared_pos.ravel(),
                                      filename=os.path.join(folder, self.params.sheared_pos_pdf_file))
        if self.params.create_final_pdf:
            self.visualizer.visualize(lattice, final_pos.ravel(),
                                      filename=os.path.join(folder, self.params.final_pos_pdf_file))

        # Close all figures (should save on some memory)
        if self._requires_visualization():
            plt.close("all")
