"""
This file is used to store helper functions for result handling
"""
import os
import pickle

from lattice.abstract_lattice import AbstractLattice


def create_folder_if_not_exist(path):
    """ Creates a folder at the given path if it doesn't already exist """
    if not os.path.exists(path):
        os.mkdir(path)


def create_lattice_object_pickle(lattice: AbstractLattice, pickle_path: str) -> None:
    node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data = lattice.get_lattice_data()
    pickle.dump(
        (node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data),
        open(pickle_path, "wb"),
    )
    return
