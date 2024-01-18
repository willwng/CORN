"""
Class used to generate specified lattices using a Factory design pattern
(This design pattern is used to increase abstraction by allowing for 
hiding/swapping implementation of lattices)
"""
import pickle

from lattice.abstract_lattice import AbstractLattice
from lattice.kagome_lattice import KagomeLattice
from lattice.square_lattice import SquareLattice
from lattice.triangular_lattice import TriangularLattice


class LatticeFactory:
    @staticmethod
    def create_lattice(
            lattice_type: int, length: int, height: float, generate_pi_bonds: bool
    ) -> AbstractLattice:
        """
        Create a fresh lattice from scratch
        """
        if lattice_type == 1:
            return KagomeLattice(length=length, height=height)
        elif lattice_type == 2:
            return TriangularLattice(length=length, height=height, generate_pi_bonds=generate_pi_bonds)
        elif lattice_type == 3:
            return SquareLattice(length=length, height=height)
        else:
            print(f"Invalid type of lattice: {lattice_type}")
            quit()

    @staticmethod
    def load_lattice(
            node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data, generation_type
    ) -> AbstractLattice:
        """
        Helper method to load a lattice from a pickle file
        """
        new_lattice = KagomeLattice(length=0, height=0, generate=False)
        new_lattice.load_lattice(node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data)
        new_lattice.generation_type = generation_type
        return new_lattice

    @staticmethod
    def load_lattice_from_pickle(
            pickle_file: str, generation_type: int, set_bonds_active: bool
    ) -> AbstractLattice:
        """
        Load a lattice from a pickle file
        """
        node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data = pickle.load(open(pickle_file, "rb"))
        lattice = LatticeFactory.load_lattice(node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data,
                                              generation_type=generation_type)
        if set_bonds_active:
            lattice.set_all_bonds_active()
        return lattice
