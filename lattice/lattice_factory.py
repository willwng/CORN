"""
Class used to generate specified lattices using a Factory design pattern
(This design pattern is used to increase abstraction by allowing for 
hiding/swapping implementation of lattices)
"""
import pickle
from typing import List

from lattice.abstract_lattice import AbstractLattice
from lattice.bond import Bond
from lattice.kagome_lattice import KagomeLattice
from lattice.node import Node
from lattice.pi_bond import PiBond
from lattice.square_lattice import SquareLattice
from lattice.triangular_lattice import TriangularLattice


class LatticeFactory:
    @staticmethod
    def create_lattice(
            lattice_type: int, length: int, height: float, generation_type: int, generate_pi_bonds: bool
    ) -> AbstractLattice:
        if lattice_type == 0:
            return KagomeLattice(120, 100.5, generation_type)
        elif lattice_type == 1:
            return KagomeLattice(length, height, generation_type)
        elif lattice_type == 2:
            return TriangularLattice(length, height, generation_type, generate_pi_bonds)
        elif lattice_type == 3:
            return SquareLattice(length, height, generation_type)
        else:
            print(f"Invalid type of lattice: {lattice_type}")
            quit()

    @staticmethod
    def load_lattice(node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data,
                     generation_type) -> AbstractLattice:
        new_lattice = KagomeLattice(0, 0, 0, generate=False)
        new_lattice.load_lattice(node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data)
        new_lattice.generation_type = generation_type
        return new_lattice

    @staticmethod
    def load_lattice_from_pickle(pickle_file: str, generation_type: int, set_bonds_active: bool) -> AbstractLattice:
        node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data = pickle.load(open(pickle_file, "rb"))
        lattice = LatticeFactory.load_lattice(node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data,
                                              generation_type=generation_type)
        if set_bonds_active:
            lattice.set_all_bonds_active()
        return lattice

    @staticmethod
    def create_from_base(length: int, height: float, generation_type: int, nodes: List[Node], bonds: List[Bond],
                         pi_bonds: List[PiBond]) -> AbstractLattice:
        new_lattice = KagomeLattice(length, height, generation_type, generate=False)
        new_lattice.nodes = nodes
        new_lattice.bonds = bonds
        new_lattice.pi_bonds = pi_bonds
        return new_lattice
