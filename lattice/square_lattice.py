"""
Class to represent square lattices
"""
from typing import List

from lattice.abstract_lattice import AbstractLattice
from lattice.node import Node


class SquareLattice(AbstractLattice):
    def generate_nodes(self, length, height) -> List[Node]:
        """
        Generates the nodes within the SquareLattice

        :param length: number of nodes per column of lattice
        :type length: int
        :param height: height of the lattice
        :type height: float
        """
        h = 0
        node_id = 0
        while h < height:
            for i in range(1, length + 1, 1):
                node = Node(i, h, node_id)
                node_id += 1
                self.nodes.append(node)
            h += 1
        print("Generated " + str(len(self.nodes)) + " nodes of Square lattice")
        return self.nodes

    def generate_bonds(self) -> None:
        super().generate_bonds()

    def __init__(self, length: int, height: float):
        self.length = length
        self.height = height
        self.nodes = []
        self.bonds = []
        self.pi_bonds = []
        self.name = "Square"
        self.max_neighbors = 6
        self.height_increment = 1
        self.generate_nodes(length, height)
        self.generate_bonds()
        self.generate_pi_bonds()
