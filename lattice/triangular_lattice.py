"""
Class to represent triangular lattices
"""
import math
from typing import List

from lattice.abstract_lattice import AbstractLattice
from lattice.node import Node


class TriangularLattice(AbstractLattice):
    def generate_nodes(self, length, height) -> List[Node]:
        """
        Generates the nodes within the TriangularLattice

        :param length: number of nodes per column of lattice
        :type length: int
        :param height: height of the lattice
        :type height: float
        """
        # Even Rows
        h = 0
        node_id = 0
        while h < height:
            for i in range(0, length, 1):
                node = Node(i, h, node_id)
                node_id += 1
                self.nodes.append(node)
            h += math.sqrt(3.0)
        # Odd rows
        h = math.sqrt(3) / 2
        while h < height:
            for i in range(0, length, 1):
                node = Node(i + 0.5, h, node_id)
                node_id += 1
                self.nodes.append(node)
            h += math.sqrt(3.0)
        print("Generated " + str(len(self.nodes)) + " nodes of Triangular lattice")
        max_height = max([node.y for node in self.nodes])
        self.height = max_height
        return self.nodes

    def generate_bonds(self) -> None:
        super().generate_bonds()

    def __init__(self, length: int, height: float, generate_pi_bonds: bool):
        self.length = length
        self.height = height
        self.nodes = []
        self.bonds = []
        self.pi_bonds = []
        self.name = "Triangular"
        self.max_neighbors = 10
        self.height_increment = math.sqrt(3.0) / 2.0
        self.generate_nodes(length, height)
        self.generate_bonds()
        if generate_pi_bonds:
            self.generate_pi_bonds()
