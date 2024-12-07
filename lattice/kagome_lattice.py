"""
Class to generate Kagome lattices
"""
import math
from typing import List

from lattice.abstract_lattice import AbstractLattice
from lattice.node import Node


class KagomeLattice(AbstractLattice):
    def generate_nodes(self, length: int, height: float) -> List[Node]:
        """
        Generates the nodes within the KagomeLattice

        :param length: number of nodes per column of lattice
        :type length: int
        :param height: height of the lattice
        :type height: float
        """
        # Even Rows
        h = 0
        node_id = 0
        while h < height:
            for i in range(1, length + 1, 1):
                node = Node(i, h, node_id)
                node_id += 1
                self.nodes.append(node)
            h += math.sqrt(3.0)
        # Odd rows
        h = math.sqrt(3) / 2
        alt = 0
        while h < height:
            alt = alt ^ 1
            for i in range(0, length, 2):
                node = Node(i + alt + 0.5, h, node_id)
                node_id += 1
                self.nodes.append(node)
            h += math.sqrt(3.0)
        print("Generated " + str(len(self.nodes)) + " nodes of Kagome lattice")
        return self.nodes

    def generate_bonds(self) -> None:
        super().generate_bonds()

    def __init__(self, length: int, height: float, generate: bool = True):
        self.length = length
        self.height = height
        self.nodes = []
        self.bonds = []
        self.pi_bonds = []
        self.name = "Kagome"
        self.height_increment = math.sqrt(3.0) / 2.0
        if generate:
            self.generate_nodes(length, height)
            self.generate_bonds()
            self.generate_pi_bonds()
