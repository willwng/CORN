"""
Class representing Pi Bonds in lattice, a series of two co-linear bonds with a shared vertex node
PiBonds two pointers to each of the bonds, as well as three pointers to the vertex, and two edge nodes
"""
from typing import Tuple

from lattice.bond import Bond
from lattice.node import Node


class PiBond(object):
    # Instance Variables:
    # bond_1, bond_2 : Two bonds of the pi bond
    # vertex : Vertex node of the pi bond.
    # edge1, edge2 : Other two nodes in the pi bond
    # bend_mod : The bending modulus of the pi bond
    # The use of slots should save a bit on memory usage
    __slots__ = ["bond_1", "bond_2", "vertex", "edge1", "edge2", "bend_mod"]
    bond_1: Bond
    bond_2: Bond
    vertex: Node
    edge1: Node
    edge2: Node
    bend_mod: float

    def __init__(self, b_1: Bond, b_2: Bond, v: Node, e1: Node, e2: Node):
        self.bond_1 = b_1
        self.bond_2 = b_2
        self.vertex = v
        self.edge1 = e1
        self.edge2 = e2

    def get_bond1(self) -> Bond:
        """
        Returns the first bond of this pi-bond
        """
        return self.bond_1

    def get_bond2(self) -> Bond:
        """
        Returns the second bond of this pi-bond
        """
        return self.bond_2

    def get_vertex_node(self) -> Node:
        """
        Returns the vertex of this pi-bond
        """
        return self.vertex

    def get_edge_nodes(self) -> tuple[Node, Node]:
        """
        Returns a tuple containing the edge nodes of this pi-bond
        """
        return self.edge1, self.edge2

    def set_bend_mod(self, bend_mod: float) -> None:
        """
        Sets the bending modulus of this pi-bond
        """
        self.bend_mod = bend_mod

    def exists(self) -> bool:
        """
        Returns whether this pi-bond exists (both bonds exist)
        """
        assert self.bond_1 is not None
        assert self.bond_2 is not None
        return self.bond_1.exists() and self.bond_2.exists()
